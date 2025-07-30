"""
This file performs anomaly detection on a list of nodes using pre-trained Transformer models and a MTS feature extraction library (TSFEL). 

It first divides the MTS data for each node into segments based on job execution times. 
For each segment, it uses TSFEL to extract features and determine the most likely cluster that the segment belongs to based on pre-calculated cluster centers. 
It then loads the corresponding pre-trained Transformer model for that cluster and performs anomaly detection on the segment. 
The results from all segments are then combined to produce a final anomaly score for the entire MTS of the node.
"""
import os
import time
import json
import argparse
import tsfel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_cluster, special_distance, get_file_row_index, jobtime, ConfigHandler
from TransformerAndMoe import TransformerModel, model_predict


def result_contact(
    test_processed,
    reconstructed,
    weight_path,
    total_x,
    total_reconstructed,
    total_score_weight,
    total_metric_score,
):
    """
    Concatenates the results of anomaly detection for a single MTS segment.

    Args:
        test_processed: Processed test data segment.
        reconstructed: Reconstructed data segment from the model.
        weight_path: Path to the file containing feature weights.
        total_x: Accumulated processed test data.
        total_reconstructed: Accumulated reconstructed data.
        total_score_weight: Accumulated anomaly scores.

    Returns:
        tuple: Updated total_x, total_reconstructed, total_times, and total_score_weight.
    """
    x_all_processed = test_processed[-len(reconstructed) :]
    all_reconstructed = reconstructed

    weight_dict = None
    if not weight_path is None:
        with open(weight_path, "r", encoding="utf-8") as f:
            weight_dict = json.load(f)
    reconstruct_error = np.square(x_all_processed - reconstructed)
    weight_list = list(weight_dict.values())
    all_score = np.sum(reconstruct_error * weight_list, axis=1).reshape(-1, 1)
    metric_score = np.sum(reconstruct_error * weight_list, axis=0)
    print(metric_score.shape)
    # contact result
    if total_x.size == 0:
        total_x = x_all_processed
        total_reconstructed = all_reconstructed
        total_score_weight = all_score
        total_metric_score = metric_score

    else:
        total_x = np.concatenate((total_x, x_all_processed), axis=0)
        total_reconstructed = np.concatenate(
            (total_reconstructed, all_reconstructed), axis=0
        )
        total_score_weight = np.concatenate((total_score_weight, all_score), axis=0)
        total_metric_score = total_metric_score + metric_score

    return total_x, total_reconstructed, total_score_weight, total_metric_score


def tsfel_match(used_data, tsfel_cfg, metrics, cluster_num, center_subfolder, center_file):
    """
    Matches a MTS segment to a cluster using TSFEL features and distance calculation.

    Args:
        used_data: MTS segment to be matched.
        tsfel_cfg: Configuration for TSFEL feature extraction.
        metrics: List of candidate metrics.
        cluster_num: Number of clusters.
        center_subfolder: Path to the folder containing cluster centers.
        center_file: File name of the cluster center.

    Returns:
        int: The matched cluster label.
    """
    extract_feature = []
    temp_list = []
    for i in range(used_data.shape[1]):
        temp_df = used_data[:, i]
        X = tsfel.time_series_features_extractor(tsfel_cfg, temp_df)
        if len(extract_feature) == 0:
            for j in X.columns:
                extract_feature.append("_".join(j.split("_")[1:]))
        temp_list.append(X.reset_index(drop=True))
    tsfel_df = pd.concat(temp_list, axis=0)
    tsfel_df.columns = extract_feature
    tsfel_df.index = metrics

    FFT_index = -1
    features_list = tsfel_df.columns.tolist()
    for i in range(len(features_list)):
        if features_list[i].split("_")[0] == "FFT mean coefficient":
            FFT_index = i
            break
    feature_indices = [
        i for i in features_list if i.split("_")[0] == "FFT mean coefficient"
    ]
    new_column_list = [i for i in features_list if not i in feature_indices]
    new_column_list.insert(FFT_index, "FFT mean coefficient")

    average_values = tsfel_df[feature_indices].mean(axis=1)

    tsfel_df["FFT mean coefficient"] = average_values
    result = tsfel_df.reindex(columns=new_column_list)

    used_tsfel_data = result.fillna(0).values

    label_list = list(range(cluster_num))
    distance_list = []
    for detect_label in label_list:
        center_path = f"{center_subfolder}/{detect_label}/{center_file}"
        distance = special_distance(center_path, used_tsfel_data, metrics)
        distance_list.append(distance)
    node_judge = pd.DataFrame({"label": label_list, "distance": distance_list})
    node_judge.sort_values(by="distance", inplace=True, ascending=True)
    match_label = node_judge["label"].iloc[0]

    return match_label


def predict_transformer(
    node_list,
    config,
    tsfel_cfg,
):
    """
    Performs anomaly detection using Transformer models for a list of nodes.

    Args:
        node_list: List of node names to be processed.
        config: Configuration object.
        metrics: List of metrics.
        tsfel_cfg: Configuration for TSFEL feature extraction.

    """
    weight_file = config.metric_weight_file

    job_model_folder = f"{config.model_dir}/job"
    nojob_model_folder = f"{config.model_dir}/nojob"

    job_subfolder = config.job_center_feature
    nojob_subfolder = config.nojob_center_feature

    job_label_node_dict = load_cluster(config.job_cluster_file)
    nojob_label_node_dict = load_cluster(config.nojob_cluster_file)

    job_clusternum = len(job_label_node_dict.keys())

    with open(config.metric_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    all_test_start_time = str(config.valid_end)
    all_end_time = str(config.test_end)
    all_test_start_time_T = all_test_start_time.replace(" ", "T")
    all_end_time_T = all_end_time.replace(" ", "T")

    start_stamp = time.mktime(time.strptime(all_test_start_time, "%Y-%m-%d %H:%M:%S"))
    node_list.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for node in tqdm(node_list):
        job_time_list = jobtime(
            node,
            config.jobinfo,
            all_test_start_time=all_test_start_time_T,
            all_end_time=all_end_time_T,
            min_length=config.min_length * config.interval,
        )

        nojob_label = -1
        for key in nojob_label_node_dict.keys():
            if node in nojob_label_node_dict[key]:
                nojob_label = key
                break
        if nojob_label < 0:
            print(f"node {node} haven't found nojob cluster")
            nojob_label = 0

        node_df = pd.read_csv(f"{config.source_folder}/online/{node}/metric.csv")
        test_data = node_df[metrics].values
        
        last_time = all_test_start_time
        total_end_time = all_end_time
        total_x = np.array([])
        total_reconstructed = np.array([])
        total_score_weight = np.array([])
        total_metric_score = np.array([])
        input_dim = len(metrics)
        nhead = config.nhead
        dim_feedforward = (
            (input_dim // nhead + 1) * nhead if input_dim % nhead != 0 else input_dim
        )
        num_layers = config.num_layers
        num_experts = config.num_experts
        k = config.k

        for start_time, end_time in job_time_list:

            if end_time > total_end_time:
                end_time = total_end_time
            if start_time < last_time:
                start_time = last_time
            if end_time < last_time:
                continue

            if start_time > last_time:
                # idle waiting time before this operation
                start_row, end_row = get_file_row_index(
                    last_time, start_time, start_stamp, config.interval
                )
                if end_row >= test_data.shape[0]:
                    end_row -= 1

                if start_row >= end_row:
                    print(f"==last_time: {last_time}  start_time: {start_time}")
                    print(f"==start_row: {start_row}  end_row: {end_row}")
                else:
                    test_processed = test_data[start_row:end_row]
                    model_save_path = os.path.join(nojob_model_folder, str(nojob_label))
                    model = TransformerModel(
                        input_dim=input_dim,
                        nhead=nhead,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        device=device,
                        num_experts=num_experts,
                        k=k,
                    ).to(device)
                    
                    model.load_state_dict(
                        torch.load(
                            os.path.join(model_save_path, config.model_name),
                            map_location=torch.device(device),
                        )
                    )
                    model.eval()

                    reconstructed, original = model_predict(model, test_processed)

                    weight_path = f"{nojob_subfolder}/{nojob_label}/{weight_file}"

                    total_x, total_reconstructed, total_score_weight, total_metric_score = (
                        result_contact(
                            test_processed,
                            reconstructed,
                            weight_path,
                            total_x,
                            total_reconstructed,
                            total_score_weight,
                            total_metric_score,
                        )
                    )

                last_time = start_time
            # job execution
            start_row, end_row = get_file_row_index(start_time, end_time, start_stamp, config.interval)
            test_processed = test_data[start_row:end_row]
            if (end_row - start_row) < config.match_len:
                print("job time not enough")
                used_data = test_processed
            else:
                used_data = test_processed[:config.match_len]
            
            job_label = 0
            job_label = tsfel_match(
                used_data, tsfel_cfg, metrics, job_clusternum, job_subfolder, config.center_file
            )

            model_save_path = os.path.join(job_model_folder, str(job_label))

            model = TransformerModel(
                input_dim=input_dim,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                device=device,
                num_experts=num_experts,
                k=k,
            ).to(device)
            
            model.load_state_dict(
                torch.load(
                    os.path.join(model_save_path, config.model_name),
                    map_location=torch.device(device),
                )
            )
            model.eval()

            reconstructed, original = model_predict(model, test_processed)

            weight_path = f"{job_subfolder}/{job_label}/{weight_file}"
            total_x, total_reconstructed, total_score_weight, total_metric_score = (
                result_contact(
                    test_processed,
                    reconstructed,
                    weight_path,
                    total_x,
                    total_reconstructed,
                    total_score_weight,
                    total_metric_score,
                )
            )

            last_time = end_time

        if total_end_time > last_time:
            # idle waiting time between last assignment and 0:00 on last day
            start_row, end_row = get_file_row_index(
                last_time, total_end_time, start_stamp, config.interval
            )

            test_processed = test_data[start_row:end_row]
            model_save_path = os.path.join(nojob_model_folder, str(nojob_label))

            model = TransformerModel(
                input_dim=input_dim,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                device=device,
                num_experts=num_experts,
                k=k,
            ).to(device)
            
            model.load_state_dict(
                torch.load(
                    os.path.join(model_save_path, config.model_name),
                    map_location=torch.device(device),
                )
            )
            model.eval()

            reconstructed, original = model_predict(model, test_processed)

            weight_path = f"{nojob_subfolder}/{nojob_label}/{weight_file}"
            total_x, total_reconstructed, total_score_weight, total_metric_score = (
                result_contact(
                    test_processed,
                    reconstructed,
                    weight_path,
                    total_x,
                    total_reconstructed,
                    total_score_weight,
                    total_metric_score,
                )
            )
            last_time = total_end_time

        save_folder = f'{config.result_dir}/{node}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        np.save(os.path.join(save_folder, config.all_processed_data), total_x)
        np.save(
            os.path.join(save_folder, config.all_reconstructed), total_reconstructed
        )
        score_csv_path = f"{save_folder}/{config.score_file}"
        save_score = pd.DataFrame(total_score_weight)
        save_score.columns = ["score"]
        save_score.to_csv(score_csv_path, index=False)

        df = pd.DataFrame(total_metric_score, index=metrics, columns=['score'])
        df_sorted = df.sort_values(by='score', ascending=False)
        df_sorted.to_csv(f"{save_folder}/metric_score.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detection'
    )
    
    parser.add_argument('--interval', required=True,  default=10, type=int, help='data sampling interval')
    parser.add_argument('--match_len', required=True,  default=200, type=int, help='the length of the header data used for pattern matching')
    parser.add_argument('--max_epochs', type=int, default=30, help='the epoch of training')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch_size of training')
    parser.add_argument('--window_size', type=int, default=20, help='the windowsize of data')
    parser.add_argument('--min_length', default=11, type=int, help='minimum job length ')
    parser.add_argument('--nhead', default=3, help="the number of heads in the Transformer's multi-head attention mechanism")
    parser.add_argument('--num_layers', default=3, help='the number of encoder layers')
    parser.add_argument('--num_experts', default=3, type=int, help='the number of experts')
    parser.add_argument('--k', default=1, type=int, help='the number of experts to select')

    cfg_file = "config.yml"
    cfg_para = {"dataset": None, "rawdata_dir": None, "config_file": cfg_file}
    cfg = ConfigHandler(cfg_para, parser).config

    tsfel_cfg = tsfel.get_features_by_domain()
    node_list = os.listdir(f"{cfg.source_folder}/online")

    predict_transformer(
        node_list,
        cfg,
        tsfel_cfg,
    )
