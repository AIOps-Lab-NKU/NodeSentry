"""
This file manages the training process for fine-grained model sharing. 

It loads the MTS segments for each cluster, trains a dedicated Transformer model on it, and saves the trained model for later use in anomaly detection. 
It handles both "job" and "no-job" data separately, ensuring that each cluster has its own specialized model.  
The training process includes visualizing and saving the training loss for each model.
"""
import os
import json
import random
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from TransformerAndMoe import model_fit
from utils import load_cluster, ensure_folder_exist, ConfigHandler


def data_contact(
    config,
    label,
    metrics,
    source_folder,
    is_job=True,
):
    """
    Processes and saves MTS segments based on job status and label.

    Args:
        config: Configuration object.
        label: The label of the current data group.
        metrics: A list of metrics to be used.
        source_folder: Path to the source data folder.
        is_job: Boolean indicating whether the data is for job execution or idle waiting.
    """
    cut_length = config.batch_size + config.window_size - 1

    if is_job:  # job
        job_folder = os.path.join(source_folder, "job")
        distance_path = f"{config.job_center_feature}/{label}/{config.distance_file}"
        with open(distance_path, "r", encoding="utf-8") as f:
            node_distance = json.load(f)
        data_batch = []
        for full_name in tqdm(list(node_distance.keys())):
            node_name, job_number = full_name.split("_", 1)
            data_path = f"{job_folder}/{node_name}/metric_{job_number}.csv"

            job_df = pd.read_csv(data_path)
            job_data = job_df[metrics]
            add_num = node_distance[full_name]
            distance_column = pd.Series(add_num, index=job_data.index, name="distance")
            job_data["distance"] = distance_column
            for i in range(job_data.shape[0] // cut_length):
                temp_data = job_data.iloc[
                    i * cut_length : (i + 1) * cut_length, :
                ].values
                data_batch.append(np.asarray(temp_data, dtype=np.float32))

    else:  # no-job
        nojob_path = os.path.join(source_folder, "nojob")
        distance_path = f"{config.nojob_center_feature}/{label}/{config.distance_file}"
        with open(distance_path, "r", encoding="utf-8") as f:
            node_distance = json.load(f)
        data_batch = []

        for node_name in tqdm(list(node_distance.keys())):
            data_path = f"{nojob_path}/{node_name}.csv"

            nojob_df = pd.read_csv(data_path)
            nojob_data = nojob_df[metrics]
            add_num = node_distance[node_name]
            distance_column = pd.Series(add_num, index=nojob_data.index, name="distance")
            nojob_data["distance"] = distance_column
            for i in range(nojob_data.shape[0] // cut_length):
                temp_data = nojob_data.iloc[
                    i * cut_length : (i + 1) * cut_length, :
                ].values
                data_batch.append(np.asarray(temp_data, dtype=np.float32))
    # shuffle
    random.shuffle(data_batch)
    data_all = np.concatenate(data_batch, axis=0)
    return data_all


def model_sharing_train(label, data_all, config, is_job=True):
    """
    Trains a Transformer model for a specific cluster and saves it.

    Args:
        label: The cluster label for which the model is trained.
        data_all: The training data for the specific cluster.
        config: Configuration object containing training parameters.
        is_job: Boolean indicating whether the data is for job execution or idle waiting.
    """
    if is_job:
        model_save_path = os.path.join(config.model_dir, f"job/{str(label)}")
    else:
        model_save_path = os.path.join(config.model_dir, f"nojob/{str(label)}")
    ensure_folder_exist(model_save_path)

    if is_job:
        temp_save_folder = os.path.join(config.result_dir, "job")
    else:
        temp_save_folder = os.path.join(config.result_dir, "nojob")
    ensure_folder_exist(temp_save_folder)
    draw_save_path = os.path.join(
        temp_save_folder, f"loss_in_label{label}_and_epoch{config.max_epochs}.png"
    )

    model, train_losses = model_fit(
        data_all,
        config.max_epochs,
        config.nhead,
        config.num_layers,
        config.num_experts,
        config.k,
    )
    # draw loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.savefig(draw_save_path)
    plt.close()

    torch.save(model.state_dict(), f"{model_save_path}/{config.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-Grained Model Sharing'
    )
    parser.add_argument('--max_epochs', required=True, type=int, default=30, help='the epoch of training')
    parser.add_argument('--batch_size', required=True, type=int, default=50, help='the batch_size of training')
    parser.add_argument('--window_size', required=True, type=int, default=20, help='the windowsize of data')
    parser.add_argument('--nhead', default=3, help="the number of heads in the Transformer's multi-head attention mechanism")
    parser.add_argument('--num_layers', default=3, help='the number of encoder layers')
    parser.add_argument('--num_experts', default=3, type=int, help='the number of experts')
    parser.add_argument('--k', default=1, type=int, help='the number of experts to select')

    cfg_file = "config.yml"
    cfg_para = {"dataset": None, "rawdata_dir": None, "config_file": cfg_file}
    cfg = ConfigHandler(cfg_para, parser).config


    source_folder = f'{cfg.source_folder}/offline'
    with open(cfg.metric_file, "r", encoding="utf-8") as f:
        metric_used = json.load(f)
    # job
    label_node_dict = load_cluster(cfg.job_cluster_file)
    used_list = list(label_node_dict.keys())
    for label in tqdm(used_list):
        data_all = data_contact(
            cfg,
            label,
            metric_used,
            source_folder,
            is_job=True,
        )
        model_sharing_train(label, data_all, cfg, is_job=True)

    # nojob
    label_node_dict = load_cluster(cfg.nojob_cluster_file)
    used_list = list(label_node_dict.keys())
    for label in tqdm(used_list):
        data_all = data_contact(
            cfg,
            label,
            metric_used,
            source_folder,
            is_job=False,
        )
        model_sharing_train(label, data_all, cfg, is_job=False)