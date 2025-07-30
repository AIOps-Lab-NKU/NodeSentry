"""
This script performs coarse-grained clustering on MTS of HPC jobs and idle nodes.

The results, including normalized features, cluster assignments, cluster centers, distances, and feature weights, are saved to files for downstream analysis, such as anomaly detection.
"""
import os
import argparse
import json
import warnings
import tsfel
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering
from utils import (
    euclidean_distance,
    select_node,
    load_cluster,
    ensure_folder_exist,
    ConfigHandler,
)

warnings.filterwarnings("ignore")


def extract_feature_in_df(config, df, metric):
    """
    Extracts feature indices from a DataFrame for specified metrics.

    Args:
        config (dict): TSFEL configuration dictionary.
        df (pd.DataFrame): Input DataFrame containing MTS.
        metric (list): List of metric names to extract features from.
    
    Returns:
        pd.DataFrame: DataFrame with feature indices.
    """
    extract_feature = []
    temp_list = []
    df = df[metric]
    for i in range(df.shape[1]):
        temp_df = df.iloc[:, i]
        X = tsfel.time_series_features_extractor(config, temp_df)
        if len(extract_feature) == 0:
            for j in X.columns:
                extract_feature.append("_".join(j.split("_")[1:]))
        temp_list.append(X.reset_index(drop=True))

    result_df = pd.concat(temp_list, axis=0)
    result_df.columns = extract_feature

    result_df.index = metric
    return result_df


def FFT_mean_df(df):
    """
    Averages features associated with 'FFT mean coefficient'.

    Calculates the average of features with names like 'FFT mean coefficient_0', ..., 'FFT mean coefficient_255'.
    Replaces these features with a single 'FFT mean coefficient' column in the original position.

    Args:
        df (pd.DataFrame): Input DataFrame with FFT mean coefficient features.

    Returns:
        pd.DataFrame: DataFrame with averaged FFT mean coefficient feature.
    """
    features_list = df.columns.tolist()
    FFT_index = -1
    for i, element in enumerate(features_list):
        if element.split("_")[0] == "FFT mean coefficient":
            FFT_index = i
            break

    feature_indices = [
        i for i in features_list if i.split("_")[0] == "FFT mean coefficient"
    ]

    new_column_list = [i for i in features_list if not i in feature_indices]
    new_column_list.insert(FFT_index, "FFT mean coefficient")

    average_values = df[feature_indices].mean(axis=1)
    df["FFT mean coefficient"] = average_values
    result = df.reindex(columns=new_column_list)

    return result


def df_miu_sigma(df):
    """
    Standardizes each row of a DataFrame using mean and standard deviation.

    Args:
        df (pd.DataFrame): Input DataFrame to be standardized.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    row_means = df.mean(axis=1)
    row_stds = df.std(axis=1)

    normalized_data = ((df.T - row_means) / row_stds).T

    return normalized_data


def extract_features(config):
    """
    Extracts feature indices from offline data for different states (job/nojob).

    Args:
        config: Configuration object.
    """
    tsfel_cfg = tsfel.get_features_by_domain()
    with open(config.metric_file, "r", encoding="utf-8") as f:
        metric_candidates = json.load(f)

    data_folder = f"{config.source_folder}/offline"
    state_list = os.listdir(data_folder)
    normal_folder = config.normalized_folder
    for state in state_list:
        state_save_folder = os.path.join(
            normal_folder, state
        )
        ensure_folder_exist(state_save_folder)
        state_folder = f"{data_folder}/{state}"
        if state == "job":
            for node in tqdm(os.listdir(state_folder)):
                node_folder = os.path.join(state_folder, node)
                for job_file in os.listdir(node_folder):
                    file_full_path = os.path.join(
                        node_folder, job_file
                    )
                    job_number = job_file.split("_", 1)[1]
                    result_save_full_path = f"{state_save_folder}/{node}_{job_number}"
                    job_df = pd.read_csv(file_full_path)
                    metric_feature_df = extract_feature_in_df(
                        tsfel_cfg, job_df, metric_candidates
                    )
                    mean_FFT_df = FFT_mean_df(metric_feature_df)
                    normalized_df = df_miu_sigma(mean_FFT_df)
                    normalized_df.to_csv(result_save_full_path)
        elif state == "nojob":
            for nojob_file in tqdm(os.listdir(state_folder)):
                file_full_path = os.path.join(
                    state_folder, nojob_file
                )
                nojob_df = pd.read_csv(file_full_path)
                if nojob_df.shape[0] <= config.min_length:
                    print(f"The length of {nojob_file} is not enough !!!")
                    continue
                result_save_full_path = f"{state_save_folder}/{nojob_file}"
                metric_feature_df = extract_feature_in_df(
                    tsfel_cfg, nojob_df, metric_candidates
                )
                mean_FFT_df = FFT_mean_df(metric_feature_df)
                normalized_df = df_miu_sigma(mean_FFT_df)
                normalized_df.to_csv(result_save_full_path)


def calcu_matrix(data_dict):
    """
    Calculates and saves the Euclidean distance matrix for a dictionary of DataFrames.

    Args:
        data_dict (dict): Dictionary where keys are data identifiers and values are pandas DataFrames.
        
    Returns:
        pd.DataFrame: Distance matrix.
    """
    keys = list(data_dict.keys())
    result_df = pd.DataFrame(index=keys, columns=keys)

    for i in tqdm(range(0, len(keys))):
        for j in range(i + 1, len(keys)):
            euclidean_distance = []
            key1 = keys[i]
            key2 = keys[j]
            df1 = data_dict[key1].T
            df2 = data_dict[key2].T

            for col in df1.columns:
                dist = euclidean(df1[col], df2[col])
                euclidean_distance.append(dist)

            result_df.loc[key1, key2] = sum(euclidean_distance)

    return result_df


def Hiera(n_clusters, dist_matrix, node_list):
    """
    Performs hierarchical clustering on a distance matrix.

    Args:
        n_clusters (int): The number of clusters to form.
        dist_matrix (np.ndarray): The distance matrix.
        node_list (list): List of node names corresponding to the distance matrix.

    Returns:
        tuple: A tuple containing a DataFrame with node names and their assigned cluster labels,
               and the fitted AgglomerativeClustering model.
    """
    # Hierarchical clustering
    HAC_model = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="complete"
    )
    clustering = HAC_model.fit(dist_matrix)

    # Save result in a new DataFrame
    result = pd.DataFrame({"Node": node_list, "Label": clustering.labels_}).sort_values(
        ["Label", "Node"]
    )

    return result, clustering


def HAC(config):
    """
    Performs hierarchical clustering with a specified number of clusters for both "job" and "nojob" data.
    Calculates and saves the clustering results to CSV files.

    Args:
        config: Configuration object containing parameters for clustering.
    """
    # job execution
    job_folder_path = f"{config.normalized_folder}/job"
    job_nodefile_df = {}
    for node_file in os.listdir(job_folder_path):
        node_files_path = os.path.join(job_folder_path, node_file)
        if node_file.endswith(".csv"):
            file_name = node_file.split(".")[0]
            data = pd.read_csv(node_files_path, index_col=0)
            job_nodefile_df[file_name] = data.fillna(0)

    # sort
    sorted_list = sorted(
        job_nodefile_df.items(),
        key=lambda x: (int(x[0].split("_")[0][2:]), int(x[0].split("_")[1][3:])),
    )
    job_sorted_dict = dict(sorted_list)

    job_df = calcu_matrix(job_sorted_dict)

    # idle waiting
    nojob_folder_path = f"{config.normalized_folder}/nojob"
    nojob_nodefile_df = {}
    for node_file in os.listdir(nojob_folder_path):
        if node_file.endswith(".csv"):
            file_name = node_file.split(".")[0]
            file_path = os.path.join(nojob_folder_path, node_file)
            data = pd.read_csv(file_path, index_col=0)
            nojob_nodefile_df[file_name] = data.fillna(0)

    # sort
    sorted_list = sorted(nojob_nodefile_df.items(), key=lambda x: (int(x[0][2:])))
    nojob_sorted_dict = dict(sorted_list)

    nojob_df = calcu_matrix(nojob_sorted_dict)

    job_result_save_folder = f"{config.cluster_folder}/job"
    nojob_result_save_folder = f"{config.cluster_folder}/nojob"
    if not os.path.exists(job_result_save_folder):
        os.makedirs(job_result_save_folder)

    if not os.path.exists(nojob_result_save_folder):
        os.makedirs(nojob_result_save_folder)

    job_result_file = f"{job_result_save_folder}/job_{config.job_num}.csv"
    nojob_result_file = f"{nojob_result_save_folder}/nojob_{config.nojob_num}.csv"

    # Perform hierarchical clustering for job data
    job_df = job_df.T.fillna(job_df)
    # Fill the upper triangle of the distance matrix with the lower triangle
    dist_matrix = job_df.values
    dist_matrix += dist_matrix.T - np.diag(dist_matrix.diagonal())
    dist_matrix = np.nan_to_num(dist_matrix)

    result, clustering = Hiera(config.job_num, dist_matrix, job_df.columns)
    result.to_csv(job_result_file, index=False)

    # Perform hierarchical clustering for nojob data
    nojob_df = nojob_df.T.fillna(nojob_df)

    dist_matrix = nojob_df.values
    dist_matrix += dist_matrix.T - np.diag(dist_matrix.diagonal())
    dist_matrix = np.nan_to_num(dist_matrix)

    result, clustering = Hiera(config.nojob_num, dist_matrix, nojob_df.columns)
    result.to_csv(nojob_result_file, index=False)


def feature_center(config, is_job=True):
    """
    Calculates cluster centers, distances, and feature weights based on hierarchical clustering results.

    Args:
        config: Configuration object containing parameters for data processing.
        is_job (bool, optional): Flag indicating whether to process job data (True) or nojob data (False). Defaults to True.
    """
    with open(config.metric_file, "r", encoding='utf-8') as f:
        indicators = json.load(f)
    if is_job:
        source_folder = f"{config.source_folder}/offline/job"
        label_node_dict = load_cluster(config.job_cluster_file)
        normal_data_folder = f"{config.normalized_folder}/job"
        center_feature_folder = config.job_center_feature
        label_num = config.job_num
    else:
        source_folder = f"{config.source_folder}/offline/nojob"
        label_node_dict = load_cluster(config.nojob_cluster_file)
        normal_data_folder = f"{config.normalized_folder}/nojob"
        center_feature_folder = config.nojob_center_feature
        label_num = config.nojob_num

    for label in tqdm(range(label_num)):
        node_list_in_label = label_node_dict[label]
        dataframes = []
        for node_name in node_list_in_label:
            file_path = f"{normal_data_folder}/{node_name}.csv"
            df = pd.read_csv(file_path, index_col=0).fillna(0)
            dataframes.append(df)

        # Calculate the average of all files as the initial cluster center
        avg_df = pd.concat(dataframes).groupby(level=0).mean()

        # Select some nodes closest to the cluster center
        select_node_list = select_node(
            label_node_dict,
            label,
            normal_data_folder,
            center_df=avg_df,
            node_num_max=config.node_num,
        )

        dataframes_new = []
        select_node_df = {}
        for node_name in select_node_list:
            file_path = f"{normal_data_folder}/{node_name}.csv"
            df = pd.read_csv(file_path, index_col=0).fillna(0)
            dataframes_new.append(df)
            select_node_df[node_name] = df

        # Calculate new cluster center
        avg_df = pd.concat(dataframes_new).groupby(level=0).mean()

        # Calculate distances
        node_distance = {}
        for node_ful_name, node_df in select_node_df.items():
            average_distance = euclidean_distance(node_df, avg_df)
            node_distance[node_ful_name] = average_distance

        min_val = min(node_distance.values())
        max_val = max(node_distance.values())

        # normalization
        distances_dict = {
            key: (
                1 - (value - min_val) / (max_val - min_val) if max_val != min_val else 1
            )
            for key, value in node_distance.items()
        }
        # Save
        target_subfolder = os.path.join(center_feature_folder, str(label))
        ensure_folder_exist(target_subfolder)

        distances_path = os.path.join(target_subfolder, config.distance_file)
        with open(distances_path, "w", encoding='utf-8') as json_file:
            json.dump(distances_dict, json_file, indent=4)

        center_path = os.path.join(target_subfolder, config.center_file)
        avg_df.to_csv(center_path)

        node_metric_MAC = pd.DataFrame(0, index=select_node_list, columns=indicators)

        for job_ful_name in select_node_list:  # cn4518_job80
            if is_job:
                node_name, job_num = job_ful_name.split("_", 1)
                df = pd.read_csv(f"{source_folder}/{node_name}/metric_{job_num}.csv")
            else:
                df = pd.read_csv(f"{source_folder}/{job_ful_name}.csv")
            for metric in indicators:
                mean_absolute_change = df[metric].diff().dropna().abs().mean()
                node_metric_MAC.loc[job_ful_name, metric] = mean_absolute_change

        means = pd.Series(0, index=node_metric_MAC.columns)  # initialize Series

        total_sum = sum(distances_dict.values())
        normalized_dict = {
            key: value / total_sum for key, value in distances_dict.items()
        }

        for idx, row in node_metric_MAC.iterrows():
            weight = normalized_dict[idx]  # weight
            means += row * weight

        normalized_means = (
            1 - (means - means.min()) / (means.max() - means.min())
            if means.max() != means.min()
            else pd.Series(1, index=node_metric_MAC.columns)
        )

        # Make sum equal to 1
        weight_means = normalized_means / normalized_means.sum()

        # Save
        save_dict = weight_means.to_dict()
        with open(f"{target_subfolder}/{config.metric_weight_file}", "w", encoding='utf-8') as json_file:
            json.dump(save_dict, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Coarse-Grained Clustering'
    )
    
    parser.add_argument('--job_num', required=True, type=int, default=30, help='the number of clusters for job data')
    parser.add_argument('--nojob_num', required=True, type=int, default=30, help='the number of clusters for nojob data')
    parser.add_argument('--node_num', required=True, type=int, default=30, help='the number of nodes to select when calculating cluster centers')
    parser.add_argument('--min_length', default=11, type=int, help='minimum job length ')
    

    cfg_file = "config.yml"
    cfg_para = {"dataset": None, "rawdata_dir": None, "config_file": cfg_file}
    cfg = ConfigHandler(cfg_para, parser).config


    extract_features(cfg)
    HAC(cfg)
    feature_center(cfg)
    feature_center(cfg, is_job=False)
