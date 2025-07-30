"""
This file provides utility functions for selecting a subset of nodes from a cluster based on their distance to the cluster center.
"""
import os
import argparse
import yaml
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean


class ConfigHandler:
    """
    Handles configuration files and command-line arguments.

    Loads a default configuration from a YAML file, updates it with command-line arguments,
    and completes directory paths based on the configuration.
    """
    def __init__(self, run_time_para, parser=None):
        # load default config
        dir_ = os.path.dirname(os.path.abspath(__file__))  # Returns the parent path
        dir_ = os.path.dirname(dir_)  # Returns the parent path
        config_path = os.path.join(dir_, run_time_para['config_file'])
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config_dict = yaml.load(f, Loader=yaml.FullLoader)  # Returns the key-value pairs of the configuration file

        # update config according to executing parameters
        if parser is None:
            parser = argparse.ArgumentParser()
        for field, value in self._config_dict.items():
            parser.add_argument(f'--{field}', default=value)
        for field, value in run_time_para.items():
            parser.add_argument(f'--{field}', default=value)
        self._config = parser.parse_args()

        # complete config
        self._trans_format()
        self._complete_dirs()

    def _trans_format(self):
        """
        Converts invalid formats in the configuration to valid ones.

        Replaces 'None' strings with None values and converts numeric strings to integers or floats.
        """
        config_dict = vars(self._config)
        for item, value in config_dict.items():
            if value == 'None':
                config_dict[item] = None
            elif isinstance(value, str) and is_number(value):
                if value.isdigit():
                    value = int(value)
                else:
                    value = float(value)
                config_dict[item] = value

    def _complete_dirs(self):
        """
        Completes directory paths in the configuration based on dataset and other parameters.
        """
        if self._config.dataset:
            if self._config.result_dir:
                self._config.result_dir = self._make_dir(self._config.result_dir)
        else:
            return 

    def _make_dir(self, dir_):
        """
        Creates a directory if it doesn't exist and returns the path.

        Args:
            dir_ (str): The directory path to be created.

        Returns:
            str: The complete directory path.
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        par_dir = os.path.dirname(cur_dir)
        dir_ = os.path.join(par_dir, dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        return dir_

    @property
    def config(self):
        """
        Returns the configuration object.

        Returns:
            argparse.Namespace: The configuration object.
        """
        return self._config



def is_number(s):
    """
    Checks if a string represents a number (integer or float).

    Args:
        s (str): The string to be checked.

    Returns:
        bool: True if the string represents a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def ensure_folder_exist(folder_path):
    """
    Ensures that a folder exists at the specified path. Creates the folder if it doesn't exist.

    Args:
        folder_path (str): The path to the folder to be checked/created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def select_node(
    cluster_node,
    label,
    job_folder,
    center_df,
    node_num_max=10,
):
    """
    Selects a certain number of nearest nodes from a cluster based on their distance to the cluster center.

    Args:
         cluster_node (dict): Dictionary mapping cluster labels to lists of node names.
        label (int): The label of the cluster to select nodes from.
        job_folder (str): Path to the folder containing node data files.
        center_df (pd.DataFrame, optional): DataFrame representing the cluster center. Defaults to None.
        node_num_max (int, optional): Maximum number of nodes to select. Defaults to 30.

    Returns:
        list: A list of selected node names.
    """
    node_list = cluster_node[label]
    if len(node_list) > node_num_max:
        distance_list = []
        for full_name in node_list:
            data_path = f"{job_folder}/{full_name}.csv"
            distance = euclidean_distance(
                center_df,
                pd.read_csv(data_path, index_col=0).fillna(0),
            )
            distance_list.append(distance)
        node_judge = pd.DataFrame({"full_name": node_list, "distance": distance_list})
        node_judge.sort_values(by="distance", inplace=True, ascending=True)
        node_list = node_judge["full_name"].head(node_num_max).tolist()
    return node_list


def euclidean_distance(
    df1, df2
):
    """
    Calculates the Euclidean distance between corresponding columns.
    Returns the average of the calculated distances.
    """
    euclidean_distances = []

    df1 = df1.T
    df2 = df2.T
    for col in df1.columns:
        dist = euclidean(df1[col], df2[col])
        euclidean_distances.append(dist)

    average_distance = np.mean(np.nan_to_num(euclidean_distances, nan=0))

    return average_distance


def load_cluster(file_path):
    """
    Loads clustering results from a CSV file and organizes them into a dictionary.

    Reads a CSV file containing node names and their assigned cluster labels. Creates a dictionary
    where keys are cluster labels and values are lists of node names belonging to that cluster.

    Args:
        file_path (str): The path to the CSV file containing clustering results.

    Returns:
        dict: A dictionary mapping cluster labels to lists of node names.
    """
    cluster_node = {}
    df_file = pd.read_csv(file_path)
    for _, row in df_file.iterrows():
        # Load result of Hierarchical clustering
        label = int(row["Label"])
        name = row["Node"]
        if label in cluster_node:
            cluster_node[label].append(name)
        else:
            cluster_node[label] = [name]
    return cluster_node
