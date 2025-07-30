"""
Utility functions for configuration handling, data processing, time conversion, job time extraction.
"""
import os
import time
import argparse
import yaml
import numpy as np
import pandas as pd
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


def special_distance(center_file, array_data, metric_candidate):
    """
    Calculates the average Euclidean distance between a data array and a center data file for specific metrics.

    Loads the center data from a file, selects the specified metrics, and calculates the average Euclidean distance
    between the center data and the given data array.

    Args:
        center_file (str): Path to the CSV file containing the center data.
        array_data (np.ndarray): The data array to compare with the center data.
        metric_candidate (list): List of metric names to consider for distance calculation.

    Returns:
        float: The average Euclidean distance.
    """
    center_data = pd.read_csv(center_file, index_col=0).loc[metric_candidate, :].fillna(0).values

    euclidean_distances = []
    for col1, col2 in zip(center_data.T, array_data.T):
        distance = euclidean(col1, col2)
        euclidean_distances.append(distance)

    average_distance = np.mean(np.nan_to_num(euclidean_distances, nan=0))

    return average_distance



def get_file_row_index(start_time, end_time, file_start_stamp, interval):
    """
    Calculates the start and end row indices in a file based on given timestamps and time interval.

    Converts the given start and end times to timestamps, calculates the corresponding row indices
    in a file with a known start timestamp and a specified time interval between rows, and returns the indices.

    Args:
        start_time (str): Start time in the format "%Y-%m-%d %H:%M:%S".
        end_time (str): End time in the format "%Y-%m-%d %H:%M:%S".
        file_start_stamp (int): The timestamp of the first row in the file.
        interval (int): The time interval between rows in seconds.

    Returns:
        tuple: A tuple containing the start and end row indices.
    """
    start_stamp = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    end_stamp = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
    start_index = (start_stamp - file_start_stamp) // interval
    if (start_stamp - file_start_stamp) % interval != 0:
        start_index += 1
    end_index = (end_stamp - file_start_stamp) // interval
    if (end_stamp - file_start_stamp) % interval != 0:
        end_index += 1
    return int(start_index), int(end_index)


def stamp2time(timeStamp):
    """
    Converts a timestamp to a human-readable date and time string.

    Args:
        timeStamp (float): The timestamp to be converted.

    Returns:
        str: The date and time string in the format "%Y-%m-%d %H:%M:%S".
    """
    time_local = time.localtime(timeStamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    
    return dt


def jobtime(node, job_file, all_test_start_time, all_end_time, min_length):
    """
    Extracts the start and end times of jobs that used a specific node from a job information file.

    Reads a job information file and extracts the start and end times of jobs that used the given node.
    Filters the jobs based on a minimum length and a specified time range.

    Args:
        node (str): The node name.
        job_file (str): The path to the job information file.
        all_test_start_time (str): The start time of the overall test period in the format "%Y-%m-%dT%H:%M:%S".
        all_end_time (str): The end time of the overall test period in the format "%Y-%m-%dT%H:%M:%S".
        min_length (int): The minimum length of a job in seconds.

    Returns:
        list: A list of tuples, where each tuple represents the start and end time of a job.
    """
    matched_times = []
    with open(job_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            node_list = line.split("|")[0].split(",")
            if node in node_list:
                start_time = line.split("|")[1]
                end_time = line.split("|")[2]
                if end_time == 'Unknown':
                    end_time = all_end_time
                if end_time < all_test_start_time:
                    continue
                start_stamp = time.mktime(time.strptime(start_time, "%Y-%m-%dT%H:%M:%S"))
                start_stamp = max(start_stamp, time.mktime(time.strptime(all_test_start_time, "%Y-%m-%dT%H:%M:%S")))
                end_stamp = time.mktime(time.strptime(end_time, "%Y-%m-%dT%H:%M:%S"))
                end_stamp = min(end_stamp, time.mktime(time.strptime(all_end_time, "%Y-%m-%dT%H:%M:%S")))
                if end_stamp - start_stamp > min_length:
                    matched_times.append((stamp2time(start_stamp), stamp2time(end_stamp)))

    sorted_times = sorted(matched_times, key=lambda x: x[0])
    return sorted_times