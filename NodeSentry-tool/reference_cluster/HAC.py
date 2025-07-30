import os
import pandas as pd
import tsfel
import warnings
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

# Ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_features(data):
    # Filter out non-numeric columns to ensure only numeric data is processed
    numeric_data = data.select_dtypes(include=[float, int])

    cfg = tsfel.get_features_by_domain()

    # Set sampling frequency to 1
    for domain in cfg.keys():
        for feature in cfg[domain].keys():
            if 'parameters' in cfg[domain][feature] and 'fs' in cfg[domain][feature]['parameters']:
                cfg[domain][feature]['parameters']['fs'] = 1

    feature_vector = tsfel.time_series_features_extractor(cfg, numeric_data, fs=1, verbose=0)  # Explicitly set fs=1
    return feature_vector

def calculate_distance_matrix(features):
    # Ensure all features are numeric
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    distance_matrix = pdist(features, metric='euclidean')  # Use Euclidean distance, other options: manhattan, cosine
    return distance_matrix

def hierarchical_clustering(distance_matrix):
    linkage_matrix = sch.linkage(distance_matrix, method='ward')  # Use 'ward', other options: single, complete, average
    return linkage_matrix

def save_cluster_result(linkage_matrix, node_names, save_path):
    cluster_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')  # Assuming 4 clusters
    clusters = {i: [] for i in set(cluster_labels)}
    for node, label in zip(node_names, cluster_labels):
        clusters[label].append(node)

    with open(save_path, 'w') as f:
        for label, nodes in clusters.items():
            line = f"{label}|" + "|".join(nodes) + "\n"
            f.write(line)

def main():
    data_path = os.path.join(os.path.dirname(__file__), "../node_data")
    save_path = os.path.join(os.path.dirname(__file__), "../config_files/cluster_result.txt")

    print("Reading CSV files...")
    data_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    data = []
    node_names = []

    for file in data_files:
        df = pd.read_csv(os.path.join(data_path, file))
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        data.append(df)
        node_names.append(file.split('.')[0])

    print(f"Read {len(data_files)} CSV files.")

    print("Extracting features...")
    feature_matrix = pd.DataFrame()

    for df in data:
        feature_vector = extract_features(df)
        feature_matrix = pd.concat([feature_matrix, feature_vector], ignore_index=True)

    print(f"Extracted features for {len(data_files)} nodes.")

    print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(feature_matrix)
    print("Distance matrix calculation completed.")

    print("Performing hierarchical clustering...")
    linkage_matrix = hierarchical_clustering(distance_matrix)  # Use uncompressed distance matrix
    print("Hierarchical clustering completed.")

    print("Saving clustering data...")
    save_cluster_result(linkage_matrix, node_names, save_path)
    print(f"Clustering data saved to {save_path}.")

if __name__ == "__main__":
    main()
