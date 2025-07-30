import os
import pandas as pd
import tsfel
import warnings
from sklearn.cluster import DBSCAN

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

def perform_dbscan_clustering(features, eps=0.5, min_samples=5):
    # Ensure all features are numeric
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(features)
    return cluster_labels

def save_cluster_result(cluster_labels, node_names, save_path):
    clusters = {i: [] for i in set(cluster_labels) if i != -1}  # Ignore noise points
    for node, label in zip(node_names, cluster_labels):
        if label != -1:  # Ignore noise points
            clusters[label].append(node)

    with open(save_path, 'w') as f:
        for label, nodes in clusters.items():
            line = f"{label}|" + "|".join(nodes) + "\n"
            f.write(line)

    # Inform if no clusters were formed
    if not clusters:
        print("All points were marked as noise; no clusters were formed.")

def main():
    data_path = os.path.join(os.path.dirname(__file__), "../node_data")
    save_path = os.path.join(os.path.dirname(__file__), "../config_files/cluster_result.txt")

    print("Reading CSV files...")
    data_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    data = []
    node_names = []

    for file in data_files:
        df = pd.read_csv(os.path.join(data_path, file))
        
        # Check and remove 'index' column
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        # Check and remove 'timestamp' column
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        
        # Append processed data
        data.append(df)
        node_names.append(file.split('.')[0])

    print(f"Read {len(data_files)} CSV files.")

    print("Extracting features...")
    feature_matrix = pd.DataFrame()

    for df in data:
        feature_vector = extract_features(df)
        feature_matrix = pd.concat([feature_matrix, feature_vector], ignore_index=True)

    print(f"Extracted features for {len(data_files)} nodes.")

    if len(feature_matrix) < 5:
        print("Sample size is less than min_samples; all points will be marked as noise.")

    print("Performing DBSCAN clustering...")
    cluster_labels = perform_dbscan_clustering(feature_matrix, eps=5, min_samples=3)  # Set eps=5, min_samples=3
    print("DBSCAN clustering completed.")

    print("Saving clustering data...")
    save_cluster_result(cluster_labels, node_names, save_path)
    print(f"Clustering data saved to {save_path}.")

if __name__ == "__main__":
    main()
