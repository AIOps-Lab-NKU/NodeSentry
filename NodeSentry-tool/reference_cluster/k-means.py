import os
import pandas as pd
import tsfel
import warnings
from sklearn.cluster import KMeans

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

def perform_kmeans_clustering(features, n_clusters=4):
    # Ensure all features are numeric
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def save_cluster_result(cluster_labels, node_names, save_path):
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

    print("Performing K-Means clustering...")
    cluster_labels = perform_kmeans_clustering(feature_matrix, n_clusters=4)  # Assuming 4 clusters
    print("K-Means clustering completed.")

    print("Saving clustering data...")
    save_cluster_result(cluster_labels, node_names, save_path)
    print(f"Clustering data saved to {save_path}.")

if __name__ == "__main__":
    main()
