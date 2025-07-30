import sys
import os
import numpy as np
import pandas as pd

# Add 'model' directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../model')
sys.path.append(model_dir)

# Receive the command-line argument as data path
data_path = sys.argv[1]

communicate_file_path = os.path.join(current_dir, '../communicate_files/annotation_history.txt')

node_name = data_path.split("/")[-1].split('.')[0]
print("Node name is:", node_name)

# Load data
df = pd.read_csv(data_path)
data = df.iloc[:, 1:].values  # Assume the first column is timestamp, others are metrics

# Calculate mean and standard deviation
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# Calculate anomaly scores: points where absolute differences exceed 3 standard deviations
anomaly_scores = np.abs(data - mean) > 3 * std

# Mark each time point as anomaly: if more than 5% of the metrics are anomalous, it's an anomaly point
threshold_percentage = 0.05  # 5% threshold
anomalies = np.sum(anomaly_scores, axis=1) > (data.shape[1] * threshold_percentage)

# Print debug information
print("Mean:", mean)
print("Standard deviation:", std)
print("Anomaly scores:", anomaly_scores)
print("Anomaly flags:", anomalies)

# Record indices of anomaly data
line_to_write = ""
for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        line_to_write += str(i)  # Use relative index of test data
        line_to_write += ","
line_to_write = line_to_write[:-1] + "\n"

# Print debug information for file content
print("Content to write to file:", line_to_write)

# Output to communication file
if line_to_write.strip():  # Ensure there is content to write
    with open(communicate_file_path, 'a') as f:
        f.write(line_to_write)
    print("Anomaly indices written to file:", communicate_file_path)
else:
    print("No anomalies detected, historical annotation results will be drawn for you.")
