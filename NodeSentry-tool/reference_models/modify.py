import sys
import os
import pandas as pd

# Add 'model' directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../model')

# Receive the command-line argument as data path
data_path = sys.argv[1]

communicate_file_path = os.path.join(current_dir, '../communicate_files/annotation_history.txt')

node_name = data_path.split("/")[-1].split('.')[0]
print("Node name is:", node_name)

# Label file path
modified_label_file_path = os.path.join(current_dir, '../labels/', node_name + '.csv')

# Check if modified label file exists
if os.path.exists(modified_label_file_path):
    # Load modified label data
    label_df = pd.read_csv(modified_label_file_path)
    # Get anomaly indices
    anomalies = label_df[label_df['label'] == 1]['index'].values
else:
    print("Modified label file not found.")
    anomalies = []

# Record indices of anomaly data
line_to_write = ",".join(map(str, anomalies)) + "\n"

# Print debug information for file content
print("Content to write to file:", line_to_write)

# Output to communication file
if line_to_write.strip():  # Ensure there is content to write
    with open(communicate_file_path, 'a') as f:
        f.write(line_to_write)
    print("Anomaly indices written to file:", communicate_file_path)
else:
    print("No anomalies detected, historical annotation results will be drawn for you.")
