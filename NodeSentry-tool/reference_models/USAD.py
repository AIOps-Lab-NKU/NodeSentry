import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add 'model' directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../model')
sys.path.append(model_dir)

# Receive the command-line argument as data path
data_path = sys.argv[1]

communicate_file_path = os.path.join(current_dir, '../communicate_files/annotation_history.txt')

node_name = data_path.split("/")[-1].split('.')[0]
print("Node name is:", node_name)

# Import USAD model and utility functions
from ..model.usad import UsadModel, training, testing
from ..model.utils import get_default_device, to_device

# Load data
df = pd.read_csv(data_path)
data = df.iloc[:, 1:].values  # Assume the first column is timestamp, others are metrics

# Preprocess data
device = get_default_device()
train_size = int(0.7 * len(data))
train_data = data[:train_size]  # First 70% of data for training
test_data = data  # All data for testing

# Convert data to tensors and load into DataLoader
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=False)  # Keep time order, no shuffling
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=32, shuffle=False)

# Initialize USAD model
w_size = train_data.shape[1]  # Input metric dimension
z_size = 12  # Latent variable dimension, can be adjusted
model = UsadModel(w_size, z_size).to(device)

# Train model
history = training(10, model, train_loader, train_loader)  # Assume 10 epochs for training

# Test model
test_results = testing(model, test_loader)

# If test_results is a list, process each tensor
if isinstance(test_results, list):
    test_results_cpu = np.concatenate([result.cpu().numpy() for result in test_results])
else:
    test_results_cpu = test_results.cpu().numpy()

# Mark anomaly data
threshold = np.percentile(test_results_cpu, 95)  # Choose 95th percentile as threshold
anomalies = test_results_cpu > threshold

# Record indices of anomaly data
line_to_write = ""
for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        line_to_write += str(i)  # Use relative index of test data
        line_to_write += ","
line_to_write = line_to_write[:-1] + "\n"

# Output to communication file
if line_to_write.strip():  # Ensure there is content to write
    with open(communicate_file_path, 'a') as f:
        f.write(line_to_write)
    print("Anomaly indices written to file:", communicate_file_path)
else:
    print("No anomalies detected, historical annotation results will be drawn for you.")
