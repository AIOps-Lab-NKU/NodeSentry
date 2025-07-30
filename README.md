# NodeSentry: Effective Node-Level Anomaly Detection in HPC Systems via Coarse-Grained Clustering and Fine-Grained Model Sharing

## Project Introduction

The framework encompasses two principal phases: the offline model training phase and the online anomaly detection phase. During the offline model training phase, we first preprocess the raw MTS. We then perform coarse-grained clustering on the processed MTS, which enables the model to train on the clusters rather than on individual MTS segments. Next, we assign expert networks to different segments within the cluster for model sharing. In the online anomaly detection phase, we extract features from preprocessed MTS to match patterns in the cluster library. Subsequently, the appropriate model is dynamically assigned to identify anomalies in the target nodes.

## Directory Structure


```
NodeSentry-model 
├── data                           # Directory for storing time series data
├── clustering                     # Directory for coarse-grained clustering methods and results
│   ├── classify_result            # Directory for storing clustering results
│   ├── feature_center             # Directory for storing cluster centers and feature weights
│   ├── normalized_result          # Directory for storing standardized feature indices 
│   ├── clustering.py              # Coarse-grained clustering code
│   └── utils.py                   # Utility functions for clustering
├── model_sharing                  # Directory for fine-grained model sharing methods and results
│   ├── model                      # Directory for storing trained models
│   ├── result                     # Directory for storing anomaly detection results
│   ├── detection.py               # Anomaly detection code
│   ├── training.py                # Fine-grained model sharing training code
│   ├── TransformerAndMoe.py       # Transformer model training and detection related function
│   └── utils.py                   # Utility functions for data processing and model sharing
├── cluster.sh                     # Coarse-grained clustering command
├── config.yml                     # Configuration file for setting parameters and file paths
├── detect.sh                      # Anomaly detection command
├── requirements.txt               # Environment packages
└── train.sh                       # Fine-grained model sharing training command
```

## Data Description

### Dataset Overview
This open-source dataset is designed for research on anomaly detection. It contains information related to 7 nodes and 17 jobs. The data is divided into training and testing sets, and can be used for developing and validating anomaly detection algorithms.

### Data File Description
The main information in the dataset is stored in a file named `Job`, formatted as follows:

| Field Name | Description |
|------------|-------------|
| JobID      | Unique identifier for each job |
| NodeList   | List of nodes the job ran on |
| Start      | Job start time |
| End        | Job end time |
| Tag        | Indicates whether the job belongs to the training or testing set |

### Sample Data
Below is a snippet from the `Job` file:

```
JobID|NodeList|Start|End|Tag
1|Node-1,Node-3,Node-2,Node-4|2024-07-09T23:13:30|2024-07-10T04:43:49|train
2|Node-1,Node-3,Node-2,Node-4|2024-07-09T00:00:00|2024-07-09T13:40:26|train
...
```

## Installation Steps

1. Download the project
2. Install dependencies:

```
pip install -r requirements.txt
```

Note: Python version is 3.8.10

3. **Configuration before use**:

- **config.yml**: Configuration file containing parameters and paths for data processing and model training.
- **metric.json**: Metrics file containing the metrics involved in training.
- **data**: Data folder storing MTS segments for coarse-grained clustering, fine-grained model sharing training, and anomaly detection.
- **jobinfo**: Job scheduling list, including job state information.

## How to Use

1. **Data Preparation:** Prepare MTS and configuration files. The data folder should be structured as follows:

```
/data/
├── Job              # File containing all job information
├── Train/           # Training data
│   └── Node-1
│       └── Node-1_Job-1.csv
│       └── Node-1_Job-2.csv
│   └── Node-2
│       ...
└── Test/            # Testing data
    └── Node-1_Job-13.csv
    ...
```

2. **Coarse-grained Clustering:** For example, execute `python clustering.py --job_num 20 --nojob_num 10 --node_num 10` to perform coarse-grained clustering. This will output the clustering results, cluster centers and feature indices for each cluster, feature weights, etc. We have encapsulated this command into the `cluster.sh` and can directly execute `sh cluster.sh`.
3. **Fine-grained Model Sharing Training:** Execute `python training.py --max_epochs 30 --batch_size 50 --window_size 20 --nhead 3 --num_layers 3 --num_experts 3 --k 1` to perform fine-grained model sharing training and save the trained model. We have encapsulated this command into the `train.sh` and can directly execute `sh train.sh`.
4. **Anomaly Detection:** Execute `python detection.py --interval 10 --match_len 200` to perform anomaly detection. This will output the model reconstructed data and anomaly score for each node. We have encapsulated this command into the `detect.sh` and can directly execute `sh detect.sh`.
