
# Cluster Adjustment and Anomaly Labeling Software for High-Performance Computing Systems

## Project Overview

This project is a toolkit for clustering adjustment and anomaly detection in high-performance computing systems. It provides a flexible interface operation where users can execute different clustering and annotation strategies by clicking buttons. The project provides sample algorithms for clustering and anomaly detection, and users can modify or add their own clustering methods and models.

## Directory Structure

```
├── communicate_files        # Directory for communication
│   └── annotation_history.txt  # File for caching anomaly detection results
├── config_files             # Configuration files
│   ├── cluster_adjust.txt   # Save the adjusted clustering results
│   ├── cluster_result.txt   # Classification results of the selected clustering method
│   ├── metric_used.txt      # Metric data
│   └── time_scope.txt       # Time step data
├── labels                 # Directory for storing anomaly detection annotated data, named after the original data
├── model                  # Directory for storing machine learning models
├── node_data              # Directory containing time series data, each CSV file corresponds to a node's time series data
│   └── node_x.csv         # Each CSV file contains timestamps in the first column and other metrics in subsequent columns
├── reference_cluster       # Sample implementations of clustering algorithms, users can modify or add new methods
│   ├── DBSCAN.py           # DBSCAN clustering algorithm
│   ├── HAC.py              # Hierarchical clustering algorithm
│   └── k-means.py          # K-Means clustering algorithm
├── reference_models        # Sample anomaly detection methods, users can modify or add new methods
│   ├── 3-sigma.py          # Anomaly detection based on 3-Sigma rule
│   ├── modify.py           # Re-annotate the data that has already been marked
│   └── USAD.py             # Anomaly detection based on USAD model
├── ListPicker.py           # Implementation of list picker
├── ListSignalPicker.py     # Implementation of signal list picker
├── main.py                 # Main program file
├── MultiCanvas.py          # Tool for multi-canvas display
├── PlotCanvas2.py          # Data plotting tool
├── scrolled_frame.py       # Scrollable frame implementation
└── SmallListSignalPicker.py # Small signal picker implementation
```

## Main Funtions

### 1. Clustering Algorithms
- **DBSCAN**: A density-based clustering algorithm suitable for discovering clusters of arbitrary shapes and handling noise data.
- **K-Means**: A classic clustering algorithm that divides the data into a predefined number of clusters.
- **Hierarchical Clustering (HAC)**: Uses hierarchical clustering to discover the hierarchical structure in the data.

> **Note**: The above clustering algorithms are located in the `reference_cluster` folder. Users can modify or add other clustering algorithms as needed.

### 2. Anomaly Detection Algorithms
- **3-Sigma**: Based on the traditional 3-Sigma rule, marking points that deviate more than three standard deviations from the mean as anomalies (sample).
- **USAD**: A model designed for unsupervised anomaly detection in time series data.
- **Modify**: Allows re-annotation and modification of already labeled data.

> **Note**: These anomaly detection algorithms are located in the `reference_models` folder. Users can modify or add other detection algorithms as needed.

### 3. Data Visualization
- **PlotCanvas2**: Supports plotting time series data, allowing users to visualize data based on different metrics.
- **ScrollableCanvas**: Implements a scrollable multi-canvas view, suitable for displaying high-dimensional data.

## Data Description

The data in the `node_data` folder is the core data used for clustering and anomaly detection in this project. These data are stored in CSV format, and each CSV file represents the time series data of one node.

- **First column**: Represents the timestamp, recording the time of the data points.
- **Other columns**: Represent the values of different metrics, with each column corresponding to a time series feature, such as system performance metrics, sensor data, etc.

Example:

```
timestamp, metric1, metric2, metric3
2024-01-01 00:00:00, 0.5, 0.7, 0.2
2024-01-01 00:01:00, 0.6, 0.8, 0.3
...
```

 Note*: Since the original data cannot be disclosed, the node_data folder contains generated fake data.

## Installation Steps

1. Download the project:

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Note*: Some model dependencies may require manual configuration.

3. **Pre-use configuration**:

- **metric_used.txt**: Configuration file containing the metrics for clustering and annotation.
- **time_scope.txt**: Contains the time indices for the time series data.
- **node_data**: Folder where time series data that needs to be labeled for anomalies or clustered is placed.

## How to Use

### 1. Run the main program

After installing the dependencies, simply run the following command to start the main program:

```bash
python main.py
```

All clustering methods and anomaly detection models can be operated through the program interface. There is no need to input commands through the terminal.

### 2. Operation Instructions

#### **1. Annotation Operation**
- In the interface, select the time series data from the `node_data` folder.
- Use the `Display` button to switch between time axis and data segmentation views.
- Choose the metric to annotate, apply annotation strategies, or manually annotate.
    - The mouse wheel can zoom the view, and the left mouse button can drag the subplot.
    -  The mouse wheel: The first click sets the start of the annotation, and the second click sets the end of the annotation.
    - Right-click: Used for deleting annotations, the first click sets the start of the deletion, and the second click sets the end of the deletion.

#### **2. Clustering Operation**
- The clustering results are saved in the `config_files/cluster_data.txt` file.
- You can choose different clustering strategies to cluster the node data in the `node_data` folder.
- Click on a category to view the changes of different nodes under the same metric.
- Click on a node to highlight it in the graph.
- In the top right corner, you can assign the selected node to a new category, and the results will be saved in the `cluster_data_save.txt` file.

## Contribution Guide

If you have any suggestions or improvements for this project, feel free to submit a `Pull Request` or `Issue`. We welcome your contributions!
