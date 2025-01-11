#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Evaluate the model v2 with test data")
print("------------")

## 6.2 Metrics of model v2
import os
import pickle
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import (accuracy_score, 
                            precision_score, 
                            recall_score, 
                            f1_score)

base_dir = Path("/home/tanyongsheng_net/data")
staging_dir = base_dir / "staging"
split_data_dir = staging_dir / "split_data"
metrics_dir = staging_dir / "metrics"
staging_model_dir = staging_dir / "model"
metrics_file = metrics_dir / "model_metrics_v2.json"

# Ensure the folders exist
print("Creating necessary directories...")
os.makedirs(staging_dir, exist_ok=True)
os.makedirs(split_data_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(staging_model_dir, exist_ok=True)

## Load the trained model
print("Loading the 1st trained model...")
pipe = pickle.load(open(staging_model_dir / 'trained_model_v2.pkl', 'rb'))
print("Model loaded successfully!")

# Load the test data
print("Loading the test data...")
X_test = pd.read_csv(split_data_dir / "X_test.csv")
y_test = pd.read_csv(split_data_dir / "y_test.csv")
print(f"Test data loaded: {X_test.shape[0]} samples")


## Make predictions using the pipeline
print("Making predictions using the trained model...")
y_pred = pipe.predict(X_test)
print("Predictions made!")

## Calculate multiple metrics
print("Calculating evaluation metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # 'binary' for binary classification
recall = recall_score(y_test, y_pred, average='binary')       # 'binary' for binary classification
f1 = f1_score(y_test, y_pred, average='binary')               # 'binary' for binary classification

## Export all metrics
metrics = {
    'Model name': 'Decision Tree',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
}
print(metrics)

# Save the metrics to a JSON file
print("Saving metrics to JSON file...")

with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved at {metrics_dir}!")
