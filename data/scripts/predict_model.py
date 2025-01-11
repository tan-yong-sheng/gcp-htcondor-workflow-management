#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Predict with deployed model")
print("------------")

import os
import pickle
import pandas as pd
from pathlib import Path

base_dir = Path("/home/tanyongsheng_net/data")
model_dir = base_dir / "model"
staging_dir = base_dir / "staging" 
split_data_dir = staging_dir / "split_data"

# Ensure the test data files exist
if not split_data_dir.exists():
    print(f"Error: {split_data_dir} does not exist.")
    exit(1)

print("Loading the test data...")
X_test = pd.read_csv(split_data_dir / "X_test.csv")
y_test = pd.read_csv(split_data_dir / "y_test.csv")

# Load the saved model
print("Loading the deployed model...")
with open(model_dir / 'deployed_model.pkl', 'rb') as f:
    deployed_model = pickle.load(f)
    print("Model loaded successfully!")
    
# Make predictions
print("Making predictions...")
y_pred = deployed_model.predict(X_test)
print(f"Predictions: {y_pred}")

print("Task completed!")
