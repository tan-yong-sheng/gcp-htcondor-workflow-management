#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Decision tree for Loan Prediction")
print("------------")

import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

base_dir = Path("/home/tanyongsheng_net/data")
staging_dir = base_dir / "staging"
split_data_dir = staging_dir / "split_data"
staging_model_dir = staging_dir / "model"

# Create necessary filepath directories for data export
os.makedirs(staging_model_dir, exist_ok=True)

# Load previously splitted train data
print("Loading training data...")
X_train = pd.read_csv(split_data_dir / "X_train.csv")
y_train = pd.read_csv(split_data_dir / "y_train.csv").squeeze()

# Define a pipeline for modeling
print("Defining the model pipeline...")
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the numerical data via z-score normalization
    ('model', DecisionTreeClassifier()) # Step 2: Train the model
])

## Fit the pipeline to the training data
print("Performing modeling...")
pipe.fit(X_train, y_train)

# Export the trained model
print("Saving the model v2...")
pickle.dump(pipe, open(staging_model_dir / 'trained_model_v2.pkl', 'wb'))
