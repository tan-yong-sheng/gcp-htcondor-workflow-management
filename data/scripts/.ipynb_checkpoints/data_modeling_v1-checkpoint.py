#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Modeling with Logistic Regression for Loan Prediction")
print("------------")

## Objective: Train our previous best model for deployment
# Import necessary libraries
import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

base_dir = Path("/home/tanyongsheng_net/data")
staging_dir = base_dir / "staging"
split_data_dir = staging_dir / "split_data"
staging_model_dir = staging_dir / "model"

# Create necessary filepath directories
os.makedirs(staging_model_dir, exist_ok=True)

# Load previously splitted train data
print("Loading training data...")
X_train = pd.read_csv(split_data_dir / "X_train.csv")
y_train = pd.read_csv(split_data_dir / "y_train.csv").squeeze()

# Define a pipeline for modeling
print("Defining the model pipeline...")
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the numerical data via z-score normalization
    ('model', LogisticRegression()) # Step 2: Train the model
])
## Fit the pipeline to the training data
print("Performing modeling...")
pipe.fit(X_train, y_train)

# Export the trained model
print("Saving the trained model...")
pickle.dump(pipe, open(staging_model_dir / 'trained_model_v1.pkl', 'wb'))
