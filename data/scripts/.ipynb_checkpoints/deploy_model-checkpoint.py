#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Deploy best model for production")
print("------------")

import os
import pandas as pd
import shutil
import json
from pathlib import Path

base_dir = Path("/home/tanyongsheng_net/data")
staging_dir = base_dir / "staging" 
model_dir = base_dir / "model"
staging_model_dir = staging_dir / "model"
metrics_dir = staging_dir / "metrics"

# Create necessary folder path
os.makedirs(model_dir, exist_ok=True)
print(f"Model directory created or already exists: {model_dir}")

# Read metrics file for both trained models
# Read metrics files for both trained models
print("Loading metrics for trained models...")
with open(metrics_dir / "model_metrics_v1.json", 'r') as f:
    trained_metrics_v1 = json.load(f)
with open(metrics_dir / "model_metrics_v2.json", 'r') as f:
    trained_metrics_v2 = json.load(f)

# deploy model with higher precision value
if trained_metrics_v1["Precision"] > trained_metrics_v2["Precision"]:
    model_to_deploy = "trained_model_v1.pkl"
    print("Deployed model v1 based on higher precision.")
elif trained_metrics_v1["Precision"] < trained_metrics_v2["Precision"]:
    model_to_deploy = "trained_model_v2.pkl"
    print("Deployed model v2 based on higher precision.")
else:
    if trained_metrics_v1["F1 Score"] >= trained_metrics_v2["F1 Score"]:
        model_to_deploy = "trained_model_v1.pkl"
        print("Deployed model v1 based on higher F1 score as a tie-breaker.")
    else:
        model_to_deploy = "trained_model_v2.pkl"
        print("Deployed model v2 based on higher F1 score as a tie-breaker.")

# Copy the selected model to the production directory
print(f"Copying {model_to_deploy} to the production model directory...")
shutil.copy(staging_model_dir / model_to_deploy, model_dir / "deployed_model.pkl")
print(f"Deployed model saved as 'deployed_model.pkl' in {model_dir}!")
