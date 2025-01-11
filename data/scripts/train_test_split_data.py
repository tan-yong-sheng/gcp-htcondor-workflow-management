#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Perform train test split")
print("------------")

# Import necessary libraries
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

base_dir = Path("/home/tanyongsheng_net/data")
staging_dir = base_dir / "staging"
split_data_dir = staging_dir / "split_data"

# Create necessary filepath directories
os.makedirs(staging_dir, exist_ok=True)
os.makedirs(split_data_dir, exist_ok=True)

data = pd.read_csv(split_data_dir / "processed_data.csv")

## Define the target and features
X = data.drop('loan_status', axis=1)  # 'loan_status' is the target column
y = data['loan_status']

# Part 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Export train test split data
X_train.to_csv(split_data_dir / "X_train.csv", index=False)
y_train.to_csv(split_data_dir / "y_train.csv", index=False)
X_test.to_csv(split_data_dir / "X_test.csv", index=False)
y_test.to_csv(split_data_dir / "y_test.csv", index=False)
