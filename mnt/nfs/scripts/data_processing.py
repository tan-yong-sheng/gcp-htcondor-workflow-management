#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Load data from NFS and perform data pre-processing")
print("------------")

# Import necessary libraries
import os
import pandas as pd
from pathlib import Path

base_dir = Path("/home/tanyongsheng_net/data")
staging_dir = base_dir / "staging"
split_data_dir = staging_dir / "split_data"
CSV_file = base_dir / "loan_data.csv"

# Create necessary filepath directories
os.makedirs(staging_dir, exist_ok=True)
os.makedirs(split_data_dir, exist_ok=True)

# Part 1: load data from NFS
print("Loading data from NFS...")
data = pd.read_csv(CSV_file)

# Part 2: Data Preprocessing
## 2.1 Handling Missing Values
  ## To identify all the numeric columns in the dataset, compute the mean value of each column,
  ## and fill the missing values (NaN) in each column with the computed mean value.
print("Handling missing values...")
numeric_columns = data.select_dtypes(include=['number']).columns
for col in numeric_columns:
    mean_value = data[col].mean()
    data[col] = data[col].fillna(mean_value)

## 2.2 Encoding Categorical Variables
print("Encoding categorical variables...")
data = pd.get_dummies(data, drop_first=True)

## Part 3: Export cleaned data
print("Saving processed data to NFS disk...")
data.to_csv(split_data_dir / "processed_data.csv")
