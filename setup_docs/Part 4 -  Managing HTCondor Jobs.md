# Part 4 -  Managing HTCondor Jobs via Jupyter notebook
Part 4: Managing HTCondor Jobs via Jupyter notebook
---------------------------------------------------

Pre-requisites
--------------

1\. In your **HTCondor Submit**, 

*   make sure you have install Python dependencies:

```text-plain
> pip install htcondor==24.0.1
```

*   maker sure you've installed NFS server in HTCondor Submit as outlined in Part 2 Guide.

2\. In your **HTCondor Executor**,  

*   make sure you have install Python dependencies:

```text-plain
> sudo apt update
> sudo apt install python3-pip -y
> sudo pip install pandas scikit-learn
```

*   make sure you have install NFS common in HTCondor Executor as outlined in Part 2 Guide.

Part 1: Create Python executable files via Jupyter Notebook
-----------------------------------------------------------

### Step 1: Data Processing

*   We're using Jupyter notebook to create a Python file for data processing at `/home/tanyongsheng_net/data/scripts/data_processing.py`

```text-plain
%%writefile ./data/scripts/data_processing.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/data_processing.py` is executable.

```text-plain
# make sure the python script is executable
!chmod 764 ./data/scripts/data_processing.py
```

*   The sample for the above steps is as follows: 

![](/images/2_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

### Step 2: Train test split

*   We're using Jupyter notebook to create a Python file for train test split at `/home/tanyongsheng_net/data/scripts/train_test_split_data.py`

```text-plain
%%writefile ./data/scripts/train_test_split_data.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/train_test_split_data.py` is executable.

```text-plain
# make sure the python script is executable
!chmod 764 ./data/scripts/train_test_split_data.py
```

### Step 3: Modeling - Logistic Regression

*   We're using Jupyter notebook to create a Python file for our Logistic Regression modeling at `/home/tanyongsheng_net/data/scripts/data_modeling_v1.py`

```text-plain
%%writefile ./data/scripts/data_modeling_v1.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/data_modeling_v1.py` is executable.

```text-plain
# make sure the python script is executable
!chmod 764 ./data/scripts/data_modeling_v1.py
```

### Step 4: Modeling - Decision Tree

*   We're using Jupyter notebook to create a Python file for our Decision Tree modeling at `/home/tanyongsheng_net/data/scripts/data_modeling_v2.py`

```text-plain
%%writefile ./data/scripts/data_modeling_v2.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/data_modeling_v2.py` is executable.

```text-plain
# make sure the python script is executable
!chmod 764 ./data/scripts/data_modeling_v2.py
```

### Step 5: Evaluate trained model for Logistic Regression

*   We're using Jupyter notebook to create a Python file for evaluate our Decision Tree modeling with test data at `/home/tanyongsheng_net/data/scripts/evaluate_model_v1.py`

```text-plain
%%writefile ./data/scripts/evaluate_model_v1.py
#!/usr/bin/env python3
print("------------")
print("06-ml-pipeline-in-htcondor-executor.ipynb")
print("Task description: Evaluate the model v1 with test data")
print("------------")

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
metrics_file = metrics_dir / "model_metrics_v1.json"

# Ensure the folders exist
print("Creating necessary directories...")
os.makedirs(staging_dir, exist_ok=True)
os.makedirs(split_data_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(staging_model_dir, exist_ok=True)

## Load the trained model
print("Loading the 1st trained model...")
pipe = pickle.load(open(staging_model_dir / 'trained_model_v1.pkl', 'rb'))
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
    'Model name': 'Logistic Regression',
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/evaluate_model_v1.py` is executable.

```text-plain
!chmod 764 ./data/scripts/evaluate_model_v1.py
```

### Step 6: Evaluate trained model for Decision Model

*   We're using Jupyter notebook to create a Python file for evaluate our Decision Tree modeling with test data at `/home/tanyongsheng_net/data/scripts/evaluate_model_v2.py`

```text-plain
%%writefile ./data/scripts/evaluate_model_v2.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/evaluate_model_v2.py` is executable.

```text-plain
!chmod 764 ./data/scripts/evaluate_model_v2.py
```

### Step 7: Deploy best model

*   We're using Jupyter notebook to create a Python file to deploy the best model between Logistic Regression and Decision Tree model at `/home/tanyongsheng_net/data/scripts/deploy_model.py`.

```text-plain
%%writefile ./data/scripts/deploy_model.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/deploy_model.py` is executable.

```text-plain
!chmod 764 ./data/scripts/deploy_model.py
```

### Step 8: Predict with deployed model

*   We're using Jupyter notebook to create a Python file to perform prediction via the best model at `/home/tanyongsheng_net/data/scripts/predict_model.py`

```text-plain
%%writefile ./data/scripts/predict_model.py
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
```

*   Also, make sure the python script: `/home/tanyongsheng_net/data/scripts/predict_model.py` is executable.

```text-plain
!chmod 764 ./data/scripts/predict_model.py
```

Part 2: Create DAG file for workflow management
-----------------------------------------------

At last, we're using Python to create a DAG file to manage workflow of our machine learning training task.

```python
import htcondor
import os
from htcondor import dags
from pathlib import Path
import shutil

base_dir = Path("/home/tanyongsheng_net/data")

# Function wrapper to create submit file
def create_submit_file(task_name, request_cpus=1, request_memory="128MB", request_disk="128MB", **kwargs):
    sub = htcondor.Submit({
        "executable": base_dir / f"scripts/{task_name}.py",  # Use bash to execute shell commands
        "request_cpus": request_cpus,            # Number of CPU cores required
        "request_memory": request_memory,      # Memory required
        "request_disk": request_disk,        # Disk space required
        "output": base_dir / f"output/{task_name}.out",  # Standard output file
        "error": base_dir / f"error/{task_name}.err",    # Standard error file
        "log": base_dir / f"log/{task_name}.log",        # Log file
    })
    # Update default parameters with any additional kwargs
    sub.update(kwargs)
    return sub

data_processing_sub = create_submit_file(task_name="data_processing")
train_test_split_sub = create_submit_file(task_name="train_test_split_data")
data_modeling_v1_sub = create_submit_file(task_name="data_modeling_v1", request_memory="200MB", request_disk="128MB")
data_modeling_v2_sub = create_submit_file(task_name="data_modeling_v2", request_memory="200MB", request_disk="128MB")

evaluate_model_v1_sub = create_submit_file(task_name="evaluate_model_v1")
evaluate_model_v2_sub = create_submit_file(task_name="evaluate_model_v2")
deploy_model_sub = create_submit_file(task_name="deploy_model")
predict_model_sub = create_submit_file(task_name="predict_model")


def create_dag():
    # Initialize the DAG
    dag = dags.DAG()

    # Define job layers for each task
    ## Task 1: data processing
    data_processing_layer = dag.layer(
        name="data_processing",
        submit_description=data_processing_sub
    )

    ## Task 2: train test split
    train_test_split_layer = dag.layer(
        name="train_test_split",
        submit_description=train_test_split_sub 
    )
    train_test_split_layer.add_parents([data_processing_layer])
    
    ## Data modeling v1 and v2 concurrently
    ## Task 3: Data modeling v1
    data_modeling_v1_layer = dag.layer(
        name="modeling_v1",
        submit_description=data_modeling_v1_sub
    )
    data_modeling_v1_layer.add_parents([train_test_split_layer])

    # Task 4: Data modeling v2
    data_modeling_v2_layer = dag.layer(
        name="modeling_v2",
        submit_description=data_modeling_v2_sub
    )
    data_modeling_v2_layer.add_parents([train_test_split_layer])

    # Task 5: Evaluate trained model v1
    evaluate_model_v1_layer = dag.layer(
        name="evaluate_model_v1",
        submit_description=evaluate_model_v1_sub
    )
    evaluate_model_v1_layer.add_parents([data_modeling_v1_layer])

    # Task 6: Evaluate trained model v2
    evaluate_model_v2_layer = dag.layer(
        name="evaluate_model_v2",
        submit_description=evaluate_model_v2_sub
    )
    evaluate_model_v2_layer.add_parents([data_modeling_v2_layer])

    # Task 7: Deploy best model
    deploy_model_layer = dag.layer(
        name="deploy_best_model",
        submit_description=deploy_model_sub
    )
    deploy_model_layer.add_parents([evaluate_model_v1_layer, evaluate_model_v2_layer])

    # Task 8: Predict with best model
    predict_model_layer = dag.layer(
        name="loan_prediction",
        submit_description=predict_model_sub
    )
    predict_model_layer.add_parents([deploy_model_layer])
    return dag

dag = create_dag()

# Set up the DAG directory
# Write the DAG to disk
dag_dir = os.path.abspath("./data/dags/")
os.makedirs(dag_dir, exist_ok=True)
dag_file = dags.write_dag(dag, dag_dir)
```

After executing the above scripts in Jupyter notebook, `dagfile.dag` will be created:

*   File location: `/home/tanyongsheng_net/data/dags/dagfile.dag`

![](/images/4_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

Some other submit files (`.sub`) will also be created as follows:

![](/images/Part%204%20-%20%20Managing%20HTCondor%20Jo.jpg)

*   File location: `/home/tanyongsheng_net/data/dags/data_processing.sub`

![](/images/5_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/train_test_split.sub`

![](/images/6_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/modeling_v1.sub`

![](/images/7_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/modeling_v2.sub`

![](/images/8_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/evaluate_model_v1.sub`

![](/images/9_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/evaluate_model_v2.sub`

![](/images/10_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/deploy_best_model.sub`

![](/images/11_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

*   File location: `/home/tanyongsheng_net/data/dags/loan_prediction.sub`

![](/images/12_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

```text-plain
> cd ~/data/dags
> condor_submit_dag -f dagfile.dag
```

![](/images/15_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

```text-plain
> watch -n 1 condor_q
```

![](/images/16_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

Result
------

According to the diagram below, we have successfully run all 8 tasks via this DAGMAN workflow management tool.

*   File location: `~/data/dags/dagfile.dag.metrics`

![](/images/14_Part%204%20-%20%20Managing%20HTCondor%20Jo.png)

References
----------

*   **Submitting and Managing Jobs via jupyter notebook:** [https://notebook.community/htcondor/htcondor/docs/apis/python-bindings/tutorials/Submitting-and-Managing-Jobs](https://notebook.community/htcondor/htcondor/docs/apis/python-bindings/tutorials/Submitting-and-Managing-Jobs) 
*   **Submitting and Managing Jobs** [https://htcondor.readthedocs.io/en/latest/apis/python-bindings/tutorials/Submitting-and-Managing-Jobs.html](https://htcondor.readthedocs.io/en/latest/apis/python-bindings/tutorials/Submitting-and-Managing-Jobs.html) 
*   **Advanced Job Submission and Management** [https://htcondor.readthedocs.io/en/latest/apis/python-bindings/tutorials/Advanced-Job-Submission-And-Management.html](https://htcondor.readthedocs.io/en/latest/apis/python-bindings/tutorials/Advanced-Job-Submission-And-Management.html)