## Machine Learning Workflow Management via HTCondor

### Project Background
Our team is training a loan prediction model to classify which customer has a risk of default. Because the data might be growing large in the future, our team is preparing to move the machine learning training task to a distributed environment.

### Introduction to our Project Architecture
To manage the training of our loan prediction machine learning model, we've implemented a workflow using HTCondor. Before delving into our machine learning workflow, let's first examine the architecture of HTCondor itself.

- HTCondor consists of a central manager and a submission node, with multiple execution nodes coordinated through a central negotiation process. At the core of this architecture is a Condor High-Throughput Computing framework that depends on major components: 
- HTCondor Submit Node (Scheduler): This is where we send our jobs to the HTCondor system. We define our training jobs using submit files, which specify how to execute our Python scripts that handle the model training process. We also use this Submit Node to run our Jupyter Notebooks for work planning and development.
- HTCondor Central Manager (Negotiator and Collector): The Central Manager houses both the Negotiator and Collector, which are responsible for task allocation and resource management. The Negotiator matches our training jobs to available execution nodes, and the Collector monitors available resources across all worker nodes.
- HTCondor Execution Node (startd): Each Execution Node is where we run the Python scripts to carry out the core model training process on the available worker nodes, after the tasks are submitted to the HTCondor system via HTCondor Submit.
Shared Storage (NFS): All nodes in our HTCondor setup access shared data using NFS. This ensures data accessibility and integrity, allowing our Python scripts to read input data, write results, and access any necessary files from any node, essential for distributed training.

Such a setup easily allows collaboration between submission, execution, and management components while maintaining shared-resource integrity. This architecture is particularly well-suited for very scalable computational tasks that require distributed resource utilization. The employment of Jupyter Notebooks on the submission node provides an intuitive, interactive interface to the system, which is ideal for both research and production environments.

![image](https://github.com/user-attachments/assets/0e017405-51c4-410a-a77c-01dab7ae0542)

### Machine Learning Workflow Managment via HTCondor

Our final output is [06-ml-pipeline-in-htcondor-executor.ipynb](06-ml-pipeline-in-htcondor-executor.ipynb), where other jupyter notebook are the intermediate notebook that we divide the job by the expected challenges, as follows:
- [01-intro-to-htcondor-python.ipynb](01-intro-to-htcondor-python.ipynb): Trial to use Python bindings for HTCondor
- [02-running-sklearn-in-condor-executor.ipynb](02-running-sklearn-in-condor-executor.ipynb): Trial to execute Python script with Python libraries like `pandas` and `sklearn` in HTCondor Executor
- [03-running-sklearn-with-docker-in-condor-executor.ipynb](03-running-sklearn-with-docker-in-condor-executor.ipynb): Initiative to explore Docker runtime for HTCondor jobs, we're not using this in the final as still stuck with the access to the NFS filesystem if using this method. See more info at [./setup_docs/Extra/Docker Runtime Setup Guide on.md](./setup_docs/Extra/Docker%20Runtime%20Setup%20Guide%20on.md)
- [04-create-dag-file-via-python.ipynb](04-create-dag-file-via-python.ipynb): Trial to create DAG file via Python library: `htcondor.dags` API
- [05-loading-csv-from-nfs-in-htcondor-executor.ipynb](05-loading-csv-from-nfs-in-htcondor-executor.ipynb): Trial to read and write CSV in HTCondor Executor from shared NFS folder

If you're interested with our step-by-step setup guide, see here:
- [Part 1 - HTCondor's Installation](setup_docs/Part%201%20-%20HTCondor's%20Installation.md)
- [Part 2 - Install NFS File System](setup_docs/Part%202%20-%20Install%20NFS%20File%20System.md)
- [Part 3 - Install Jupyter notebook on HTCondor Submit](setup_docs/Part%203%20-%20Install%20Jupyter%20notebook.md)
- [Part 4 - Managing HTCondor Jobs](setup_docs/Part%204%20-%20%20Managing%20HTCondor%20Jobs.md)

And here are the challenges we've met during the setup. Please feel free to read here: [Part 5 - Challenges we have met](setup_docs/Part%205%20-%20Challenges%20we%20have%20met.md)

--------------------------------------------

So, back to our topic:

- Below is the machine learning workflow we're running via HTCondor ecosystem:

![image](https://github.com/user-attachments/assets/b69c43e1-e780-499b-9258-4d3658b8958d)

The workflow for loan prediction in HTCondor includes the following essential steps:

- Data Input & Preprocessing: NFS stores unprocessed loan data (CSV). Several compute nodes are coordinated by HTCondor to retrieve this data, perform preprocessing (one-hot encoding, feature engineering, cleaning, and handling missing values), and then save the processed data back to NFS.
- Data Partition: The task of splitting the produced data into training (80%) and testing (20%) groups is assigned to HTCondor. Executor Nodes perform the split, retrieve the preprocessed data, and save the processed data in NFS.
- Training Models in a Distributed Manner: One of the Executor nodes will be assigned by HTCondor for Logistic Regression model training. While another Executor node will be assigned by HTCondor for Decision Tree’s training.
- Evaluating Models in a Distributed Manner: Several nodes use the test data to evaluate the trained logistic regression and decision tree models. Metrics like accuracy and recall are evaluated, and the results are aggregated to provide a comprehensive assessment.
- Model Deployment: The highest-performing model, according to evaluation metrics, is saved to the shared NFS folder and thus could be implemented across all nodes.
- Loan Forecasting: Using the developed model, new loan data can be examined across several Executor nodes in order to make predictions.
