## Machine Learning Workflow Management via HTCondor

Our final output is [06-ml-pipeline-in-htcondor-executor.ipynb](06-ml-pipeline-in-htcondor-executor.ipynb), where other jupyter notebook are the intermediate notebook that we divide by the job by the expected challenges, as follows:
- [01-intro-to-htcondor-python.ipynb](01-intro-to-htcondor-python.ipynb): Trial to use Python bindings for HTCondor
- [02-running-sklearn-in-condor-executor.ipynb](02-running-sklearn-in-condor-executor.ipynb): Trial to execute Python script with Python libraries like `sklearn` in HTCondor Executor
- [03-running-sklearn-with-docker-in-condor-executor.ipynb](03-running-sklearn-with-docker-in-condor-executor.ipynb): Initiative to explore Docker runtime for HTCondor jobs, we're not using this in the final as still stuck with the access to the NFS filesystem if using this method. See more info at [./setup_docs/Extra/Docker Runtime Setup Guide on.md](./setup_docs/Extra/Docker Runtime Setup Guide on.md)
- [04-create-dag-file-via-python.ipynb](04-create-dag-file-via-python.ipynb): Trial to create DAG file via Python library: `htcondor.dags` API
- [05-loading-csv-from-nfs-in-htcondor-executor.ipynb](05-loading-csv-from-nfs-in-htcondor-executor.ipynb): Trial to read and write CSV in HTCondor Executor

So, back to our topic:

> Below is the machine learning workflow we're running via HTCondor ecosystem:

![image](https://github.com/user-attachments/assets/b69c43e1-e780-499b-9258-4d3658b8958d)
