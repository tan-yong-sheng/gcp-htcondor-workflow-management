# Docker Runtime Setup Guide on HTCondor
Docker Runtime On HTCondor Executor Machines:
---------------------------------------------

Actually I don't like to use `sudo pip install …` on the HTCondor Executors, because othe security reasons.

1\. Install Docker in HTCondor Executors

```bash
# Add Docker's official GPG key:
> sudo apt-get update
> sudo apt-get install ca-certificates curl
> sudo install -m 0755 -d /etc/apt/keyrings
> sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
> sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
> echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
> sudo apt-get update
```

```bash
> sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

2\. Verify that the installation is successful by running the `hello-world` image:

```text-plain
> sudo service docker start
> sudo docker run hello-world
```

![](/images/Docker%20Runtime%20Setup%20Guide%20on.png)

![](/images/1_Docker%20Runtime%20Setup%20Guide%20on.png)

3\. Create the directory: `~/ml_py310_image`

```bash
> mkdir -p ~/ml_py310_image
> nano ~/ml_py310_image/Dockerfile
```

*   File location: `~/ml_py310_image/Dockerfile`

```bash
# Use an official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# copy requirements.txt file inside the image and
# pip install everything
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
```

*   File location: `requirements.txt`

```bash
# requirements.txt
scikit-learn==1.6.0
pandas==2.2.3
numpy==2.2.1
```

4\. Building the Docker's Python image:

```bash
> cd ~/ml_py310_image
> sudo docker build -t pyenv310:v1 .
```

```bash
> sudo docker images
```

![](/images/2_Docker%20Runtime%20Setup%20Guide%20on.png)

*    Check which groups that the user `condor` is a part of:

```text-plain
> groups condor
```

![](/images/3_Docker%20Runtime%20Setup%20Guide%20on.png)

*   Check if the `condor` user has the permission to run Docker service

```text-plain
> sudo -u condor docker run hello-world
```

And ya, it's permission error!

![](/images/4_Docker%20Runtime%20Setup%20Guide%20on.png)

Let's try to add `condor` user in `docker` group 

```text-plain
> sudo usermod -aG docker condor
> sudo systemctl restart condor
```

![](/images/5_Docker%20Runtime%20Setup%20Guide%20on.png)

*   Yah! Now, it's working!

```text-plain
> sudo -u condor docker run hello-world
```

![](/images/6_Docker%20Runtime%20Setup%20Guide%20on.png)

How HTCondor can utilize this Docker runtime for its HTCondor jobs
------------------------------------------------------------------

To use Docker as your runtime, you can just can create a submit file with ‘Docker’ universe instead of ‘vanilla’ at `/home/tanyongsheng_net/data/scripts/train_iris_classifier/train_iris_classifer.sub`

```text-plain
universe                = docker
docker_image            = pyenv310:v1
executable              = ./data/scripts/train_iris_classifier.py
arguments               = /etc/hosts
should_transfer_files   = YES
transfer_input_files     = ./data/scripts/train_iris_classifier.py
when_to_transfer_output = ON_EXIT
request_cpus            = 1
request_memory          = 128MB
request_disk            = 128MB
output                  = ./data/output/train_iris_classifier.out
error                   = ./data/error/train_iris_classifier.err
log                     = ./data/log/train_iris_classifier.log

queue
```

*   Here is the Python script at `/home/tanyongsheng_net/data/scripts/train_iris_classifier.py`, to be executed via the submit file above

```text-plain

#!/usr/bin/env python3
# Import necessary libraries
print("------------")
print("03-running-sklearn-with-docker-in-condor-executor.ipynb")
print("------------")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for algorithms like Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Task in Docker runtime is completed!")
```

However, we didn't explore the possibility of Docker runtime to access to the NFS filesystem. It might be an interesting topic to explore in the future.

References
----------

*   **Using Docker on HTCondor** [https://abpcomputing.web.cern.ch/guides/docker\_on\_htcondor/](https://abpcomputing.web.cern.ch/guides/docker_on_htcondor/) 
*   **Exercise 9a: Docker Universe** [https://batchdocs.web.cern.ch/tutorial/exercise9a.html](https://batchdocs.web.cern.ch/tutorial/exercise9a.html) 
*   **HTCondor: Containers** [https://batchdocs.web.cern.ch/containers/index.html](https://batchdocs.web.cern.ch/containers/index.html) 
*   **Docker Universe Applications** [https://htcondor.readthedocs.io/en/v9\_0/users-manual/docker-universe-applications.html](https://htcondor.readthedocs.io/en/v9_0/users-manual/docker-universe-applications.html)
    
*   **Build a Docker Container Image** [https://chtc.cs.wisc.edu/uw-research-computing/docker-build.html](https://chtc.cs.wisc.edu/uw-research-computing/docker-build.html)
    
*   **Running HTC Jobs Using Docker Containers** [https://chtc.cs.wisc.edu/uw-research-computing/docker-jobs](https://chtc.cs.wisc.edu/uw-research-computing/docker-jobs)