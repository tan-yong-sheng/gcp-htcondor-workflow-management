# Part 1 - HTCondor's Installation
-----------------------------------

1.  Create 4 VMs in Google Cloud Platform with the following specifications:
    
    |     |     |     |     |     |     |     |
    | --- | --- | --- | --- | --- | --- | --- |
    | Instance Name | Role | Machine type | vCPUs | Memory (RAM) | Disk Storage | Task to be run |
    | VM1 | Condor Host | e2-medium | 2   | 4 GB | 10 GB | *   Condor Host Program |
    | VM2 | Condor Submit | e2-medium | 1   | 4 GB | 30 GB | *   Condor Submit Program<br>*   Jupyter notebook |
    | VM3 | Condor Executor 01 | e2-medium | 2   | 4 GB | 10 GB | *   Condor Executor Program with Python environment |
    | VM4 | Condor Executor 02 | e2-medium | 2   | 4 GB | 10 GB | *   Condor Executor Program with Python environment |
    

![](/images/10_Part%201%20-%20HTCondor's%20Installati.jpg)

Just a note here, I am using the default network assigned by Google System for all 4 VM instances, without change any settings here.

![](/images/Part%201%20-%20HTCondor's%20Installati.jpg)

Step 1: Set up Central Manager (VM1 - Condor Host)
--------------------------------------------------

1\. SSH into condor-host VM:

![](/images/7_Part%201%20-%20HTCondor's%20Installati.jpg)

2\. Update the Ubuntu's repository:

```bash
> sudo apt update
```

3\. Edit hosts file:

```bash
> sudo nano /etc/hosts
```

*   Add this line to `/etc/hosts` file, but you need to change it to your VMs' internal ip address:

```text-plain
# new ip addresses added
10.128.0.2 CondorHost
10.128.0.6 SubmissionHost
10.128.0.4 Executor01
10.128.0.5 Executor02
```

![](Part%201%20-%20HTCondor's%20Installati.png)

After adding the 4 machine's IP into the `/etc/hosts` file, press `Ctrl + X` and then type `y`

4\. Install HTCondor's [Central manager](https://htcondor.readthedocs.io/en/latest/getting-htcondor/admin-quick-start.html) node:

```text-plain
> curl -fsSL https://get.htcondor.org | sudo GET_HTCONDOR_PASSWORD="abc123" /bin/bash -s -- --no-dry-run --central-manager CondorHost
```

(Note: you could change the password: `abc123` yourselves, to make sure it is consistent when you install HTCondor submit and HTCondor Executor)

Source: [HTCondor's admin guide](https://htcondor.readthedocs.io/en/latest/getting-htcondor/admin-quick-start.html)

5\. Restart HTCondor:

```text-plain
 > sudo systemctl restart condor
 > sudo systemctl status condor
```

![](8_Part%201%20-%20HTCondor's%20Installati.jpg)

### Step 2: Set up Submit Node (VM2)

1\. Again, ssh into HTCondor Submit as we've done before. Then run this command in the terminal to update its Ubuntu's repository

```bash
> sudo apt update
```

2\. Edit hosts file:

```bash
> sudo nano /etc/hosts
```

*   Add this line to `/etc/hosts` file, but you need to change it to your HTCondor Host VM's internal ip address:

```text-plain
# new ip address added
10.128.0.2 CondorHost
```

3\. Install HTCondor's submit node:

```text-plain
%sh
> curl -fsSL https://get.htcondor.org | sudo GET_HTCONDOR_PASSWORD="abc123" /bin/bash -s -- --no-dry-run --submit CondorHost
```

(Note:  you could change the password: `abc123` yourselves, but please make sure the password is the same accross the HTCondor VMs.)

Source: [HTCondor's admin guide](https://htcondor.readthedocs.io/en/latest/getting-htcondor/admin-quick-start.html)

4\. Restart HTCondor:

```text-plain
 %sh
 > sudo systemctl restart condor
 > sudo systemctl status condor
```

![](9_Part 1%20-%20HTCondor's%20Installati.jpg)

### Step 3: Set up Executor Nodes (VM3 and VM4)

Repeat these steps for both executor VMs:

1\. SSH into the executor VM  
2\. Update Ubuntu:

```bash
> sudo apt update
```

3\. Edit hosts file:

```bash
> sudo nano /etc/hosts
```

*   Add this line to `/etc/hosts` file, but you need to change it to your VMs' internal ip address:

```text-plain
# new ip address added
10.128.0.2 CondorHost
```

![](2_Part 1%20-%20HTCondor's%20Installati.png)

5\. Install HTCondor Executor:

```bash
> curl -fsSL https://get.htcondor.org | sudo GET_HTCONDOR_PASSWORD="abc123" /bin/bash -s -- --no-dry-run --execute CondorHost
```

(Note: you could change the password: `abc123` yourselves, but please make sure the password is the same accross the VMs.)

Source: [HTCondor's admin guide](https://htcondor.readthedocs.io/en/latest/getting-htcondor/admin-quick-start.html)

6\. Install Python dependencies that you need for your machine learning task:

```bash
> sudo apt update
> sudo apt install python3-pip -y
> sudo pip install pandas scikit-learn
```

7\. After running those scripts on respective VMs, you will need to run these commands:

```bash
 > sudo usermod -aG condor tanyongsheng_net
 > # sudo usermod -aG condor docker # only when you want to use Docker runtime
 > sudo systemctl restart condor
 > sudo systemctl status condor
```
