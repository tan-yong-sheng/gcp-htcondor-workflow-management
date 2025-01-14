# Part 5 - Challenges we have met
----------

1\. Efforts to set up Shared NFS storage to be accessed by HTCondor Submits and multiple HTCondor Executors

![](/images/Part%205%20-%20Challenges%20%20we%20have%20me.png)

![](/images/1_Part%205%20-%20Challenges%20we%20have%20me.png)

2\. Using Universe runtime with HTCondor requires `sudo pip install` for Python libraries, which is undesirable due to its modification of the system. While Docker is a preferred option, insufficient documentation hinders its adoption.

3\. HTCondor jobs can become stuck in the 'Idle' state indefinitely due to resource allocation errors, like requesting 100GB of disk space when it's not available. Critically, no clear error message is generated, make it hard to debug the cause of the problem.

![](/images/2_Part%205%20-%20Challenges%20we%20have%20me.png)

4\. Multi-processing & threading seems not allowed via Python library, e.g.,running `GridSearchCV` from `sklearn` python library will terminate the job. At the end, we give up this implementation as it is hard to debug at the time being.

![](/images/Part%205%20-%20Challenges%20we%20have%20me.jpg)

We may need to manually implement hyperparameter tuning by training multiple models simultaneously with different parameter configurations. Some helpful references include:
- [Practice: Passing Multiple Arguments to Multiple Jobs with One Submit File](https://chtc.cs.wisc.edu/uw-research-computing/htc-passing-arguments-multiple)
- [Submitting Multiple Jobs With
HTCondor](https://osg-htc.org/user-school-2022/materials/htcondor/files/osgus22-htc-htcondor-PART2.pdf)

Based on these resources, our idea is to create a script that handles hyperparameter tuning by customizing the Python code to accept input arguments (or parameters) for modeling. Here’s an example (not yet tested):

- File location: `/home/tanyongsheng_net/data/scripts/hyperparameter_tuning.py`
```python
# using argparse to accept hyperparameters from HTCondor's submit file
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('C', help='Specify an input file', type=float)
parser.add_argument('penalty', help='Specify an output file', type=str)
parser.add_argument('solver', help='Specify an output file', type=str)
parser.add_argument('max_iter', help='Specify an output file', type=int)
args = parser.parse_args()

<.... PYTHON SCRPIT TO TRAIN MODEL HERE WHICH UTILIZE THE HYPERPARAMETERS ABOVE ....>
```

We'll be using queue to loop the arguments row by row to train a single model with different hyperparameters (= hyperparameter tuning):

- File location: `/home/tanyongsheng_net/data/scripts/hyperparameter_tuning.sub`
```
executable = /home/tanyongsheng_net/data/scripts/hyperparameter_tuning.py
arguments = $(arg1) $(arg2) $(arg3) $(arg4)
request_cpus = 2
request_memory = 300MB
request_disk = 300MB
output = /home/tanyongsheng_net/data/output/hyperparameter_tuning.$(Process).out
error = /home/tanyongsheng_net/data/error/hyperparameter_tuning.$(Process).err
log = /home/tanyongsheng_net/data/log/hyperparameter_tuning.log

queue arg1, arg2, arg3, arg4 from /home/tanyongsheng_net/data/scripts/hyperparameter_tuning.txt
```


- File location: `/home/tanyongsheng_net/data/scripts/hyperparameter_tuning.txt`
```
0.01 l1 liblinear 100
0.01 l1 liblinear 200
0.01 l1 liblinear 300
0.01 l2 liblinear 100
0.01 l2 liblinear 200
0.01 l2 liblinear 300
0.1 l1 liblinear 100
0.1 l1 liblinear 200
0.1 l1 liblinear 300
0.1 l2 liblinear 100
0.1 l2 liblinear 200
0.1 l2 liblinear 300
1 l1 liblinear 100
1 l1 liblinear 200
1 l1 liblinear 300
1 l2 liblinear 100
1 l2 liblinear 200
1 l2 liblinear 300
10 l1 liblinear 100
10 l1 liblinear 200
10 l1 liblinear 300
10 l2 liblinear 100
10 l2 liblinear 200
10 l2 liblinear 300
100 l1 liblinear 100
100 l1 liblinear 200
100 l1 liblinear 300
100 l2 liblinear 100
100 l2 liblinear 200
100 l2 liblinear 300
```
