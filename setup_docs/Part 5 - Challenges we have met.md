# Part 5 - Challenges we have met
Challenges
----------

1\. Shared NFS storage to be accessed by multiple HTCondor Executors

![](/images/Part%205%20-%20Challenges%20%20we%20have%20me.png)

![](/images/1_Part%205%20-%20Challenges%20we%20have%20me.png)

2\. Using Universe runtime with HTCondor requires `sudo pip install` for Python libraries, which is undesirable due to its modification of the system. While Docker is a preferred option, insufficient documentation hinders its adoption.

3\. HTCondor jobs can become stuck in the 'Idle' state indefinitely due to resource allocation errors, like requesting 100GB of disk space when it's not available. Critically, no error message is generated, obscuring the cause of the problem.

![](/images/2_Part%205%20-%20Challenges%20we%20have%20me.png)

4\. Multi-processing & threading seems not allowed via Python library, e.g.,running `GridSearchCV` from `sklearn` python library will terminate the job. At the end, we give up this implementation as it is hard to debug at the time being.

![](/images/Part%205%20-%20Challenges%20we%20have%20me.jpg)