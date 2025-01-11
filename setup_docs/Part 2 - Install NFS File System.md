# Part 2 - Install NFS File System
Part 2: NFS File System Setup Guide
-----------------------------------

Step 1: Configure NFS Server (in HTCondor Submit Node)
------------------------------------------------------

1\. Install NFS server packages:

```text-plain
> sudo apt update
> sudo apt install nfs-kernel-server
```

![](4_Part 2 - Install NFS File Syst.jpg)

2\. Create the shared directory:

```text-plain
> sudo mkdir -p /home/tanyongsheng_net/data
```

3\. Configure NFS exports:

```text-plain
> sudo nano /etc/exports
```

*   Add this line to `/etc/exports` file:

```text-plain
/home/tanyongsheng_net/data *(rw,sync,no_subtree_check,no_root_squash)
```

![](15_Part 2 - Install NFS File Syst.png)

4\. Apply the NFS configuration:

```text-plain
> sudo exportfs -a
```

![](1_Part 2 - Install NFS File Syst.jpg)

5\. Restart NFS service:

```text-plain
> sudo systemctl restart nfs-kernel-server
> sudo systemctl status nfs-kernel-server
```

6\. Set up permissions:

```text-plain
> sudo mkdir /home/nobody
> sudo chown nobody:nogroup /home/nobody/
> sudo chmod 777 /home/nobody/

> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net
> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net/data
> sudo chmod 777 /home/tanyongsheng_net
```

*   Change permission access to Condor group

```text-plain
> sudo mkdir /home/nobody
> sudo chown nobody:nogroup /home/nobody/
> sudo chmod 777 /home/nobody/

> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net
> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net/data
> sudo chmod 777 /home/tanyongsheng_net
> sudo chmod 777 /home/tanyongsheng_net/data
```

*   Test if this works for `Condor` group to create and write the file

```text-plain
> sudo -u condor touch /home/tanyongsheng_net/data/testfile.txt
```

![](17_Part 2 - Install NFS File Syst.png)

### Step 2: Configure NFS Common (in HTCondor Executor Nodes)

1\. Install NFS common packages:

```text-plain
> sudo apt update
> sudo apt install nfs-common
```

*   Edit hosts file:

```text-plain
> sudo nano /etc/hosts
```

![](16_Part 2 - Install NFS File Syst.png)

```text-plain
> sudo mkdir -p /home/tanyongsheng_net/data
```

![](10_Part 2 - Install NFS File Syst.png)

```text-plain
> sudo mount CondorSubmit:home/tanyongsheng_net/data /home/tanyongsheng_net/data/
```

![](13_Part 2 - Install NFS File Syst.png)

### Bonus tips: Auto-mounting for HTCondor Executor

An alternate way to mount an NFS share in HTCondor Executor is to add a line to the `/etc/fstab` file. The line must state the hostname of the NFS server, the directory on the server being exported, and the directory on the local machine where the NFS share is to be mounted.

The general syntax for the line in `/etc/fstab` file is as follows:

*   File location: `/etc/fstab`

```text-plain
CondorSubmit:home/tanyongsheng_net/data /home/tanyongsheng_net/data nfs rsize=8192,wsize=8192,timeo=14,intr
```

![](14_Part 2 - Install NFS File Syst.png)

*   Make sure your HTCondor's Executor have required permission to access the file in NFS filesystem

Try 

```text-plain
> sudo usermod -aG tanyongsheng_net condor
```

![](19_Part 2 - Install NFS File Syst.png)