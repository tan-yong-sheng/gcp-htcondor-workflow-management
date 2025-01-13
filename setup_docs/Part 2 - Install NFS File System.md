# Part 2 - Install NFS File System
-----------------------------------

## Step 1: Configure NFS Server (in HTCondor Submit Node)

1\. Install NFS server packages:

```bash
> sudo apt update
> sudo apt install nfs-kernel-server
```

![](/images/4_Part%202%20-%20Install%20NFS%20File%20Syst.jpg)

2\. Create the shared directory:

```bash
> sudo mkdir -p /home/tanyongsheng_net/data
```

3\. Configure NFS exports:

```bash
> sudo nano /etc/exports
```

*   Add this line to `/etc/exports` file:

```text-plain
/home/tanyongsheng_net/data *(rw,sync,no_subtree_check,no_root_squash)
```

![](/images/15_Part%202%20-%20Install%20NFS%20File%20Syst.png)

4\. Apply the NFS configuration:

```bash
> sudo exportfs -a
```

![](/images/1_Part%202%20-%20Install%20NFS%20File%20Syst.jpg)

5\. Restart NFS service:

```bash
> sudo systemctl restart nfs-kernel-server
> sudo systemctl status nfs-kernel-server
```

6\. Set up permissions:

```bash
> sudo mkdir /home/nobody
> sudo chown nobody:nogroup /home/nobody/
> sudo chmod 777 /home/nobody/

> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net
> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net/data

> sudo usermod -aG tanyongsheng_net condor

> sudo chmod 777 /home/tanyongsheng_net
> sudo chmod 777 /home/tanyongsheng_net/data
```

*   Test if this works for `Condor` group to create and write the file

```bash
> sudo -u condor touch /home/tanyongsheng_net/data/testfile.txt
```

![](/images/17_Part%202%20-%20Install%20NFS%20File%20Syst.png)

### Step 2: Configure NFS Common (in HTCondor Executor Nodes)

1\. Install NFS common packages:

```bash
> sudo apt update
> sudo apt install nfs-common
```

2\. Edit hosts file:

```bash
> sudo nano /etc/hosts
```

![](/images/16_Part%202%20-%20Install%20NFS%20File%20Syst.png)

3. Create folder directory:
 
```bash
> sudo mkdir -p /home/tanyongsheng_net/data
```

![](/images/10_Part%202%20-%20Install%20NFS%20File%20Syst.png)


4\. Set up permissions:

```bash
> sudo mkdir /home/nobody
> sudo chown nobody:nogroup /home/nobody/
> sudo chmod 777 /home/nobody/

> sudo usermod -aG tanyongsheng_net condor

> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net
> sudo chown tanyongsheng_net:condor /home/tanyongsheng_net/data
> sudo chmod 777 /home/tanyongsheng_net
> sudo chmod 777 /home/tanyongsheng_net/data
```

5. Mount the NFS file system on Condor Executors

```bash
> sudo mount CondorSubmit:home/tanyongsheng_net/data /home/tanyongsheng_net/data/
```

![](/images/13_Part%202%20-%20Install%20NFS%20File%20Syst.png)

## Step 2: Auto-mounting for HTCondor Executor

An alternate way to mount an NFS share in HTCondor Executor is to add a line to the `/etc/fstab` file. The line must state the hostname of the NFS server, the directory on the server being exported, and the directory on the local machine where the NFS share is to be mounted.

The general syntax for the line in `/etc/fstab` file is as follows:

*   File location: `/etc/fstab`

```text-plain
CondorSubmit:home/tanyongsheng_net/data /home/tanyongsheng_net/data nfs rsize=8192,wsize=8192,timeo=14,intr
```

![](/images/14_Part%202%20-%20Install%20NFS%20File%20Syst.png)

