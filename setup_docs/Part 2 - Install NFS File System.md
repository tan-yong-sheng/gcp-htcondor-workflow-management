# Part 2 - Install NFS File System
-----------------------------------

> Note:  I am thinking to change the nfs directory from /home/tanyongsheng_net/data to /mnt/nfs , because permission issue mentioned at Step 6

## Step 1: Configure NFS Server (in HTCondor Submit Node)

1\. Install NFS server packages:

```bash
> sudo apt update
> sudo apt install nfs-kernel-server
```

![](/images/4_Part%202%20-%20Install%20NFS%20File%20Syst.jpg)

2\. Create the shared directory:

```bash
> sudo mkdir -p /mnt/nfs
```

3\. Configure NFS exports:

```bash
> sudo nano /etc/exports
```

*   Add this line to `/etc/exports` file:

```text-plain
/mnt/nfs *(rw,sync,no_subtree_check,no_root_squash)
```
![image](https://github.com/user-attachments/assets/aa0f9e10-f1c6-4199-9eb4-042964842a21)

4\. Apply the NFS configuration:

```bash
> sudo exportfs -a
```

![image](https://github.com/user-attachments/assets/ff0f185b-3c5c-4419-9353-e3919e73b5fb)


5\. Restart NFS service:

```bash
> sudo systemctl restart nfs-kernel-server
> sudo systemctl status nfs-kernel-server
```

![image](https://github.com/user-attachments/assets/b243762c-74fd-4a57-8e79-67d763ea1698)

6\. Set up permissions:

```bash
> sudo chmod -R 777 /mnt
```

*   Test if this works for `Condor` user or `nobody` user to create and write the file in NFS directory

```bash
> sudo -u condor touch /mnt/nfs/testfile.txt
> sudo -u nobody touch /mnt/nfs/testfile2.txt
```

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
> sudo mkdir -p /mnt/nfs
```

![](/images/10_Part%202%20-%20Install%20NFS%20File%20Syst.png)


4\. Set up permissions:

```bash
> sudo mkdir /home/nobody
> sudo chown nobody:nogroup /home/nobody/
> sudo chmod 777 /home/nobody/

> sudo chmod -R 777 /mnt
```

*   Test if this works for `Condor` group to create and write the file in NFS directory

```bash
> sudo -u condor touch /mnt/nfs/testfile.txt
> sudo -u nobody touch /mnt/nfs/testfile2.txt
```


5. Mount the NFS file system on Condor Executors

```bash
> sudo mount CondorSubmit:mnt/nfs /mnt/nfs
```

## Step 2: Auto-mounting for HTCondor Executor

An alternate way to mount an NFS share in HTCondor Executor is to add a line to the `/etc/fstab` file. The line must state the hostname of the NFS server, the directory on the server being exported, and the directory on the local machine where the NFS share is to be mounted.

The general syntax for the line in `/etc/fstab` file is as follows:

*   File location: `/etc/fstab`

```text-plain
CondorSubmit:mnt/nfs /mnt/nfs nfs rsize=8192,wsize=8192,timeo=14,intr
```

![](/images/14_Part%202%20-%20Install%20NFS%20File%20Syst.png)

