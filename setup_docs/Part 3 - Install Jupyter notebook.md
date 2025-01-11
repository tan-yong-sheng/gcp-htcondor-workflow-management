# Part 3 - Install Jupyter notebook (On HTCondor Submit Node)
--------------------------------------------------------------

Step 1: Install Jupyter Notebook
--------------------------------

1\. On submit host, install required Python libraries, and start the Jupyter notebook:

```bash
> pip install jupyter
> python3 -m notebook
```

2\. Allow the Jupyter Notebook to be accessed remotely by changing the configuration of your Jupyter Notebook

```bash
> jupyter notebook --generate-config
```

3\. Then run the code below:

```bash
> echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py
```

Step 2: Share your Jupyter Notebook with Ngrok
----------------------------------------------

Open another SSH terminal to execute the following commands:

1\. Install ngrok via Apt with the following command:

```bash
> curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
	| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
	&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
	| sudo tee /etc/apt/sources.list.d/ngrok.list \
	&& sudo apt update \
	&& sudo apt install ngrok 
```

2\. Go to Ngrok dashboard to get your Ngrok's auth token: [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) 

![](/images/Part%203%20-%20Install%20Jupyter%20noteb.jpg)

3\. Authenticate your ngrok agent:

```bash
> ngrok authtoken <YOUR_AUTH_TOKEN>
```

4\. Share your jupyter notebook online via

```text-plain
> ngrok http http://localhost:8888
```

5\. Click “Endpoints” tab, and click the URL there to access:

![](/images/1_Part%203%20-%20Install%20Jupyter%20noteb.jpg)

Reference
---------

*   How to Share your Jupyter Notebook in 3 Lines of Code with Ngrok [https://towardsdatascience.com/how-to-share-your-jupyter-notebook-in-3-lines-of-code-with-ngrok-bfe1495a9c0c](https://towardsdatascience.com/how-to-share-your-jupyter-notebook-in-3-lines-of-code-with-ngrok-bfe1495a9c0c)
