# Intent Classification Service


## Introduction 


## Setup

- create `.env` file inside the directory and place your `OPENAI_API_KEY`  

### 1. Using Docker 

For AMD64 architecture (Intel):  

- `chmod +x [run_container.sh](./run_container.sh), to make the script executable`   
- `./run_container.sh`  to build and run the app 

For ARM architecture (M chips):  



Wait a couple of minutes for the container to build and the classifier service will start automatically with the default settings.   

- ADD SCREENSHOT  

### 2. Using python 3.11 and a virtual enviroment   

- create a virtual environment (e.g. `python3.11 -m venv venv`)  
- activate it (e.g. `source venv/bin/activate`)  
- install requirements (e.g. `pip install -r requirements.txt`)  
- run the app (e.g. `python server.py`) with default parameters  
- or use the [entrypoint.sh](./entrypoint.sh) file to alter configuration parameters  
    -  `chmod +x [entrypoint.sh](./entrypoint.sh), to make the script executable`   
    -  `./entrypoint.sh`  run the app with specified parameters  


Wait a few seconds for the model to load and once the server is up you can use [make_requests notebook](./notebooks/make_requests.ipynb) to make your requests and test the API. 

You'll soon see the model evaluation metrics in your console. For detailed explanations on these metrics, refer to [evaluation_metrics_notebook](./notebooks/evaluation_metrics.ipynb). Explore [gpt_model_loader_notebook](./notebooks/gpt_model_loader.ipynb) for  examples of few-shot and zero-shot loading outputs.  





