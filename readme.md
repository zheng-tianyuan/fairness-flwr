# Project File Descriptions

`dataset` 
```shell
-- mimic-iv https://physionet.org/content/mimiciv/2.2/ , set up by following https://github.com/EmmaRocheteau/TPC-LoS-prediction
-- isic https://owkin.github.io/FLamby/, which can be used directly
```

The project includes the following basic files:

- `client.py`: Client-side file, responsible for handling local model training and updates. It processes the data locally and sends updates to the server.

- `server.py`: Listens for client connection requests and updates the global model. This server-side script manages the overall coordination of the distributed training process.

- `strategy`: A directory containing aggregation strategies, such as FedAvg and FedRep. These scripts define how the updates from clients are aggregated to update the global model.

- `shell.sh`: Script for project startup and client resource allocation. It includes commands for initiating the training process and distributing resources among clients.

This project includes four distinct algorithms:
- `FedAvg`
- `FedRep`
- `Ditto`
- `Local`
- `FedBN`
- `FedAdam`


### Launch the project

Launch:
- `chmod 777 shell.sh` 
- `./shell.sh`
- `example`:chmod 777 ditto.sh , ./ditto.sh

### Build your own project

- `client.py`: You can transfer your centralized algoritm to client directly. It will contain the parts of model,params initialization,optimization,train,test,val. You can put them into client.py or import them from other files, but the file we  execute will be only client.py. For some personalized Fl methods(ditto,fedrep), we will build a local model for each client. In this case we need to update the centralized algoritm to satisfied it.

- `server.py`: For some another type personalized Fl methods(FedAdam), we don't need to build the build a local model for each client. So we can import them directly:

```python
from flwr.server.strategy import FedAdam
import flwr as fl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedAvg algorithm.")
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--num_rounds', type=int, required=True)
    args = parser.parse_args()
    strategy = FedAdam(
        fraction_fit=1.0,  # Sample 10% of available clients for the next round
        min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
        min_available_clients=10,  # Minimum number of clients that need to be connected to the server before a training round can start
    )
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(args.num_rounds),
        strategy=strategy
    )


- `strategy`: You can inherit from the existing strategy and override some functions to adapt to your algorithm.

- `shell.sh`: You can add any parameters you want to the client and server in this file, and control the allocation of server resources.