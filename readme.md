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



### Launch the project

This project includes four distinct algorithms:
- `FedAvg`
- `FedRep`
- `Ditto`
- `Local`

Launch:
chmod 777 shell.sh
./shell.sh

