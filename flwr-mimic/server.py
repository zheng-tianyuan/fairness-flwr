"""Flower server example."""
# from fedavg import FedAvg 
from fedrep import CustomStrategy
import flwr as fl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedAvg algorithm.")
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--num_rounds', type=int, required=True)
    args = parser.parse_args()
    strategy = CustomStrategy(
    algorithm=args.algorithm,
    fraction_fit=1.0,  # Sample 10% of available clients for the next round
    min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
    min_available_clients=10,  # Minimum number of clients that need to be connected to the server before a training round can start
)
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(args.num_rounds),
        strategy=strategy
    )
