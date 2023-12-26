import os
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple
from dataset import *
from model import Baseline
import flwr as fl
import numpy as np
import torch
import cifar


# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()

        # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]
        

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()

        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
        

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('--i', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    
    train_dataset = FedIsic2019(center=args.i, train=True, pooled=False)
    test_dataset = FedIsic2019(center=args.i, train=False, pooled=False) 
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
    )
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}
    print(num_examples)
    # Load model
    # model = cifar.Net().to(DEVICE).train()
    model = Baseline().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)


if __name__ == "__main__":
    main()
