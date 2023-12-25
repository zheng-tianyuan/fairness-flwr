import copy
import os
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
import argparse
from dataset import *
from model import Baseline
import traceback
import cifar
from typing import Dict, List, Tuple
from torch import nn
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn

class LocalHead(nn.Module):
    def __init__(self, device, input_features, num_classes):
        super(LocalHead, self).__init__()
        self.device = device
        self.fc = nn.Linear(input_features, num_classes).to(self.device)

    def forward(self, features):
        # 将输入张量移动到指定设备上
        features = features.to(self.device)
        return self.fc(features)


class FedrepClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.model_save_path = 'tmp/'
        self.best_model_path = 'tmp/best'
        self.tmp_acc=0.0
        
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[List[np.ndarray], int, Dict]:
            
            model_name = f"model_client_{self.cid}.pt"
            model_path = self.model_save_path + model_name
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path)
                self.model.load_state_dict(state_dict)
            else:
                pass
    
            # Set model parameters, train model, return updated model parameters
            self.set_parameters(parameters)
            cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
            torch.save(self.model.state_dict(), model_path)

            return self.get_parameters(config={}), self.num_examples["trainset"], {}
    
    def evaluate(self, parameters, config):
        
        # load previous locally updated model from saved files
        model_name = f"model_client_{self.cid}.pt"
        model_path = self.model_save_path + model_name
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        else:
            pass
        # Set model parameters/weights
        self.set_parameters(parameters)

        try:
            loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)

            if accuracy > self.tmp_acc:
                torch.save(self.model.state_dict(), self.best_model_path + f"best_client_{self.cid}.pt")
                self.tmp_acc=accuracy

        except Exception as e:
            print('---> evaluate() having Errors!!!')
            traceback.print_exc()
            print(e)

        print(self.cid,self.num_examples["testset"], {"accuracy": float(accuracy)})
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
        batch_size=32,
        shuffle=True,
        num_workers=1,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
    )
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    # 创建 FedRepBaseline 模型
    model = Baseline()
    model.to(DEVICE)

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))
        
    # Start client
    client = FedrepClient(args.i,model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)


if __name__ == "__main__":
    main()

