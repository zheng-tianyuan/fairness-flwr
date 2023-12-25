from typing import Dict
import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar
from fedavg import FedAvg
from arguments import initialise_lstm_arguments
from model import BaseLSTM

def get_parameters(model):
    return [values.cpu().numpy() for _, values in model.state_dict().items()]

def main():

    args = initialise_lstm_arguments()

    torch.manual_seed(args.random_seed) # pytorch random seed
    np.random.seed(args.random_seed) # numpy random seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,  # number of local epochs
        }
        return config
    
    # initialize the model
    model = BaseLSTM(config=args, F=101, D=1, no_flat_features=31)
    # print('============================')
    # print(model)

    # configure the strategy
    strategy = FedAvg(
        fraction_fit=args.frac,
        fraction_evaluate=args.frac,
        min_fit_clients=args.num_users,
        min_eval_clients=args.num_users,
        min_available_clients=args.num_users,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model.to(device))),
    )
    print('We are here!!!')


    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
