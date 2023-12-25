# from strategy.fedadam import FedAdam
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    bytes_to_ndarray,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
import numpy as np
import torch
from arguments import initialise_lstm_arguments
from model import BaseLSTM
import matplotlib.pyplot as plt
import os

args = initialise_lstm_arguments()
#net = TempPointConv(config=args, F=87, D=293, no_flat_features=65)
net = BaseLSTM(config=args, F=101, D=1, no_flat_features=31)


def parameters_to_ndarrays(parameters: Parameters):
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


class FedAvg(Strategy):
    def __init__(self,algorithm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm=algorithm
        self.f1_history = []
        self.auroc_history=[]
        self.auprc_history=[]
        file_path = f'/home/zty/Mdata/result/mimic/{algorithm}/'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger = logging.getLogger(f'mimic_{algorithm}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 防止将日志消息传播到根日志记录器
        fh = logging.FileHandler(f'/home/zty/Mdata/result/mimic/{algorithm}/acc.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(message)s'))  # 设置简单的格式
        self.logger.addHandler(fh)

        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is not None:
            try:
                if server_round % 50 == 0:
                    print(f"Saving round {server_round} aggregated_parameters...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(parameters_aggregated)
                    # Convert `List[np.ndarray]` to PyTorch`state_dict`
                    params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                    net.load_state_dict(state_dict, strict=True)
                    # Save the model
                    torch.save(net.state_dict(), f"/home/zty/Mdata/result/mimic/{self.algorithm}/round_{server_round}.pt")
            except Exception as e:
                print(e)
                print('aggregate_fit ERROR!!!')

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)    
        print("------------------------------------------")
        print("evaluate aggregated loss: ", loss_aggregated)
        print("evaluate aggregated metrics: ", metrics_aggregated)
        print("------------------------------------------")

        log_message = (f"{server_round}: loss: {loss_aggregated} metrics: {metrics_aggregated}")

        self.logger.info(log_message)
        # print(metrics_aggregated)
        f1_score = metrics_aggregated.get('F1')  
        auroc = metrics_aggregated.get('AUROC')
        auprc = metrics_aggregated.get('AUPRC')
        self.f1_history.append(f1_score)
        self.auprc_history.append(auprc)
        self.auroc_history.append(auroc)
        plt.figure()
        plt.plot(self.f1_history, marker='o')
        plt.title("F1score Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("F1score Aggregated")
        # 保存图表为文件
        plt.savefig(f"/home/zty/Mdata/result/mimic/ditto/F1score.png")
        plt.close()

        plt.figure()
        plt.plot(self.auprc_history, marker='x')
        plt.title("AUPRC Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("AUPRC")
        plt.savefig(f"/home/zty/Mdata/result/mimic/ditto/AUPRC.png")
        plt.close()

        # 绘制和保存 AUROC 图表
        plt.figure()
        plt.plot(self.auroc_history, marker='^')
        plt.title("AUROC Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("AUROC")
        plt.savefig(f"/home/zty/Mdata/result/mimic/ditto/AUROC.png")
        plt.close()

        return loss_aggregated, metrics_aggregated
    

def weighted_metric_avg(eval_metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate custom metrics obtained from multiple clients, considering each metric value directly."""

    # Calculate total number of examples
    num_total_examples = sum(num_examples for num_examples, _ in eval_metrics)

    # Initialize a dictionary to store aggregated weighted metrics
    aggregated_metrics = {}

    # Iterate through each client's results in eval_metrics
    for num_examples, metrics in eval_metrics:
        # Iterate through each metric in the metrics dictionary
        for metric_name, metric_value in metrics.items():
            # Initialize metric in aggregated dictionary if not already present
            if metric_name not in aggregated_metrics:
                aggregated_metrics[metric_name] = 0.0
            # Add weighted metric
            aggregated_metrics[metric_name] += metric_value * num_examples

    # Compute average value for each metric
    for metric_name in aggregated_metrics:
        aggregated_metrics[metric_name] /= num_total_examples

    return aggregated_metrics
