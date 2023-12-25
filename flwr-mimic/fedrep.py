# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""
import numpy as np
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import logging  
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
    bytes_to_ndarray
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import os
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class FedRep(Strategy):
    """Configurable FedRep strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float ,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 10,
        min_eval_clients: int = 10,
        min_available_clients: int = 10,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn


    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep


    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_eval_clients), self.min_available_clients


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            print('aggregate_fit no results...')
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            print('aggregate_fit have failures...')
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        weights = [weights for weights, _ in weights_results]

        mean_w: NDArrays = [
        np.mean(np.asarray(layer), axis=0) for layer in zip(*weights)  # type: ignore
        ]

        parameters_aggregated = ndarrays_to_parameters(mean_w)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            print('aggregate_evaluate No results!!!')
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            print('aggregate_evaluate have failures!!!')
            return None, {}
        loss_results = [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]

        loss_aggregated = sum([loss for _, loss in loss_results]) / len(loss_results)

        # print('aggregate_evaluate loss_aggregated: ', loss_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
            # Weigh accuracy of each client by number of examples used

        return loss_aggregated, metrics_aggregated

def parameters_to_ndarrays(parameters: Parameters):
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


class CustomStrategy(FedRep):

    def __init__(self,algorithm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []
        self.f1_history = []
        self.auprc_history = []
        self.auroc_history = []
        file_path = f'/home/zty/Mdata/result/mimic/{algorithm}/'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger = logging.getLogger(f'Mimic_{algorithm}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 防止将日志消息传播到根日志记录器
        fh = logging.FileHandler(f'/home/zty/Mdata/result/mimic/{algorithm}/f1.log')
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
        f1_score = metrics_aggregated.get('f1')  
        auroc = metrics_aggregated.get('auroc')
        auprc = metrics_aggregated.get('auprc')
        self.f1_history.append(f1_score)
        self.auprc_history.append(auprc)
        self.auroc_history.append(auroc)
        plt.figure()
        plt.plot(self.f1_history, marker='o')
        plt.title("F1score Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("F1score Aggregated")
        # 保存图表为文件
        plt.savefig(f"/home/zty/Mdata/result/mimic/fedrep/F1score.png")
        plt.close()

        plt.figure()
        plt.plot(self.auprc_history, marker='x')
        plt.title("AUPRC Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("AUPRC")
        plt.savefig(f"/home/zty/Mdata/result/mimic/fedrep/AUPRC.png")
        plt.close()

        # 绘制和保存 AUROC 图表
        plt.figure()
        plt.plot(self.auroc_history, marker='^')
        plt.title("AUROC Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("AUROC")
        plt.savefig(f"/home/zty/Mdata/result/mimic/fedrep/AUROC.png")
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