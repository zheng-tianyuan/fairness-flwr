from flwr.server.strategy.fedavg import FedAvg
from typing import List, Tuple, Optional, Dict, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar
import matplotlib.pyplot as plt
import logging
import os

class FedAvg(FedAvg):

    def __init__(self,algorithm: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm
        self.accuracy_per_round = []
        file_path = f'/home/zty/Mdata/result/isic/{algorithm}/'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger = logging.getLogger(f'FedAvg_{algorithm}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 防止将日志消息传播到根日志记录器
        fh = logging.FileHandler(f'/home/zty/Mdata/result/isic/{algorithm}/acc.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(message)s'))  # 设置简单的格式
        self.logger.addHandler(fh)


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # 聚合损失
        total_loss = sum(evaluate_res.num_examples * evaluate_res.loss for _, evaluate_res in results)
        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        loss_aggregated = total_loss / total_examples if total_examples > 0 else None
        # 聚合准确率
        total_correct = sum(evaluate_res.num_examples * evaluate_res.metrics["accuracy"] for _, evaluate_res in results)
        accuracy_aggregated = total_correct / total_examples if total_examples > 0 else None
        print(f'loss:{loss_aggregated},accuracy:{accuracy_aggregated}')
        self.accuracy_per_round.append(accuracy_aggregated)
        # 绘制图表
        plt.figure()
        plt.plot(self.accuracy_per_round, marker='o')
        plt.title("Accuracy Aggregated over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy Aggregated")
        # 保存图表为文件
        plt.savefig(f"/home/zty/Mdata/result/isic/{self.algorithm}/acc.png")
        plt.close()

        log_message = (f"{server_round}: loss: {loss_aggregated} metrics: {accuracy_aggregated}")
        self.logger.info(log_message)
        return loss_aggregated, {"accuracy": accuracy_aggregated}
    
