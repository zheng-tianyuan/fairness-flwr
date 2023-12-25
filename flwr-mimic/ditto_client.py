import copy
import os
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from flwr.common.typing import Scalar
from ditto_utils import train, val
from arguments import initialise_lstm_arguments
from model import BaseLSTM
from reader import MIMICReader
import traceback

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, pmodel, train_datareader, val_datareader, device, loss, 
                 L2_regularisation, task, local_lr, lam, model_save_path, best_model_path):
        self.cid = cid
        self.model = model
        self.pmodel = pmodel
        self.device = device
        self.loss = loss
        self.L2_regularisation = L2_regularisation
        self.task = task
        self.local_lr = local_lr
        self.lam = lam
        self.model_save_path = model_save_path
        self.best_model_path = best_model_path

        self.train_datareader = train_datareader
        self.val_datareader = val_datareader
        # self.no_train_batches = len(self.train_datareader.patients) / self.batch_size
        # self.bool_type = torch.cuda.BoolTensor if self.device == torch.device('cuda') else torch.BoolTensor

    def get_parameters(self, config):
        # print("Called Get_Parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
        # print("Called set_parameters")

    def fit(self, parameters, config):
        
        print(f"fit() on client cid={self.cid}")
        # Set model parameters/weights for global model
        self.set_parameters(parameters)

        # load previous locally updated model from saved files
        model_name = f"model_client_{self.cid}.pt"
        model_path = self.model_save_path + model_name
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.pmodel.load_state_dict(state_dict)
        else:
            self.pmodel = self.model

        try:
            # Train model
            train_loss, num_samples = train(self.cid, self.model, self.pmodel, self.local_lr, self.lam, self.train_datareader, 
                                            config["batch_size"], config["learning_rate"], config["epochs"], 
                                            self.loss, self.L2_regularisation, self.device, self.task)
            # print(f"[CLIENT {self.cid}] Training Loss: {train_loss:8.4f}" + "\r")
            # save local updated model to files to save current params of head layers for future use
            torch.save(self.pmodel.state_dict(), model_path)

        except Exception as e:
            print('Some Errors fit()!!!')
            print(e)

        return self.get_parameters({}), num_samples, {}

    def evaluate(self, parameters, config):
        
        print(f"evaluate() on client cid={self.cid}")
        # Set model parameters/weights
        self.set_parameters(parameters)
        # load previous locally updated model from saved files
        model_name = f"model_client_{self.cid}.pt"
        model_path = self.model_save_path + model_name
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.pmodel.load_state_dict(state_dict)
        else:
            self.pmodel = self.model
        try:
            # Validation model
            eval_loss, metric, num_sample = val(model=self.pmodel, val_datareader=self.val_datareader, batch_size=8, 
                                                loss_type=self.loss, device=self.device, task=self.task)
            # print(f"[CLIENT {self.cid}] Evaluation Loss: {eval_loss:8.4f}")

            print(f'cid={self.cid} saving best model...')
            tmp_file_path = self.model_save_path + f"auprc_client{self.cid}.txt"
            auprc = metric.get('auprc')
            # print(auprc)
            
            # 首先检查文件是否存在
            if os.path.exists(tmp_file_path):
                # 如果文件存在，读取内容
                with open(tmp_file_path, "r") as r:
                    content = r.read().strip()  # 去除空格和换行符

                # 尝试将内容转换为浮点数，如果失败则设置为0.0
                try:
                    previous_auprc = float(content) if content else 0.0
                except ValueError:
                    previous_auprc = 0.0
            else:
                # 如果文件不存在，则之前的auprc默认为0.0
                previous_auprc = 0.0

            # 比较当前的auprc与之前保存的auprc，如果更高，则保存当前模型
            if auprc > previous_auprc:
                torch.save(self.pmodel.state_dict(), self.best_model_path + f"best_client_{self.cid}.pt")
                with open(tmp_file_path, "w") as f:
                    f.write("{}".format(auprc))

        except Exception as e:
            print('---> evaluate() having Errors!!!')
            traceback.print_exc()
            print(e)

        return eval_loss, num_sample, {
    "Accuracy": float(metric['acc']),
    "AUROC": float(metric['auroc']),
    "AUPRC": float(metric['auprc']),
    # 如果需要 F1 分数
    "F1": float(metric['f1'])
}


def main() -> None:
    """Load data, start Client."""
    args = initialise_lstm_arguments()

    torch.manual_seed(args.random_seed) # pytorch random seed
    np.random.seed(args.random_seed) # numpy random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cid = args.cid
    print('args.cid: ', cid)
    args.loss = 'bce'
    args.L2_regularisation = 0
    args.task = 'mortality'

    SAVE_PATH = f"{args.tmp_save}/"
    BEST_MODEL_PATH = f"{args.tmp_save}/best_model/"

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)

    try:
        os.remove(SAVE_PATH + f"auprc_client{cid}.txt")
    except FileNotFoundError:
        pass

    # Load data
    datareader = MIMICReader
    data_path = '/home/zty/Mdata/fair-10/'
    train_datareader = datareader(data_path + "hosp" + str(cid) + '/train', device=device) 
    val_datareader = datareader(data_path + "hosp" + str(cid) + '/val', device=device)

    # initialize the model
    model = BaseLSTM(config=args, F=101, D=1, no_flat_features=31)
    pmodel = copy.deepcopy(model)

    # Start client
    client = FlowerClient(cid, model, pmodel, train_datareader, val_datareader, device, args.loss, 
                          args.L2_regularisation, args.task, args.local_lr, args.lam, SAVE_PATH, BEST_MODEL_PATH)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)


if __name__ == "__main__":
    main()