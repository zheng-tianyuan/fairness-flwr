import copy
import os
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from arguments import initialise_lstm_arguments
from model import FedrepBaseLSTM
from reader import MIMICReader
import traceback

import torch

def get_params(model):
    return [values.cpu().numpy() for _, values in model.state_dict().items()]

class LocalModel(torch.nn.Module):
    def __init__(self, base, head):

        super(LocalModel, self).__init__()
        self.base = base
        self.head = head
        self.sigmoid = torch.nn.Sigmoid()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bce_loss = torch.nn.BCELoss().to(DEVICE)
        
    def forward(self, X, flat, time_before_pred=5):
        B, _, T = X.shape
        out = self.base(X, flat)
        out = self.head(out)
        out = self.sigmoid(out.view(B, T - time_before_pred))

        return out
    
    def classification_loss(self, y_hat, y_los):

        loss = self.bce_loss(y_hat, y_los) * 100
        return loss

class FedRepClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_datareader,val_datareader, device, local_ep, 
                 batch_size,learning_rate, tmp_save_path, best_model_path):
        self.cid = cid
        self.model = model
        self.device = device
        self.local_ep = local_ep
        self.learning_rate = learning_rate
        self.tmp_save_path = tmp_save_path
        self.best_model_path = best_model_path
        self.batch_size = batch_size
        self.train_datareader = train_datareader
        self.val_datareader = val_datareader
        self.best_accuracy = 0.0
       
    def get_parameters(self, config):
        return get_params(self.model)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.base.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.base.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        
        print(f"fit() on client cid={self.cid}")
        # load previous locally updated model from saved files
        model_name = f"model_client_{self.cid}.pt"
        model_path = self.tmp_save_path + model_name
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            # print('========> Training load self.model from file: ', self.model.state_dict())
        else:
            pass
        # Set model parameters/weights
        self.set_parameters(parameters)
        try:
            # Train model
            train_loss, num_samples = train(self.model, self.train_datareader, self.batch_size, self.learning_rate, 
                                            self.local_ep, self.device)
    
            print(f"[CLIENT {self.cid}] Training Loss: {train_loss:8.4f}" + "\r")
            # save local updated model to files to save current params of head layers for future use
            torch.save(self.model.state_dict(), model_path)

        except Exception as e:
            print('Some Errors Here!!!')
            print(e)
            traceback.print_exc()

        return get_params(self.model), num_samples, {}


    def evaluate(self, parameters, config):
        
        print(f"evaluate() on client cid={self.cid}")
        model_name = f"model_client_{self.cid}.pt"
        model_path = self.tmp_save_path + model_name
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        else:
            pass        
        # Set model parameters/weights
        self.set_parameters(parameters)
        
        # Validation model
        eval_loss, metric, num_sample = val(self.cid, self.model, self.val_datareader, self.batch_size, self.device)
        print(f"[CLIENT {self.cid}] Evaluation Loss: {eval_loss:8.4f}" + "\r")
        print(metric)
        # print(f"[CLIENT {self.cid}] Accuracy: {metric[0]} AUROC: {metric[1]} AUPRC: {metric[2]}")

        # save best model based on accuracy
        print('saving best model...')
        if metric["acc"] > self.best_accuracy:
            self.best_accuracy = metric["acc"]
            torch.save(self.model.state_dict(), self.best_model_path + f"best_client_{self.cid}.pt")
            print('Best model saved.')
        else:
            print('Not better than before.')

        return float(eval_loss), num_sample, {'acc': metric['acc'], 'auroc': metric['auroc'], 'auprc': metric['auprc'], 'f1': metric['f1']}


import flwr as fl
from flwr.common.typing import Scalar
import torch
from torch import nn, Tensor
import numpy as np
from collections import OrderedDict
import copy
import os

def main() -> None:
    # parse input arguments
    args = initialise_lstm_arguments()
    cid = args.i
    args.L2_regularisation = 0
    args.learning_rate = 0.0072
    args.local_ep = 1
    args.batch_size = 64
    args.num_rounds = 30
    args.frac = 0.4
    args.model = 'fedrep-lstm'
    args.mode = 'train'
    args.tmp_save = 'exp_1'
    args.random_seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed) # pytorch random seed
    np.random.seed(args.random_seed) # numpy random seed

    SAVE_PATH = f"/home/zty/flwr-mimic/tmp/{args.tmp_save}/"
    BEST_MODEL_PATH = f"/home/zty/flwr-mimic/tmp/{args.tmp_save}/"

    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # remove existing saved model in tmp_model
    try:
        os.remove(SAVE_PATH + f"auprc_client{cid}.txt")
    except FileNotFoundError:
        pass

    # initialize the model
    datareader = MIMICReader
    data_path = '/home/zty/Mdata/fair-10/'
    train_datareader = datareader(data_path + "hosp" + str(cid) + '/train', device=device) 
    val_datareader = datareader(data_path + "hosp" + str(cid) + '/val', device=device)

    # initialize the model
    model = FedrepBaseLSTM(config=args, F=101, D=1, no_flat_features=31)

    # initialize model & heads
    head = copy.deepcopy(model.fc)
    # print('head:', head)
    model.fc = nn.Identity()
    model = LocalModel(model, head)
    # print('model:', model)

    client = FedRepClient(cid, model, train_datareader, val_datareader, device,args.local_ep,args.batch_size,args.learning_rate, SAVE_PATH, BEST_MODEL_PATH)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)
       

from metrics import print_metrics_mortality

BASE_EP = 1

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def remove_padding(y, mask, device):

    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y

def train(model, train_datareader, batch_size, learning_rate, local_ep, device, mort_pred_time=24):

    # bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
    base_epoch_loss = []
    head_epoch_loss = []
    num_sample = 0

    base_vars = [i for i in model.base.parameters()]
    head_vars = [v for v in model.head.parameters()]

    poptimizer = torch.optim.SGD(head_vars, lr=learning_rate)
    optimiser = torch.optim.SGD(base_vars, lr=learning_rate)

    model.to(device)
    model.train()

    print('---> Updating personalized heads for selected clients...')

    for step in range(local_ep):
        trainloader = train_datareader.batch_gen(batch_size=batch_size)

        for batch_idx, batch in enumerate(trainloader):

            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            num_sample += padded.shape[0]
            ##########################
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            ##########################
            poptimizer.zero_grad()
            y_hat_head = model(padded, flat)

            head_loss = model.classification_loss(y_hat_head, mort_labels)
            head_loss.backward()
            poptimizer.step()
            head_epoch_loss.append(head_loss.item())

    print('---> Updating global representation...')

    num_sample = 0

    for _ in range(BASE_EP):
        trainloader = train_datareader.batch_gen(batch_size=batch_size)

        for batch_idx, batch in enumerate(trainloader):
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            num_sample += padded.shape[0]
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            optimiser.zero_grad()
            y_hat = model(padded, flat)
            loss = model.classification_loss(y_hat, mort_labels)
            loss.backward()
            optimiser.step()
            base_epoch_loss.append(loss.item())

    avg_train_loss = sum(head_epoch_loss) / len(head_epoch_loss)

    return avg_train_loss, num_sample


def val(client_idx, model, val_datareader, batch_size, device, mort_pred_time=24, mode='train'):
        
    if mode == 'train':
        val_sens = np.array([])
        val_y_hat = np.array([])
        val_y = np.array([])
        bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
        num_sample = 0
        val_loss = []

        model.to(device)
        model.eval()
        val_batches = val_datareader.batch_gen(batch_size=batch_size)
        
        with torch.no_grad():
            for batch in val_batches:
                padded, mask,  flat, los_labels, mort_labels, seq_lengths = batch
                num_sample += padded.shape[0]
                ##########################
                if padded.shape[0] <= 1:
                    print('shape <= 1')
                    break
                ##########################
                y_hat = model(padded, flat)
                loss = model.classification_loss(y_hat, mort_labels)
                val_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

                val_y_hat = np.append(val_y_hat, remove_padding(y_hat[:, mort_pred_time],
                                                                mask.type(bool_type)[:, mort_pred_time], device))
                val_y = np.append(val_y, remove_padding(mort_labels[:, mort_pred_time],
                                                                    mask.type(bool_type)[:, mort_pred_time], device))
        
        metrics_list = []
        metrics_list = print_metrics_mortality(val_y, val_y_hat)

        avg_val_loss = sum(val_loss) / len(val_loss)

    return avg_val_loss, metrics_list, num_sample

if __name__ == "__main__":
    main()

