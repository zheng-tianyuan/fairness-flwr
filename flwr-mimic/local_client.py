from reader import MIMICReader
import flwr as fl
from model import BaseLSTM
from collections import OrderedDict
import torch
import numpy as np
from torch.optim import Adam
import os
from metrics import print_metrics_regression,print_metrics_mortality
from typing import Dict, Optional
from arguments import initialise_lstm_arguments,best_lstm
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import traceback


class LocalClient(fl.client.NumPyClient):
    def __init__(self,args,device,train_datareader,val_datareader):
        self.model = BaseLSTM(config=args, F=101, D=1, no_flat_features=31)
        self.device = device
        self.cid = args.i
        self.local_ep = 1
        self.batch_size = 32
        self.learning_rate = 0.00168
        self.L2_regularisation = 0
        self.loss = 'msle'
        self.task = 'mortality'
        self.mode = 'train'
        self.train_datareader = train_datareader
        self.val_datareader = val_datareader
        self.no_train_batches = len(self.train_datareader.patients) / self.batch_size


    def get_parameters(self,config: Optional[Dict[str, str]] = None):
        # print("Called Get_Parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)


    def fit(self, parameters, config=None):

        # Set model parameters/weights
        self.set_parameters(parameters)
        try:
            # Train model
            train_loss, num_samples = train(self.model, self.train_datareader, self.batch_size, self.learning_rate, self.local_ep, 
                                    self.loss, self.L2_regularisation, self.device, self.task)
            # print(f"[CLIENT {self.cid}] Training Loss: {train_loss:8.4f}" + "\r")
        
        except Exception as e:
            print('Some Errors Here!!!')
            print(e)
            traceback.print_exc()

        return self.get_parameters(), num_samples, {}


    def evaluate(self, parameters, config=None):
        
        #print(f"evaluate() on client cid={self.cid}")
        # Set model parameters/weights
        self.set_parameters(parameters)
        
        # Validation model
        eval_loss, metric, num_sample = val(self.model, self.val_datareader, self.batch_size, self.loss, self.device, self.task)
        # print(eval_loss, metric, num_sample)
        
        # print(f"[CLIENT {self.cid}] Evaluation Loss: {eval_loss:8.4f} | Metric: {metric}" + "\r")

        return float(eval_loss), num_sample, metric
    
def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels
        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.to(device)
    mask = mask.to(device)
    # 用 NaN 替换填充部分并展平
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y


def train(model, train_datareader, batch_size, learning_rate, local_ep, 
          loss_type, L2_regularisation, device, task, mort_pred_time=24):
    # bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
    train_loss = []
    train_y_hat_los = np.array([])
    train_y_los = np.array([])
    train_y_hat_mort = np.array([])
    train_y_mort = np.array([])
    num_sample = 0

    optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    model.to(device)
    model.train()

    for _ in range(local_ep):
        trainloader = train_datareader.batch_gen(batch_size=batch_size)

        for batch_idx, batch in enumerate(trainloader):
            # print('batch_idx:', batch_idx)
            # if batch_idx > (self.no_train_batches[idx] // (100 / self.config.percentage_data)):
            #     print('break')
            #     break
            # unpack batch
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None
            num_sample += padded.shape[0]

            ##########################
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            ##########################
            optimiser.zero_grad()
            # y_hat = model(padded, diagnoses, flat)
            # loss = model.classification_loss(y_hat, los_labels)
            y_hat_los, y_hat_mort = model(padded, diagnoses, flat)
            loss = model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, device, loss_type)

            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())

    avg_train_loss = sum(train_loss) / len(train_loss)
            
    return avg_train_loss, num_sample


def val(model, val_datareader, batch_size, loss_type, device, task, 
        mort_pred_time=48, mode='train'):
        
    if mode == 'train':
        
        val_y_hat_los = np.array([])
        val_y_los = np.array([])
        val_y_hat_mort = np.array([])
        val_y_mort = np.array([])
        bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
        
        num_sample = 0
        val_loss = []

        model.to(device)
        model.eval()
        val_batches = val_datareader.batch_gen(batch_size=batch_size)
        
        # with torch.no_grad():
        for batch in val_batches:
            padded, mask,  flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None
            num_sample += padded.shape[0]
            ##########################
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            ##########################
            y_hat_los, y_hat_mort = model(padded,diagnoses,flat)
            loss = model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, device, loss_type)
            val_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak
            # val_loss += loss.item()
            if task in ('LoS', 'multitask'):
                val_y_hat_los = np.append(val_y_hat_los, remove_padding(y_hat_los, mask.type(bool_type), device))
                val_y_los = np.append(val_y_los, remove_padding(los_labels, mask.type(bool_type), device))

            if task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                val_y_hat_mort = np.append(val_y_hat_mort, remove_padding(y_hat_mort[:, mort_pred_time],
                                                                    mask.type(bool_type)[:, mort_pred_time], device))
                val_y_mort = np.append(val_y_mort, remove_padding(mort_labels[:, mort_pred_time],
                                                                    mask.type(bool_type)[:, mort_pred_time], device))

        metrics_list = []
        if task in ('LoS'):
            los_metrics_list = print_metrics_regression(val_y_los, val_y_hat_los) # order: mad, mse, mape, msle, r2, kappa
            metrics_list = los_metrics_list
        
        if task in ('mortality'):
            mort_metrics_list = print_metrics_mortality(val_y_mort, val_y_hat_mort)
            metrics_list = mort_metrics_list

        if task in ('multitask'):
            los_metrics_list = print_metrics_regression(val_y_los, val_y_hat_los)
            metrics_list.append(los_metrics_list)
            mort_metrics_list = print_metrics_mortality(val_y_mort, val_y_hat_mort)
            metrics_list.append(mort_metrics_list)

        avg_val_loss = sum(val_loss) / len(val_loss)

    return avg_val_loss, metrics_list, num_sample

    
def main() -> None:
    """Load data, start CifarClient."""

    args=initialise_lstm_arguments()
    args['exp_name'] = 'StandardLSTM'
    args['dataset'] = 'MIMIC'
    args['task'] = 'mortality'
    args = best_lstm(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.i)
    
    datareader = MIMICReader
    data_path = '/home/zty/Mdata/fair-10/'
    train_datareader = datareader(data_path + "hosp" + str(args.i) + '/train', device=DEVICE) 
    val_datareader = datareader(data_path + "hosp" + str(args.i) + '/val', device=DEVICE)
   
    client = LocalClient(args, DEVICE, train_datareader, val_datareader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)



if __name__ == "__main__":
    main()