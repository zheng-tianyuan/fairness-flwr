import os
import copy
import numpy as np
import torch
from torch.optim import Adam
from metrics import print_metrics_regression, print_metrics_mortality
from fedoptimizer import PerturbedGradientDescent


def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels
        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y


def train(cid, model, pmodel, local_lr, lam, train_datareader, batch_size, learning_rate, local_ep, loss_type, 
            L2_regularisation, device, task, mort_pred_time=24):

    # bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
    train_loss = []
    num_sample = 0
    local_loss = []
    print(f'Client {cid} updating local personalized model...')
    # update local personalized models
    pmodel.to(device)
    pmodel.train()

    for _ in range(local_ep):
        poptimizer = PerturbedGradientDescent(pmodel.parameters(), lr=local_lr, lam=lam)
        # poptimizer.defaults = dict(lr=local_lr, lam=lam)

        trainloader = train_datareader.batch_gen(batch_size=batch_size)

        for batch_idx, batch in enumerate(trainloader):
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None

            num_sample += padded.shape[0]
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            poptimizer.zero_grad()
            for param in pmodel.parameters():  # Set requires_grad=True for all parameters in the model
                param.requires_grad = True
                # if param.grad is None:
                #     print("WARNING: Personalized Parameter", param.name, "has no gradient")
                #     param.grad = torch.zeros_like(param) # set the gradient to a tensor of zeros ensures that the parameter is not updated during the optimizer step and does not affect the training process
            y_hat_los, y_hat_mort = pmodel(padded, diagnoses, flat)
            ploss = pmodel.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, device, loss_type)
            ploss.backward()
            poptimizer.step(pmodel.parameters(), device)
            local_loss.append(ploss.item())

    print(f'Client {cid} updating global model...')
    print('-----------------------------------------------------')
    model.to(device)
    model.train()
    optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    for _ in range(local_ep):
        trainloader = train_datareader.batch_gen(batch_size=batch_size)
        # print('Local Epoch: {}'.format(iter))
        # running_loss = 0.0
        for batch_idx, batch in enumerate(trainloader):
            # print('batch_idx:', batch_idx)
            # if batch_idx > (self.no_train_batches[idx] // (100 / self.config.percentage_data)):
            #     print('break')
            #     break

            # unpack batch
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None

            num_sample += padded.shape[0]
            # print('num samples:', num_sample)
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
            # running_loss += loss.item()
    avg_train_loss = sum(train_loss) / len(train_loss)
    
    return avg_train_loss, num_sample


def val(model, val_datareader, batch_size, loss_type, device, task, mort_pred_time=24, mode='train'):
        
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
            padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
            diagnoses = None
            num_sample += padded.shape[0]
            ##########################
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            ##########################
            y_hat_los, y_hat_mort = model(padded, diagnoses, flat)
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
        # avg_val_loss = sum(val_loss) / len(val_loss)

        avg_val_loss = sum(val_loss) / len(val_loss)

    return avg_val_loss, metrics_list, num_sample



def test(test_hosp_id, model, test_datareader, batch_size, loss_type, device, task, mort_pred_time=48, mode='test'):
        
    if mode == 'test':

        test_y_hat_los = np.array([])
        test_y_los = np.array([])

        test_y_hat_mort = np.array([])
        test_y_mort = np.array([])

        bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
        
        num_sample = 0
        test_loss = []

        model.to(device)
        model.eval()

        test_batches = test_datareader.batch_gen(batch_size=batch_size)
        
        # with torch.no_grad():
        for batch in test_batches:

            padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
            num_sample += padded.shape[0]
            ##########################
            if padded.shape[0] <= 1:
                print('shape <= 1')
                break
            ##########################
            y_hat_los, y_hat_mort = model(padded, diagnoses, flat)
            loss = model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, device, loss_type)

            test_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak
            # val_loss += loss.item()

            if task in ('LoS', 'multitask'):
                print("The task is los!!!")
                test_y_hat_los = np.append(test_y_hat_los, remove_padding(y_hat_los, mask.type(bool_type), device))
                test_y_los = np.append(test_y_los, remove_padding(los_labels, mask.type(bool_type), device))

            if task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                print("The task is mortality!!!")
                test_y_hat_mort = np.append(test_y_hat_mort, remove_padding(y_hat_mort[:, mort_pred_time],
                                                                    mask.type(bool_type)[:, mort_pred_time], device))
                test_y_mort = np.append(test_y_mort, remove_padding(mort_labels[:, mort_pred_time],
                                                                    mask.type(bool_type)[:, mort_pred_time], device))

        test_metrics_list = []

        if task in ('LoS'):
            los_metrics_list = print_metrics_regression(test_y_los, test_y_hat_los) # order: mad, mse, mape, msle, r2, kappa
            test_metrics_list = los_metrics_list

        if task in ('mortality'):
            mort_metrics_list = print_metrics_mortality(test_y_mort, test_y_hat_mort)
            test_metrics_list = mort_metrics_list

        if task in ('multitask'):
            los_metrics_list = print_metrics_regression(test_y_los, test_y_hat_los)
            test_metrics_list.append(los_metrics_list)
            mort_metrics_list = print_metrics_mortality(test_y_mort, test_y_hat_mort)
            test_metrics_list.append(mort_metrics_list)

        avg_test_loss = sum(test_loss) / len(test_loss)

    return avg_test_loss, test_metrics_list, num_sample