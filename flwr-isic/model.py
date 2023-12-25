import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class Baseline(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 8) # for original multi-class classification
        # self.base_model._fc = nn.Linear(nftrs, 2) # for binary classification

    def forward(self, image):
        out = self.base_model(image)
        return out

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import torch.nn.functional as F

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class FedRepBaseline(nn.Module):
    def __init__(self, pretrained=True, arch_name="efficientnet-b0", num_classes=9):
        super(FedRepBaseline, self).__init__()
        self.pretrained = pretrained
        
        # Initialize the base model (EfficientNet)
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        
        # Modify the final fully connected layer for your specific number of classes
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)
        
        # Define additional layers or modifications as needed
        
    def forward(self, image):
        # Forward pass through the base model
        out = self.base_model(image)
        
        # Implement FedRep logic here, e.g., local training, parameter aggregation
        
        return out

from torch import exp, cat
from torch.nn.modules.batchnorm import _BatchNorm


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log().where(mask, torch.zeros_like(y))
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log().where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


# Mean Squared Error (MSE) loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the predictions corresponding to no data should be set to 0
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        # the we set the labels that correspond to no data to be 0 as well
        y = y.where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


class MyBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # hack to work around model.eval() issue
        if not self.training:
            self.eval_momentum = 0  # set the momentum to zero when the model is validating

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum if self.training else self.eval_momentum

        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training=True, momentum=exponential_average_factor, eps=self.eps)  # set training to True so it calculates the norm of the batch


class MyBatchNorm1d(MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class EmptyModule(nn.Module):
    def forward(self, X):
        return X
    
