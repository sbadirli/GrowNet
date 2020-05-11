import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .splinear import SpLinear

#activation = F.relu
activation = F.selu


# Binary output MLP
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False):
        super(MLP, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.hidden_layer0 = nn.Linear(dim_hidden2, dim_hidden1)
        self.hidden_layer1 = nn.Linear(dim_hidden1, 1)
        self.bn = nn.BatchNorm1d(dim_hidden2)

    def forward(self, x, lower_f):
        out = activation(self.in_layer(x))
        if lower_f is not None:
            out = torch.cat([out, lower_f], dim=1)
        out = self.bn(out)
        out = activation(self.hidden_layer0(out))
        return out, self.hidden_layer1(out).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_hidden = opt.hidden_d
        else:
            dim_hidden = opt.hidden_d * 2
        model = MLP(opt.feat_d, opt.hidden_d, dim_hidden, opt.sparse)
        return model


# Binary output MLP
class MLP2(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=False):
        super(MLP2, self).__init__()
        self.bn = bn
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.hidden_layer = nn.Linear(dim_hidden2, dim_hidden1)
        self.out_layer = nn.Linear(dim_hidden1, 1)
        if bn:
            self.bn = nn.BatchNorm1d(dim_hidden1)
            #print('Batch normalization is processed!')
    
    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
        out = activation(self.in_layer(x))
        if self.bn:
            out = self.bn(out)
        out = activation(self.hidden_layer(out))
        return out, self.out_layer(out).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP2(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model

# Binary output MLP
class MLP3(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP3, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.0)
        self.hidden_layer = nn.Linear(dim_hidden2, dim_hidden1)
        self.out_layer = nn.Linear(dim_hidden1, 1)
        if bn:
            self.bn = nn.BatchNorm1d(dim_hidden1)
            #print('Batch normalization is processed!')

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
        out = activation(self.in_layer(x))
        if self.bn:
            out = self.bn(out)
        #out = self.dropout_layer(out)
        out = activation(self.hidden_layer(out))
        return out, self.out_layer(out).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP3(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model

class DNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_hidden=20, sparse=False, bn=True, drop_out=0.3):
        super(DNN, self).__init__()
        if sparse:
            self.in_layer = SpLinear(dim_in, dim_hidden)
        else:
            self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.in_act = nn.SELU()
        hidden_layers = []
        for _ in range(n_hidden):
            hidden_layers.append(nn.Linear(dim_hidden, dim_hidden))
            if bn:
                hidden_layers.append(nn.BatchNorm1d(dim_hidden))
            hidden_layers.append(nn.SELU())
            if drop_out>0:
                hidden_layers.append(nn.Dropout(drop_out))
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.out_layer = nn.Linear(dim_hidden, 1)

    def forward(self, x):
        out = self.in_act(self.in_layer(x))
        out = self.hidden_layers(out)
        out = self.out_layer(out)
        return out.squeeze()


