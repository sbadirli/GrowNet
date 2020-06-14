import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .splinear import SpLinear


class MLP_1HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_1HL, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.out_layer = nn.Linear(dim_hidden1, 1)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm1d(dim_hidden1)
            self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.in_layer(x)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP_1HL(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_2HL, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(int(dim_hidden2), 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.lrelu(self.in_layer(x))
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP_2HL(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model

class MLP_3HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_3HL, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden2, dim_hidden1)
        self.out_layer = nn.Linear(dim_hidden1, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)
        # print('Batch normalization is processed!')

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.lrelu(self.in_layer(x))
        out = self.bn(out)
        out = self.lrelu(self.hidden_layer(out))
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP_3HL(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model

class MLP_4HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_3HL, self).__init__()
        self.in_layer = SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden2, dim_hidden1)
        self.out_layer = nn.Linear(dim_hidden1, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)
        # print('Batch normalization is processed!')

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.lrelu(self.in_layer(x)) #HL-1
        out = self.bn(out)
        out = self.lrelu(self.hidden_layer(out)) #HL-2
        out = self.bn(out)
        out = self.lrelu(self.hidden_layer(out)) #HL-3
        out = self.bn(out)
        out = self.hidden_layer(out) #HL-4
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP_3HL(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
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
            if drop_out > 0:
                hidden_layers.append(nn.Dropout(drop_out))
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.out_layer = nn.Linear(dim_hidden, 1)

    def forward(self, x):
        out = self.in_act(self.in_layer(x))
        out = self.hidden_layers(out)
        out = self.out_layer(out)
        return out.squeeze()
