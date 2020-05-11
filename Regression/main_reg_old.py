#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.nn as nn
from data.sparseloader import DataLoader
from data.data import LibSVMData, LibCSVData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP, MLP2
from models.dynamic_net import DynamicNet, ForwardType
from torch.optim import SGD, Adam

parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, required=True)
parser.add_argument('--hidden_d', type=int, required=True)
parser.add_argument('--boost_rate', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_nets', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--tr', type=str, required=True)
parser.add_argument('--te', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs_per_stage', type=int, required=True)
parser.add_argument('--correct_epoch', type=int ,required=True)
parser.add_argument('--L2', type=float, required=True)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data in ['a9a', 'ijcnn1']:
        train = LibSVMData(opt.tr, opt.feat_d)
        test = LibSVMData(opt.te, opt.feat_d)
    elif opt.data == 'covtype':
        train = LibSVMData(opt.tr, opt.feat_d, 1, 2)
        test = LibSVMData(opt.te, opt.feat_d, 1, 2)
    elif opt.data == 'mnist28':
        train = LibSVMData(opt.tr, opt.feat_d, 2, 8)
        test = LibSVMData(opt.te, opt.feat_d, 2, 8)
    elif opt.data == 'real-sim':
        train = LibSVMDataSp(opt.tr, opt.feat_d)
        test = LibSVMDataSp(opt.te, opt.feat_d)
    elif opt.data in ['criteo', 'criteo2']:
        train = LibCSVData(opt.tr, opt.feat_d, 1, 0)
        test = LibCSVData(opt.te, opt.feat_d, 1, 0)
    else:
        pass
    print(f'#Train: {len(train)}, #Test: {len(test)}')
    return train, test


def get_optim(params, lr, weight_decay):
    optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer

def accuracy(net_ensemble, test_loader):
    #TODO once the net_ensemble contains BN, consider eval() mode
    correct = 0
    total = 0
    loss = 0
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            middle_feat, out = net_ensemble.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()
        total += y.numel()
    return correct / total

def logloss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss(size_average=False)
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        loss += loss_f(out, y)
        total += y.numel()
    return loss / total

def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    return float(np.log(positive / negative))

if __name__ == "__main__":
    # prepare datasets
    train, test = get_data()
    train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    c0 = init_gbnn(train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f1 = nn.MSELoss()
    loss_f2 = nn.BCEWithLogitsLoss()
    for stage in range(opt.num_nets):
        model = MLP2.get_model(stage, opt)
        if opt.cuda:
            model.cuda()
        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        for epoch in range(opt.epochs_per_stage):
            for i, (x, y) in enumerate(train_loader):
                if opt.cuda:
                    x, y = x.cuda(), y.cuda()
                middle_feat, out = net_ensemble.forward(x)
                resid = y / (1.0 + torch.exp(y * out))
                _, out = model(x, middle_feat)
                loss = loss_f1(out, resid)
                model.zero_grad()
                loss.backward()
                optimizer.step()
        net_ensemble.add(model)

        # fully-corrective step
        if stage != 0:
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / 10, opt.L2)
            for _ in range(opt.correct_epoch):
                for i, (x, y) in enumerate(train_loader):
                    if opt.cuda:
                        x, y = x.cuda(), y.cuda()
                    _, out = net_ensemble.forward_grad(x)
                    y = (y + 1.0) / 2.0
                    loss = loss_f2(out, y)
                    net_ensemble.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Train
        acc_tr = accuracy(net_ensemble, train_loader)
        # Test
        acc_te = accuracy(net_ensemble, test_loader)
        print(f'Acc@Tr: {acc_tr}, Acc@Te: {acc_te}, Diff: {acc_tr-acc_te}')

