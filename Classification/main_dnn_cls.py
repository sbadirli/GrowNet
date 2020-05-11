#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.nn as nn
from data.sparseloader import DataLoader
from data.data import LibSVMData, LibCSVData, CriteoCSVData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP, DNN, MLP3
from models.dynamic_net import DynamicNet, ForwardType
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import SGD, Adam
from misc.auc import auc
import time


parser = argparse.ArgumentParser()
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
#parser.add_argument('--sparse', action='store_true')
parser.add_argument('--sparse', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--model_order',default='second', type=str)
parser.add_argument('--out_f', type=str, required=True)
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data in ['a9a', 'ijcnn1']:
        train = LibSVMData(opt.tr, opt.feat_d, opt.normalization)
        test = LibSVMData(opt.te, opt.feat_d, opt.normalization)
    elif opt.data == 'covtype':
        train = LibSVMData(opt.tr, opt.feat_d,opt.normalization, 1, 2)
        test = LibSVMData(opt.te, opt.feat_d, opt.normalization, 1, 2)
    elif opt.data == 'mnist28':
        train = LibSVMData(opt.tr, opt.feat_d, opt.normalization, 2, 8)
        test = LibSVMData(opt.te, opt.feat_d, opt.normalization, 2, 8)
    elif opt.data == 'higgs':
        train = LibSVMData(opt.tr, opt.feat_d,opt.normalization, 0, 1)
        test = LibSVMData(opt.te, opt.feat_d,opt.normalization, 0, 1)
    elif opt.data == 'real-sim':
        train = LibSVMDataSp(opt.tr, opt.feat_d)
        test = LibSVMDataSp(opt.te, opt.feat_d)
    elif opt.data in ['criteo', 'criteo2', 'Higgs']:
        train = LibCSVData(opt.tr, opt.feat_d, 1, 0)
        test = LibCSVData(opt.te, opt.feat_d, 1, 0)
    elif opt.data == 'yahoo.pair':
        train = LibCSVData(opt.tr, opt.feat_d)
        test = LibCSVData(opt.te, opt.feat_d)
    elif opt.data == 'Criteo_Dracula':
        train = CriteoCSVData(opt.tr, opt.feat_d, opt.normalization, 1, 0)
        test = CriteoCSVData(opt.te, opt.feat_d, opt.normalization, 1, 0)
    else:
        pass

    if opt.normalization:
        scaler = StandardScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)
    print(f'#Train: {len(train)}, #Test: {len(test)}')
    return train, test


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer

def accuracy(model, test_loader):
    #TODO once the net_ensemble contains BN, consider eval() mode
    correct = 0
    total = 0
    loss = 0
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            out = model.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()
        total += y.numel()
    return correct / total

def logloss(model, test_loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss() # Binary cross entopy loss with logits, reduction=mean by default
    for x, y in test_loader:
        if opt.cuda:
            x, y= x.cuda(), y.cuda().view(-1, 1)
        y = (y + 1) / 2
        with torch.no_grad():
            out = model.forward(x)
        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
        loss += loss_f(out, y)
        total += 1

    return loss / total

def auc_score(model, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        if opt.cuda:
            x = x.cuda()
        with torch.no_grad():
            out = model.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    score = auc(actual, posterior)
    return score

if __name__ == "__main__":
    # prepare datasets
    #torch.autograd.set_detect_anomaly(True)
    train, test = get_data()
    print(opt.data + ' training and test datasets are loaded!')
    train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    if opt.data == 'higgs':
        indices = list(range(len(train)))
        split = 1000000
        #np.random.shuffle(indices)
        train_idx = indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train, opt.batch_size, sampler = train_sampler, drop_last=True, num_workers=2)
    loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
    model_scores = []

    all_mdl_losses_te = []
    all_mdl_losses = []
    elapsed_tr = 0

    model = DNN(opt.feat_d, opt.hidden_d, 30, False,True, 0.3)  # 40 - # hidden layers, False - sparse, 0.3 - dropout probability
    if opt.cuda:
        model.cuda()

    optimizer = get_optim(model.parameters(), opt.lr, opt.L2)

    stage_mdlloss = []
    for epoch in range(opt.epochs_per_stage):
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            if opt.cuda:
                x, y= x.cuda(), y.cuda().view(-1, 1)

            ######### My addition #############
            out = model(x)
            out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
            #out = nn.functional.tanh(out)
            y = (y + 1.0) / 2.0
            loss = loss_f2(out, y).mean()  # T
            #loss = loss_f1(net_ensemble.boost_rate*out/nwtn_weights, grad_direction/nwtn_weights).sum()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #stage_mdlloss.append(loss.item()) 
        #print(net_ensemble.boost_rate)
        #net_ensemble.add(model, net_ensemble.boost_rate)
        
        elapsed_tr += time.time()-t0

        if (epoch+1) % 25 ==0:
            t_0 = time.time()
            model.eval()

            # Train
            print('Acc results from epoch := ' + str(epoch+1) + '\n')
            acc_tr = accuracy(model, train_loader)
            # Test
            acc_te = accuracy(model, test_loader)
            # AUC
            score = auc_score(model, test_loader)
            print(f'Acc@Tr: {acc_tr:.4f}, Acc@Te: {acc_te:.4f}, AUC@Te: {score:.4f}')

            model_scores.append([acc_tr, acc_te, score])

            sml_te = logloss(model, test_loader)
            sml_tr = logloss(model, train_loader)
            all_mdl_losses_te.append(sml_te)
            all_mdl_losses.append(sml_tr)

            elapsed_te = time.time() - t_0
            print(f'Epoch - {epoch+1}, Training Loss: {sml_tr: .4f}, Test Loss: {sml_te: .4f}')
            print(f'Epoch - {epoch+1}, total 25 epochs Training time: {elapsed_tr: .1f} sec and Test time: {elapsed_te: .1f} sec \n')
            elapsed_tr = 0
            model.train(True)

        
        #print('Logloss results from stage := ' + str(stage) + '\n')
        #ll_tr = logloss(net_ensemble, train_loader)
        # Test
        #ll_te = logloss(net_ensemble, test_loader)
        #print(f'Logloss@Tr: {ll_tr:.8f}, Logloss@Te: {ll_te:.8f}')
        #loss_models[stage, 0], loss_models[stage, 1] = ll_tr, ll_te

    fname = opt.data + '_dnn_clf_loss_bn'
    np.savez(fname, all_mdl_losses, all_mdl_losses_te, model_scores)

