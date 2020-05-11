#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.nn as nn
from data.sparseloader import DataLoader
from data.data import LibSVMData, LibCSVData, CriteoCSVData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP, MLP2, MLP3
from models.dynamic_net import DynamicNet, ForwardType
#from models.dynamic_net_v2 import DynamicNet_v2
from torch.optim import SGD, Adam, LBFGS
from misc.auc import auc


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
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--model_order', type=str, required=True)
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
    print(f'#Train: {len(train)}, #Test: {len(test)}')
    return train, test


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
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
    loss_f = nn.BCEWithLogitsLoss(reduction='none') # Binary cross entopy loss with logits, reduction=mean by default
    for x, y in test_loader:
        if opt.cuda:
            x, y= x.cuda(), y.cuda()
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        loss += (w*loss_f(out, y)).sum()
        #total += y.numel()
        total += w.sum()
    return loss / total

def auc_score(net_ensemble, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        if opt.cuda:
            x = x.cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    score = auc(actual, posterior)
    return score

def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    #print(f'Blind Logloss: {blind_acc}')
    return float(np.log(positive / negative))

if __name__ == "__main__":
    # prepare datasets
    #torch.autograd.set_detect_anomaly(True)
    train, test = get_data()
    print(opt.data + ' training and test datasets are loaded!')
    train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=True, num_workers=2)
    bfgs_tr_loader = DataLoader(train, train.__len__(), shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    c0 = init_gbnn(train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f1 = nn.MSELoss(reduction='none')
    loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
    loss_models = torch.zeros((opt.num_nets, 3))
    for stage in range(opt.num_nets):
        model = MLP3.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
        if opt.cuda:
            model.cuda()

        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode
        for epoch in range(opt.epochs_per_stage):
            for i, (x, y) in enumerate(train_loader):
                if opt.cuda:
                    x, y= x.cuda(), y.cuda()
                middle_feat, out = net_ensemble.forward(x)
                #resid = y / (1.0 + torch.exp(y * out)) # Make sense now, out is result of linear layer
                if opt.model_order=='first':
                    grad_direction = y / (1.0 + torch.exp(y * out))
                else:
                    grad_direction = y * (1.0 + torch.exp(-y * out))
                    out = torch.as_tensor(out)
                    nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                ######### My addition #############
                _, out = model(x, middle_feat)
                #out = nn.functional.tanh(out)
                loss = loss_f1(net_ensemble.boost_rate*out, grad_direction).mean()  # T
                #loss = loss_f1(net_ensemble.boost_rate*out/nwtn_weights, grad_direction/nwtn_weights).sum()
                model.zero_grad()
                loss.backward()
                optimizer.step()
            # L-BFGS part after SGD warm-up
            if epoch > 0:
                optimizer = LBFGS(model.parameters(), lr=opt.lr, max_iter=50)
                for i, (x, y) in enumerate(bfgs_tr_loader):
                    if opt.cuda:
                        x, y= x.cuda(), y.cuda()
                    middle_feat, out = net_ensemble.forward(x)
                    #resid = y / (1.0 + torch.exp(y * out)) # Make sense now, out is result of linear layer
                    if opt.model_order=='first':
                        grad_direction = y / (1.0 + torch.exp(y * out))
                    else:
                        grad_direction = y * (1.0 + torch.exp(-y * out))
                        out = torch.as_tensor(out)
                        nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                    def closure():
                        optimizer.zero_grad()
                        _, out = net_ensemble.forward(x)
                        #out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                        loss = loss_f1(net_ensemble.boost_rate*out, grad_direction).mean()
                        loss.backward()
                        return loss
                    optimizer.step(closure)
                    #net_ensemble.zero_grad()
        #print(net_ensemble.boost_rate)
        #net_ensemble.add(model, net_ensemble.boost_rate)
        net_ensemble.add(model)


        lr_scaler = 5
        # fully-corrective step
        if stage != 0:
            # Adjusting corrective step learning rate 
            if stage % 20 == 0:
                lr_scaler *= 2
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            for c_epoch in range(opt.correct_epoch):
                
                if c_epoch<=1:
                    for i, (x, y) in enumerate(train_loader):
                        if opt.cuda:
                            x, y = x.cuda(), y.cuda()
                        _, out = net_ensemble.forward_grad(x)
                        y = (y + 1.0) / 2.0
                        #loss = (w*loss_f2(out, y)).sum()/w.sum() #Do NOT forget to normalize 
                        loss = loss_f2(out, y).mean() # Not including weights during training!!!
                        #net_ensemble.zero_grad()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if c_epoch > 1:
                    optimizer = LBFGS(net_ensemble.parameters(), lr = opt.lr / lr_scaler, max_iter=50)
                    for i, (x, y) in enumerate(bfgs_tr_loader):
                        if opt.cuda:
                            x, y= x.cuda(), y.cuda()
                        y = (y + 1.0) / 2.0
                        #y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
                        def closure():
                            optimizer.zero_grad()
                            _, out = net_ensemble.forward_grad(x)
                            loss = loss_f2(out, y).mean()
                            #out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                            loss.backward()
                            return loss
                        optimizer.step(closure)
        
        print(net_ensemble.boost_rate)
        # store model
        net_ensemble.to_file(opt.out_f)
        net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP3.get_model(stage, opt))

        if opt.cuda:
            net_ensemble.to_cuda()
        # It seems we need to run the models in cuda again after loading from directory
        net_ensemble.to_eval() # Set the models in ensemble net to eval mode

        # Train
        print('Acc results from stage := ' + str(stage) + '\n')
        acc_tr = accuracy(net_ensemble, train_loader)
        # Test
        acc_te = accuracy(net_ensemble, test_loader)
        # AUC
        score = auc_score(net_ensemble, test_loader)
        print(f'Acc@Tr: {acc_tr:.4f}, Acc@Te: {acc_te:.4f}, AUC@Te: {score:.4f}')


        
        #print('Logloss results from stage := ' + str(stage) + '\n')
        #ll_tr = logloss(net_ensemble, train_loader)
        # Test
        #ll_te = logloss(net_ensemble, test_loader)
        #print(f'Logloss@Tr: {ll_tr:.8f}, Logloss@Te: {ll_te:.8f}')
        #loss_models[stage, 0], loss_models[stage, 1] = ll_tr, ll_te
        loss_models[stage, 0], loss_models[stage, 1], loss_models[stage, 2] = acc_tr, acc_te, score

    loss_models = loss_models.detach().cpu().numpy()
    fname = 'tr_ts_' + opt.data +'_auc'
    np.save(fname, loss_models) 

