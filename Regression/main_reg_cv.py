#!/usr/bin/env python
import numpy as np
import argparse
import copy
import torch
import torch.nn as nn
import time
from data.sparseloader import DataLoader
from data.data import LibSVMData, LibCSVData, LibSVMRegData
from data.sparse_data import LibSVMDataSp
from models.mlp import MLP_1HL, MLP_2HL, MLP_3HL
from models.dynamic_net import DynamicNet, ForwardType
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--out_f', type=str, required=True)
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data in ['ca_housing', 'ailerons', 'YearPredictionMSD', 'slice_localization']:
        train = LibSVMRegData(opt.tr, opt.feat_d, opt.normalization)
        test = LibSVMRegData(opt.te, opt.feat_d, opt.normalization)
        val = []
        if opt.cv:
            val = copy.deepcopy(train)
            print('Creating Validation set! \n')
            indices = list(range(len(train)))
            cut = int(len(train)*0.95)
            np.random.shuffle(indices)
            train_idx = indices[:cut]
            val_idx = indices[cut:]

            train.feat = train.feat[train_idx]
            train.label = train.label[train_idx]
            val.feat = val.feat[val_idx]
            val.label = val.label[val_idx]
    else:
        pass

    if opt.normalization:
        scaler = StandardScaler()
        scaler.fit(train.feat)
        train.feat = scaler.transform(train.feat)
        test.feat = scaler.transform(test.feat)
        if opt.cv:
            val.feat = scaler.transform(val.feat)
    print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
    return train, test, val


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    #optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer


def root_mse(net_ensemble, loader):
    loss = 0
    total = 0
 
    for x, y in loader:
        if opt.cuda:
            x = x.cuda()

        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        y = y.cpu().numpy().reshape(len(y), 1)
        out = out.cpu().numpy().reshape(len(y), 1)
        loss += mean_squared_error(y, out)* len(y)
        total += len(y)
    return np.sqrt(loss / total)


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

    train, test, val = get_data()
    N = len(train)
    print(opt.data + ' training and test datasets are loaded!')
    train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    if opt.cv:
        val_loader = DataLoader(val, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
    best_rmse = pow(10, 6)
    val_rmse = best_rmse
    best_stage = opt.num_nets-1
    c0 = np.mean(train.label)  #init_gbnn(train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f1 = nn.MSELoss()
    loss_models = torch.zeros((opt.num_nets, 3))
    for stage in range(opt.num_nets):
        t0 = time.time()
        model = MLP_2HL.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
        if opt.cuda:
            model.cuda()

        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode
        stage_mdlloss = []
        for epoch in range(opt.epochs_per_stage):
            for i, (x, y) in enumerate(train_loader):
                
                if opt.cuda:
                    x= x.cuda()
                    y = torch.as_tensor(y, dtype=torch.float32).cuda().view(-1, 1)
                middle_feat, out = net_ensemble.forward(x)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                grad_direction = -(out-y)

                _, out = model(x, middle_feat)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate*out, grad_direction)  # T

                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_mdlloss.append(loss.item()*len(y))

        net_ensemble.add(model)
        sml = np.sqrt(np.sum(stage_mdlloss)/N)
        


        lr_scaler = 3
        # fully-corrective step
        stage_loss = []
        if stage > 0:
            # Adjusting corrective step learning rate 
            if stage % 15 == 0:
                #lr_scaler *= 2
                opt.lr /= 2
                opt.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            for _ in range(opt.correct_epoch):
                stage_loss = []
                for i, (x, y) in enumerate(train_loader):
                    if opt.cuda:
                        x, y = x.cuda(), y.cuda().view(-1, 1)
                    _, out = net_ensemble.forward_grad(x)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    
                    loss = loss_f1(out, y) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item()*len(y))
        #print(net_ensemble.boost_rate)
        # store model
        elapsed_tr = time.time()-t0
        sl = 0
        if stage_loss != []:
            sl = np.sqrt(np.sum(stage_loss)/N)

        print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, model MSE loss: {sml: .5f}, Ensemble Net MSE Loss: {sl: .5f}')

        net_ensemble.to_file(opt.out_f)
        net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP_2HL.get_model(stage, opt))

        if opt.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval() # Set the models in ensemble net to eval mode

        # Train
        tr_rmse  = root_mse(net_ensemble, train_loader)
        if opt.cv:
            val_rmse = root_mse(net_ensemble, val_loader) 
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_stage = stage

        te_rmse  = root_mse(net_ensemble, test_loader)

        print(f'Stage: {stage}  RMSE@Tr: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Te: {te_rmse:.5f}')

        loss_models[stage, 0], loss_models[stage, 1] = tr_rmse, te_rmse

    tr_rmse, te_rmse = loss_models[best_stage, 0], loss_models[best_stage, 1]
    print(f'Best validation stage: {best_stage}  RMSE@Tr: {tr_rmse:.5f}, final RMSE@Te: {te_rmse:.5f}')
    loss_models = loss_models.detach().cpu().numpy()
    fname =  './results/' + opt.data +'_rmse'
    np.savez(fname, rmse=loss_models, params=opt) 

