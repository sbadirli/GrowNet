#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from models.mlp import MLP, MLP2, MLP3
from models.dynamic_net import DynamicNet, ForwardType
from torch.optim import SGD, Adam
from DataLoader.DataLoader import L2R_DataLoader
from Utils.utils import load_train_test_data, init_weights, get_device, eval_ndcg_at_k
from Misc.Calculations import grad_calc_, loss_calc_, grad_calc_v2, loss_calc_v2
from Misc.metrics import NDCG, DCG
import time

parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, required=True)
parser.add_argument('--hidden_d', type=int, required=True)
parser.add_argument('--boost_rate', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_nets', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
#parser.add_argument('--tr', type=str, required=True)
#parser.add_argument('--te', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs_per_stage', type=int, required=True)
parser.add_argument('--correct_epoch', type=int ,required=True)
parser.add_argument('--L2', type=float, required=True)
parser.add_argument('--sigma', type=float, required=True)
parser.add_argument('--normalization', type=bool, required=True)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data == 'yahoo':
        data_fold = None
        train_loader, df_train, test_loader, df_test = load_train_test_data(data_fold, opt.data)
    elif opt.data == 'microsoft':
        data_fold = 'Fold1'
        train_loader, df_train, test_loader, df_test = load_train_test_data(data_fold, opt.data)
    else:
        pass

    if opt.normalization:
        print(opt.normalization)
        df_train, scaler = train_loader.train_scaler_and_transform()
        df_test = test_loader.apply_scaler(scaler)

    return train_loader, df_train, test_loader, df_test


def get_optim(params, lr, weight_decay):
    optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    #optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer


def metrics(net_ensemble, test_data):
    y_real = []
    y_pred = []
    qid = -1
    count = 0
    ndcg = 0
    for i in range(len(test_data)):
        x, y, w = test_data[i]
        if qid != w:
            if qid != -1:
                # calculate metrics
                count += 1
                y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
                score = ndcg_score(y_real, y_pred)
                if np.isnan(score):
                    print(score, y_real, y_pred)
                ndcg += score
            # start new query
            y_real = []
            y_pred = []
            qid = w
        y_real.append(y)
        x = torch.from_numpy(x).unsqueeze(0).cuda()
        _, out = net_ensemble.forward(x)
        y_pred.append(out.item())

    avg_ndcg = ndcg / count
    return avg_ndcg

#def init_gbnn(train):
#    avg = 0
#    for i in range(len(train)):
#       avg += (2 ** train['1'][1] - 1) / 16
#    return float(avg / len(train))

def init_gbnn(df_train):
    avg = (2**df_train['rel'] - 1)/16
    return avg.mean()

if __name__ == "__main__":
    # prepare datasets
    device = get_device()
    print('Loading data...')
    train_loader, df_train, test_loader, df_test = get_data()

    print(f'Start training with {opt.data} dataset...')
    c0 = init_gbnn(df_train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f = nn.MSELoss()
    all_scores = []

    # NDCG parameters
    K = 10
    gain_type = 'exp2'
    ideal_dcg = NDCG(2**(K-1), gain_type)
    all_stages_loss = []
    for stage in range(opt.num_nets):
        t0 = time.time()
        model = MLP3.get_model(stage, opt)
        #model.apply(init_weights)  # Applying uniform xavier initialization for Linear layers (common in L2R)
        if opt.cuda:
            model.cuda()
        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode
        stage_resid = []
        stage_mdlloss = []
        for epoch in range(opt.epochs_per_stage):
            grad_batch, y_pred_batch = None, None
            count = 0
            for x, y in train_loader.generate_batch_per_query():
                
                if np.sum(y)==0 or len(y)<=1:
                    continue # All irrelevant docs, no useful info
                N = 1.0 / (ideal_dcg.maxDCG(y))  
                x = torch.tensor(x, dtype=torch.float32, device=device)
                # Feeding input into ensemble Net
                middle_feat, out = net_ensemble.forward(x)

                # Calculating gradient (1st or 2nd order)
                out = torch.as_tensor(out.view(-1, 1), dtype=torch.float32, device=device)
                grad_ord1, grad_ord2 = grad_calc_(y, out, gain_type, opt.sigma, N, device)
                _, out = model(x, middle_feat)

                if grad_batch is None:
                    grad_batch = -grad_ord1/grad_ord2
                    y_pred_batch = out.view(-1, 1)
                else:
                    grad_batch = torch.cat((grad_batch, -grad_ord1/grad_ord2), dim=0)
                    y_pred_batch = torch.cat((y_pred_batch, out.view(-1, 1)), dim=0)

                count += 1
                if count % opt.batch_size == 0:
                    #print(grad_batch)
                    loss = loss_f(net_ensemble.boost_rate*y_pred_batch, grad_batch)
                    stage_resid.append(grad_batch.sum().detach().cpu().numpy())
                    stage_mdlloss.append(loss.detach().cpu().numpy())
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    grad_batch, y_pred_batch = None, None


        net_ensemble.add(model)
        sr = np.mean(stage_resid)
        sml = np.mean(stage_mdlloss)
        print(f'Stage - {stage} resid: {sr}, and model loss: {sml}')

        # fully-corrective step
        stage_loss = []
        if stage != 0:
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / 5, opt.L2)
            for _ in range(opt.correct_epoch):
                loss_batch = 0.0
                count = 0
                for x, y in train_loader.generate_batch_per_query():
                    
                    if np.sum(y)==0 or len(y)<=1:
                        continue
                    N = 1.0 / (ideal_dcg.maxDCG(y))   
                    x = torch.tensor(x, dtype=torch.float32, device=device)
                    _, out = net_ensemble.forward_grad(x)
                    # Need to calculate Loss function 
                    loss_batch += loss_calc_(y, net_ensemble.boost_rate*out, gain_type, opt.sigma, N, device).mean() 
                    count += 1
                    if count % opt.batch_size == 0:

                        loss_batch = loss_batch/opt.batch_size
                        optimizer.zero_grad()
                        loss_batch.backward()
                        optimizer.step()
                        stage_loss.append(loss_batch.detach().cpu().numpy())
                        loss_batch = 0.0
                        #net_ensemble.zero_grad()
                        
                        
                    
        sl = np.mean(stage_loss)
        print(f'Stage - {stage} Loss: {sl}')
        elapsed_tr = time.time()-t0
        
        net_ensemble.to_eval() # Set the models in ensemble net to eval mode
        # Validation metrics
        #avg_ndcg = metrics(net_ensemble, test)
        ndcg_result = eval_ndcg_at_k(net_ensemble, device, df_test, test_loader, 100000, [5, 10])
        #ndcg_result = eval_ndcg_at_k(net_ensemble, device, df_train, train_loader, 100000, [5, 10])
        
        all_scores.append(ndcg_result)
        elapsed_te = time.time()-t0 - elapsed_tr
        print(f'Stage: {stage} Training time: {elapsed_tr: .1f} sec and Test time: {elapsed_te: .1f} sec \n')
        #print("finish training " + ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in ndcg_result]),'\n\n')
        #print(f'Stage: {stage}, elapsed  training time: {elapsed_tr:.1f} sec, elapsed  test time: {elapsed_te:.1f} sec, NDCG@10: {avg_ndcg}')

    fname = opt.data + '_NDCG_pairwiseloss'
    fname2 = opt.data + 'V2_loss'
    np.save(fname, all_scores)
    np.save(fname2, all_stages_loss)

