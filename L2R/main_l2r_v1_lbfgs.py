#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from models.mlp import MLP, MLP2, MLP3
from models.dynamic_net import DynamicNet, ForwardType
from torch.optim import SGD, Adam, LBFGS
from DataLoader.DataLoader import L2R_DataLoader
from Utils.utils import load_train_test_data, init_weights, get_device, eval_ndcg_at_k, get_l2r_group, eval_spearman_kendall
from Misc.Calculations import grad_calc_, loss_calc_, grad_calc_v2, loss_calc_v2
from Misc.metrics import NDCG, DCG
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_version', type=str, required=True)
parser.add_argument('--feat_d', type=int, required=True)
parser.add_argument('--hidden_d', type=int, required=True)
parser.add_argument('--boost_rate', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_nets', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs_per_stage', type=int, required=True)
parser.add_argument('--correct_epoch', type=int ,required=True)
parser.add_argument('--L2', type=float, required=True)
parser.add_argument('--sigma', type=float, required=True)
#parser.add_argument('--normalization', type=bool, required=True)
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true')) 
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
    #optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = Adam(params, lr, weight_decay=weight_decay)
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
    train_group, test_group = get_l2r_group(opt.data)
    print(f'Start training with {opt.data} dataset...')
    c0 = init_gbnn(df_train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f = nn.MSELoss()
    all_scores = []
    all_ensm_losses = []
    all_mdl_losses = []
    # NDCG parameters
    K = 10
    gain_type = 'exp2'


    for stage in range(opt.num_nets):
        t0 = time.time()

        ####  Shuffling dataset experiment 2: shuffle prior to each model ####
        df_train = df_train.sample(frac=1).reset_index(drop=True)

        model = MLP3.get_model(stage, opt)
        model.apply(init_weights)  # Applying uniform xavier initialization for Linear layers (common in L2R)
        if opt.cuda:
            model.cuda()
        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode
        stage_resid = []
        stage_mdlloss = []
        for epoch in range(opt.epochs_per_stage):
            if epoch==0:
                for q, y, x in train_loader.generate_query_batch(df_train, opt.batch_size):
                    if opt.cuda:
                        x = torch.tensor(x, dtype=torch.float32, device=device)
                        y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
                    # Feeding input into ensemble Net
                    middle_feat, out = net_ensemble.forward(x)
                    out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                    resid = y - out
                    
                    stage_resid.append(resid.sum().detach().cpu().numpy())
                    _, out = model(x, middle_feat)
                    out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                    loss = loss_f(net_ensemble.boost_rate*out, resid)
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_mdlloss.append(loss.item()) 

            # L-BFGS part after SGD warm-up
            if epoch > 0:
                optimizer = LBFGS(model.parameters(), lr=opt.lr, max_iter=50)
                for q, y, x in train_loader.generate_query_batch(df_train, df_train.shape[0]):
                    if opt.cuda:
                        x = torch.tensor(x, dtype=torch.float32, device=device)
                        y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
                    middle_feat, out = net_ensemble.forward(x)
                    out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                    resid = y - out
                    def closure():
                        optimizer.zero_grad()
                        _, out = model.forward(x, middle_feat)
                        out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                        loss = loss_f(net_ensemble.boost_rate*out, resid)
                        loss.backward()
                        #net_ensemble.zero_grad()
                        return loss
                    optimizer.step(closure)
                
        net_ensemble.add(model)
        sr = np.mean(stage_resid)
        sml = np.mean(stage_mdlloss)
        print(f'Stage - {stage} resid: {sr}, and model loss: {sml}')
        #print(f'Stage - {stage} resid: {sr}, and model loss: {sml}')

        # fully-corrective step
        stage_loss = []
        lr_scaler = 5
        if stage > 2:
            # Adjusting corrective step learning rate 
            if stage % 15 == 0:
                lr_scaler *= 2

            optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
            for c_epoch in range(opt.correct_epoch):
                if c_epoch == 0:
                    for q, y, x in train_loader.generate_query_batch(df_train, opt.batch_size):
                        if opt.cuda:
                            x = torch.tensor(x, dtype=torch.float32, device=device)
                            y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

                        _, out = net_ensemble.forward_grad(x)
                        out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                        loss = loss_f(out, y)
                        optimizer.zero_grad()
                        loss.backward()
                        #scheduler.step()
                        optimizer.step()
                        stage_loss.append(loss.item())

                # L-BFGS part after SGD warm-up
                if c_epoch > 0:
                    optimizer = LBFGS(net_ensemble.parameters(), lr=opt.lr/lr_scaler, max_iter=50)
                    for q, y, x in train_loader.generate_query_batch(df_train, df_train.shape[0]):
                        if opt.cuda:
                            x = torch.tensor(x, dtype=torch.float32, device=device)
                            y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
                        def closure():
                            optimizer.zero_grad()
                            _, out = net_ensemble.forward_grad(x)
                            out = torch.as_tensor(out, dtype=torch.float32, device=device).view(-1, 1)
                            loss = loss_f(out, y)
                            #net_ensemble.zero_grad()
                            if loss.requires_grad:
                                loss.backward()
                            #scheduler.step()
                            return loss
                        optimizer.step(closure)


        sl = 0
        if stage_loss != []:
            sl = np.mean(stage_loss)
                    
        all_ensm_losses.append(sl)
        all_mdl_losses.append(sml)
        print(f'Stage - {stage}, Boost rate: {net_ensemble.boost_rate} Loss: {sl}')          

        elapsed_tr = time.time()-t0
        
        #net_ensemble.to_eval() # Set the models in ensemble net to eval mode
        # Validation metrics
        #avg_ndcg = metrics(net_ensemble, test)
        ndcg_result = eval_ndcg_at_k(net_ensemble, device, df_test, test_loader, 100000, [5, 10])
        mean_sprm, weighted_mean_sprm, mean_kendall, weighted_mean_kendall = eval_spearman_kendall(net_ensemble, device, df_test, test_loader, test_group)
        print(f'Stage: {stage}, mean spearman: {mean_sprm: .4f}, weighted mean spearman: {weighted_mean_sprm: .4f}, mean kendall tau: {mean_kendall: .4f}, weighted mean kendall: {weighted_mean_kendall: .4f}')
        #ndcg_result = eval_ndcg_at_k(net_ensemble, device, df_train, train_loader, 100000, [5, 10])
        
        all_scores.append([ndcg_result[5], ndcg_result[10]])
        elapsed_te = time.time()-t0 - elapsed_tr
        print(f'Stage: {stage} Training time: {elapsed_tr: .1f} sec and Test time: {elapsed_te: .1f} sec \n')
        #print("finish training " + ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in ndcg_result]),'\n\n')
        #print(f'Stage: {stage}, elapsed  training time: {elapsed_tr:.1f} sec, elapsed  test time: {elapsed_te:.1f} sec, NDCG@10: {avg_ndcg}')

    fname = opt.data + '_NDCG_MSEloss_lbfgs'
    np.savez(fname, all_scores, all_ensm_losses, all_mdl_losses)

