##!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from models.mlp import MLP_1HL, MLP_2HL, MLP_3HL
from models.dynamic_net import DynamicNet, ForwardType
from torch.optim import SGD, Adam
from DataLoader.DataLoader import L2R_DataLoader
from Utils.utils import load_train_test_data, init_weights, get_device, eval_ndcg_at_k, check_for_single_queries
from Misc.Calculations import grad_calc_, loss_calc_, grad_calc_v2, loss_calc_v2
from Misc.metrics import NDCG, DCG
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_version', type=str, required=True)
parser.add_argument('--model_order',default='second', type=str)
parser.add_argument('--feat_d', type=int, required=True)
parser.add_argument('--hidden_d', type=int, required=True)
parser.add_argument('--boost_rate', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_nets', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs_per_stage', type=int, required=True)
parser.add_argument('--correct_epoch', type=int ,required=True)
parser.add_argument('--L2', type=float, required=True)
parser.add_argument('--sigma', type=float, required=True)
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
    if opt.data == 'yahoo':
        data_fold = None
        train_loader, df_train, test_loader, df_test, val_loader, df_val = load_train_test_data(opt.data_dir, data_fold, opt.data, opt.cv)
    elif opt.data == 'microsoft':
        data_fold = 'Fold1'
        train_loader, df_train, test_loader, df_test, val_loader, df_val = load_train_test_data(opt.data_dir, data_fold, opt.data, opt.cv)
    else:
        pass

    if opt.normalization:
        print(opt.normalization)
        df_train, scaler = train_loader.train_scaler_and_transform()
        df_test = test_loader.apply_scaler(scaler)
        if opt.cv:
            df_val = val_loader.apply_scaler(scaler)

    print(f'#Train: {len(df_train.index)}, #Val: {len(df_val)} #Test: {len(df_test.index)}')

    return train_loader, df_train, test_loader, df_test, val_loader, df_val


def get_optim(params, lr, weight_decay):
    #optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer


def init_gbnn(df_train):
    avg = (2**df_train['rel'] - 1)/16

    return avg.mean()

if __name__ == "__main__":
    # prepare datasets
    device = get_device()
    print('Loading data...')
    train_loader, df_train, test_loader, df_test, val_loader, df_val = get_data()

    print(f'Start training with model version {opt.model_version} on {opt.data} dataset...')
    c0 = init_gbnn(df_train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f = nn.MSELoss(reduction='none')
    all_scores = []
    all_ensm_losses = []
    all_mdl_losses = []
    dynamic_br = []
    execution_time = []

    ### Validation parameters ###
    best_ndcg  = 0
    val_ndcg   = best_ndcg
    best_stage = opt.num_nets-1

    # NDCG parameters
    K = 10
    gain_type = 'identity'
    ideal_dcg = NDCG(2**(K-1), gain_type)

    for stage in range(opt.num_nets):
        t0 = time.time()
        model = MLP_2HL.get_model(stage, opt)
        model.apply(init_weights)  # Applying uniform xavier initialization for Linear layers
        if opt.cuda:
            model.cuda()
        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode
        stage_resid = []
        stage_mdlloss = []
        for epoch in range(opt.epochs_per_stage):
            count = 0
            for q, y, x in train_loader.generate_query_batch(df_train, opt.batch_size):
                
                # Removing queries with a single doc.
                idx1 = check_for_single_queries(q, y)
                q, y, x = q[idx1], y[idx1], x[idx1]

                if opt.cuda:
                    x = torch.tensor(x, dtype=torch.float32, device=device)
                    #y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

                # Feeding input into ensemble Net
                middle_feat, out = net_ensemble.forward(x)
                out = torch.as_tensor(out.view(-1, 1), dtype=torch.float32, device=device)
                # Indexing data by qid
                uq = np.unique(q)
                grad_batch = None
                for i in range(len(uq)):
                    idx = np.where(q==uq[i])[0]
                    y_i = y[idx]
                    idx = torch.tensor(idx, device=device)
                    out_i = torch.index_select(out, 0, idx)

                    #if np.sum(y_i)==0 or len(y_i)<=1:
                    #    continue # All irrelevant docs, no useful info
                    N = 1.0 / ideal_dcg.maxDCG(y_i)

                    grad_ord1, grad_ord2 = grad_calc_(y_i, out_i, gain_type, opt.sigma, N, device)
                    if opt.model_order=='second':
                        resid = -grad_ord1/grad_ord2
                    else:
                        resid = -grad_ord1

                    if grad_batch is None:
                        grad_batch = resid
                        grad_ord2_batch = grad_ord2
                    else:
                        grad_ord2_batch = torch.cat((grad_ord2_batch, grad_ord2), dim=0)
                        grad_batch = torch.cat((grad_batch, resid), dim=0)

                _, out = model(x, middle_feat)
                out = torch.as_tensor(out.view(-1, 1), dtype=torch.float32, device=device)

                loss = loss_f(net_ensemble.boost_rate*out, grad_batch)
                loss = grad_ord2_batch*loss
                loss = loss.mean()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_resid.append(grad_batch.sum().item())
                stage_mdlloss.append(loss.item())
                #print('Model parameters after grad update \n')
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if np.isnan(param.data.sum().detach().cpu().numpy()):
                            import ipdb; ipdb.set_trace()


        net_ensemble.add(model)
        sr = -np.mean(stage_resid)
        sml = np.mean(stage_mdlloss)
        #print(f'Stage - {stage} resid: {sr}, and model loss: {sml}')

        # fully-corrective step
        stage_loss = []
        lr_scaler = 2
        if stage >3:
            
            # Adjusting corrective step learning rate 
            if stage % 15 == 0:
                #lr_scaler *= 2
                opt.lr /= 2
                opt.L2 /= 2

            optimizer = get_optim(net_ensemble.parameters(), opt.lr/lr_scaler, opt.L2)
            for _ in range(opt.correct_epoch):
                count = 0
                for q, y, x in train_loader.generate_query_batch(df_train, opt.batch_size):
                    
                    # Removing queries with a single doc
                    idx1 = check_for_single_queries(q, y)
                    q, y, x = q[idx1], y[idx1], x[idx1]

                    if opt.cuda:
                        x = torch.tensor(x, dtype=torch.float32, device=device)
                        #y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

                    _, out = net_ensemble.forward_grad(x)
                    out = torch.as_tensor(out.view(-1, 1), dtype=torch.float32, device=device)
                    uq = np.unique(q)
                    loss_batch = 0
                    for i in range(len(uq)):
                        idx = np.where(q==uq[i])[0]
                        y_i = y[idx]
                        idx = torch.tensor(idx, device=device)
                        out_i = torch.index_select(out, 0, idx)

                        #if np.sum(y_i)==0 or len(y_i)<=1:
                        #    continue # All irrelevant docs, no useful info
                        N = 1.0 / ideal_dcg.maxDCG(y_i) 
                        loss_batch += loss_calc_(y_i, out_i, gain_type, opt.sigma, N, device).mean()

                    loss_batch = loss_batch/len(uq) #opt.batch_size
                    #import ipdb; ipdb.set_trace()
                    optimizer.zero_grad()
                    loss_batch.backward()
                    optimizer.step()
                    stage_loss.append(loss_batch.item())
                    #net_ensemble.zero_grad()
        sl = 0
        if stage_loss != []:
            sl = np.mean(stage_loss)
        # Storing losses and dynamic boost rate
        dynamic_br.append(net_ensemble.boost_rate.item())
        all_ensm_losses.append(sl)
        all_mdl_losses.append(sml)
        print(f'Stage - {stage}, Boost rate: {net_ensemble.boost_rate: .4f} Loss: {sl: .4f}')
        
        elapsed_tr = time.time()-t0
        
        net_ensemble.to_eval() # Set the models in ensemble net to eval mode
        
        ndcg_result = eval_ndcg_at_k(net_ensemble, device, df_test, test_loader, 100000, [5, 10], gain_type)
        if opt.cv:
            val_result = eval_ndcg_at_k(net_ensemble, device, df_val, val_loader, 100000, [5, 10], gain_type, "Validation") 
            if val_result[5] > best_ndcg:
                best_ndcg = val_result[5]
                best_stage = stage

        
        all_scores.append([ndcg_result[5], ndcg_result[10]])
        elapsed_te = time.time()-t0 - elapsed_tr
        # Storing training and test time
        execution_time.append([elapsed_tr, elapsed_te])
        print(f'Stage: {stage} Training time: {elapsed_tr: .1f} sec and Test time: {elapsed_te: .1f} sec \n')

    ### Test results from CV ###
    te_ndcg_5, te_ndcg_10 = all_scores[best_stage][0], all_scores[best_stage][1]
    print(f'Best validation stage: {best_stage}  final Test NDCG@5: {te_ndcg_5:.5f}, NDCG@10: {te_ndcg_10:.5f}')

    fname = './results/' + opt.data +'_'+ str(opt.hidden_d) + 'u_2hl_pairwiseloss'
    np.savez(fname, all_scores=all_scores, all_ensm_losses=all_ensm_losses, all_mdl_losses=all_mdl_losses, dynamic_br=dynamic_br, execution_time=execution_time, options=opt)
