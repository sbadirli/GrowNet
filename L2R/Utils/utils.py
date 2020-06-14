"""
Common function used in training Learn to Rank
"""
from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from DataLoader.DataLoader import get_time, L2R_DataLoader
from Misc.metrics import NDCG


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0" 
    else:
        device = "cpu"
    print("Using device ", device)
    return device


def get_ckptdir(net_name, net_structure, sigma=None):
    net_name = '{}-{}'.format(net_name, '-'.join([str(x) for x in net_structure]))
    if sigma:
        net_name += '-scale-{}'.format(sigma)
    ckptdir = os.path.join(os.path.dirname(__file__), 'ckptdir')
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    ckptfile = os.path.join(ckptdir, net_name)
    print("checkpoint dir:", ckptfile)
    return ckptfile


def save_to_ckpt(ckpt_file, epoch, model, optimizer, lr_scheduler):
    ckpt_file = ckpt_file + '_{}'.format(epoch)
    print(get_time(), 'save to ckpt {}'.format(ckpt_file))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, ckpt_file)
    print(get_time(), 'finish save to ckpt {}'.format(ckpt_file))


def load_train_test_data(d_dir, data_fold, dataname, cv):
    """
    :param data_fold: str, which fold's data was going to use to train
    :return:
    """
    df_val = []
    val_loader = None

    train_file, test_file = dataname + ".train.txt", dataname +  ".test.txt"

    if data_fold:
        data_dir = os.path.join(d_dir, 'MSLR-WEB10K/')
        train_data = os.path.join(data_dir, data_fold, train_file)
        train_loader = L2R_DataLoader(train_data)
        df_train = train_loader.load()

        test_data = os.path.join(data_dir, data_fold, test_file)
        test_loader = L2R_DataLoader(test_data)
        df_test = test_loader.load()

        if cv:
            val_file = dataname + ".vali.txt"
            val_data = os.path.join(data_dir, data_fold, val_file)
            val_loader = L2R_DataLoader(val_data)
            df_val = val_loader.load()

    else: 
        data_dir = d_dir + 'Yahoo/'
        train_data = os.path.join(os.path.dirname(__file__), data_dir, train_file)
        train_loader = L2R_DataLoader(train_data)
        df_train = train_loader.load()

        test_data = os.path.join(os.path.dirname(__file__), data_dir, test_file)
        test_loader = L2R_DataLoader(test_data)
        df_test = test_loader.load()

        if cv:
            val_file = dataname + ".vali.txt"
            val_data = os.path.join(os.path.dirname(__file__), data_dir, val_file)
            val_loader = L2R_DataLoader(val_data)
            df_val = val_loader.load()
    
    return train_loader, df_train, test_loader, df_test, val_loader, df_val


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def eval_cross_entropy_loss(model, device, loader, phase="Eval", sigma=1.0):
    """
    formula in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

    C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
    when S_ij = 1:  C = log(1 + exp(-sigma(si - sj)))
    when S_ij = -1: C = log(1 + exp(-sigma(sj - si)))
    sigma can change the shape of the curve
    """
    print(get_time(), "{} Phase evaluate pairwise cross entropy loss".format(phase))
    model.eval()
    with torch.set_grad_enabled(False):
        total_cost = 0
        total_pairs = loader.get_num_pairs()
        pairs_in_compute = 0
        for X, Y in loader.generate_batch_per_query(loader.df):
            Y = Y.reshape(-1, 1)
            rel_diff = Y - Y.T
            pos_pairs = (rel_diff > 0).astype(np.float32)
            num_pos_pairs = np.sum(pos_pairs, (0, 1))
            # skip negative sessions, no relevant info:
            if num_pos_pairs == 0:
                continue
            neg_pairs = (rel_diff < 0).astype(np.float32)
            num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same
            pos_pairs = torch.tensor(pos_pairs, device=device)
            neg_pairs = torch.tensor(neg_pairs, device=device)
            Sij = pos_pairs - neg_pairs
            # only calculate the different pairs
            diff_pairs = pos_pairs + neg_pairs
            pairs_in_compute += num_pairs

            X_tensor = torch.Tensor(X).to(device)
            y_pred = model(X_tensor)
            y_pred_diff = y_pred - y_pred.t()

            # logsigmoid(x) = log(1 / (1 + exp(-x))) equivalent to log(1 + exp(-x))
            C = 0.5 * (1 - Sij) * sigma * y_pred_diff - F.logsigmoid(-sigma * y_pred_diff)
            C = C * diff_pairs
            cost = torch.sum(C, (0, 1))
            if cost.item() == float('inf') or np.isnan(cost.item()):
                import ipdb; ipdb.set_trace()
            total_cost += cost

        assert total_pairs == pairs_in_compute
        avg_cost = total_cost / total_pairs
    print(
        get_time(),
        "{} Phase pairwise corss entropy loss {:.6f}, total_paris {}".format(
            phase, avg_cost.item(), total_pairs
        ))


def eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, k_list, gain_type, phase="Eval"):
    # print("Eval Phase evaluate NDCG @ {}".format(k_list))
    ndcg_metrics = {k: NDCG(k, gain_type) for k in k_list}
    qids, rels, scores = [], [], []
    inference_model.to_eval() # Set the models in ensemble net to eval mode
    with torch.no_grad():
        for qid, rel, x in valid_loader.generate_query_batch(df_valid, batch_size):
            if x is None or x.shape[0] == 0:
                continue
            _, y_tensor = inference_model.forward(torch.Tensor(x).to(device))
            scores.append(y_tensor.cpu().numpy().squeeze())
            qids.append(qid)
            rels.append(rel)

    qids = np.hstack(qids)
    rels = np.hstack(rels)
    scores = np.hstack(scores)
    result_df = pd.DataFrame({'qid': qids, 'rel': rels, 'score': scores})
    session_ndcgs = defaultdict(list)
    for qid in result_df.qid.unique():
        result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
        rel_rank = result_qid.rel.values
        for k, ndcg in ndcg_metrics.items():
            if ndcg.maxDCG(rel_rank) == 0:
                continue
            ndcg_k = ndcg.evaluate(rel_rank)
            if not np.isnan(ndcg_k):
                session_ndcgs[k].append(ndcg_k)

    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    ndcg_result_print = ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in k_list])
    print(get_time(), "{} Phase evaluate {}".format(phase, ndcg_result_print))
    return ndcg_result

def eval_spearman_kendall(inference_model, device, df_test, test_loader, test_group):

    # Switch the model into eval mode
    inference_model.to_eval()
    batch_size = df_test.shape[0]
    with torch.no_grad():
        for qid, rel, x in test_loader.generate_query_batch(df_test, batch_size):
            _, y_tensor = inference_model.forward(torch.Tensor(x).to(device))
            preds = y_tensor.cpu().numpy().tolist()
            labels = rel.tolist()

    mean_kt, weighted_mean_kt = mean_kendall_tau(labels, preds, test_group)
    mean_sr, weighted_mean_sr = mean_spearman_rho(labels, preds, test_group)

    return mean_sr, weighted_mean_sr, mean_kt, weighted_mean_kt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def check_for_single_queries(q, y):
    idx = []
    uq = np.unique(q)
    for i in range(len(uq)):
        idd = np.where(q==uq[i])[0]
        if len(idd) > 1 and len(np.unique(y[idd]))>1:
            idx.append(idd)
    idx = np.concatenate(idx).ravel().tolist()
    return idx


def get_args_parser():
    """Common Args needed for different Learn to Rank training method.
    :rtype: ArgumentParser
    """
    parser = ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    parser.add_argument("--optim", dest="optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--leaky_relu", dest="leaky_relu", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--ndcg_gain_in_train", dest="ndcg_gain_in_train",
        type=str, default="exp2", choices=["exp2","identity"]
    )
    parser.add_argument("--small_dataset", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--double_precision", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--standardize", type=str2bool, nargs="?", const=True, default=False)
    return parser
