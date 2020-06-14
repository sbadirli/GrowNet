import numpy as np
import pandas as pd
import torch

def loss_calc_(y_true, y_pred, gain_type, sigma, N, device):

    # compute the rank order of each document
    rank_df = pd.DataFrame({"y": y_true, "doc": np.arange(y_true.shape[0])})
    rank_df = rank_df.sort_values("y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1

    pos_pairs_score_diff = 1.0 + torch.exp(-sigma * (y_pred - y_pred.t()))

    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).view(-1, 1)
    rel_diff = y_tensor - y_tensor.t()
    pos_pairs = (rel_diff > 0).type(torch.float32)
    neg_pairs = (rel_diff < 0).type(torch.float32)
    Sij = pos_pairs - neg_pairs
    if gain_type == "exp2":
        gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
    elif gain_type == "identity":
        gain_diff = y_tensor - y_tensor.t()
    else:
        raise ValueError("NDCG_gain method not supported yet {}".format(ndcg_gain_in_train))

    rank_order_tensor = torch.tensor(rank_order, dtype=torch.float32, device=device).view(-1, 1)
    decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

    loss = (0.5*sigma*(1 - Sij)*(y_pred - y_pred.t()) + torch.log(pos_pairs_score_diff))
    loss = torch.sum(loss, 1, keepdim=True)
    #import ipdb; ipdb.set_trace()
    return  loss


def grad_calc_(y_true, y_pred, gain_type, sigma, N, device):

    # compute the rank order of each document
    rank_df = pd.DataFrame({"y": y_true, "doc": np.arange(y_true.shape[0])})
    rank_df = rank_df.sort_values("y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1

    pos_pairs_score_diff = 1.0/(1.0 + torch.exp(sigma * (y_pred - y_pred.t())))

    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).view(-1, 1)
    rel_diff = y_tensor - y_tensor.t()
    pos_pairs = (rel_diff > 0).type(torch.float32)
    neg_pairs = (rel_diff < 0).type(torch.float32)
    Sij = pos_pairs - neg_pairs
    if gain_type == "exp2":
        gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
    elif gain_type == "identity":
        gain_diff = y_tensor - y_tensor.t()
    else:
        raise ValueError("NDCG_gain method not supported yet {}".format(ndcg_gain_in_train))

    rank_order_tensor = torch.tensor(rank_order, dtype=torch.float32, device=device).view(-1, 1)
    decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

    grad_ord1 = sigma * (0.5 * (1 - Sij) - pos_pairs_score_diff) 
    grad_ord2 = sigma*sigma*pos_pairs_score_diff*(1-pos_pairs_score_diff) 
    
    #import ipdb; ipdb.set_trace()

    grad_ord1 = torch.sum(grad_ord1, 1, keepdim=True)
    grad_ord2 = torch.sum(grad_ord2, 1, keepdim=True)


    #print(grad_ord1.shape, y_pred.shape)
    assert grad_ord1.shape == y_pred.shape
    check_grad = torch.sum(grad_ord1, (0, 1)).item()
    check_grad2 = torch.sum(grad_ord2, (0, 1)).item()

    if check_grad == float('inf') or np.isnan(check_grad) or check_grad2 == float('inf') or np.isnan(check_grad2):
       import ipdb; ipdb.set_trace()
        
    return  grad_ord1, grad_ord2


def grad_calc_v2(y_true, y_pred, gain_type, sigma, N, device):
    # Normalize the gradients with NDCG delta adopted from Microsoft paper


    # Only pairs with positive rel values
    # compute the rank order of each document
    rank_df = pd.DataFrame({"y": y_true, "doc": np.arange(y_true.shape[0])})
    rank_df = rank_df.sort_values("y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1

    pos_pairs_score_diff = 1.0/(1.0 + torch.exp(sigma * (y_pred - y_pred.t())))
    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).view(-1, 1)

    if gain_type == "exp2":
        gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
    elif gain_type == "identity":
        gain_diff = y_tensor - y_tensor.t()
    else:
        raise ValueError("NDCG_gain method not supported yet {}".format(ndcg_gain_in_train))

    rank_order_tensor = torch.tensor(rank_order, dtype=torch.float32, device=device).view(-1, 1)
    decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

    delta_ndcg = torch.abs(N * gain_diff * decay_diff)

    grad_ord1 = sigma * (-pos_pairs_score_diff * delta_ndcg)
    grad_ord1 = torch.sum(grad_ord1, 1, keepdim=True)

    grad_ord2 = (sigma*sigma)*pos_pairs_score_diff*(1-pos_pairs_score_diff)*delta_ndcg
    grad_ord2 = torch.sum(grad_ord2, 1, keepdim=True)


    #print(grad_ord1.shape, y_pred.shape)
    assert grad_ord1.shape == y_pred.shape
    check_grad = torch.sum(grad_ord1, (0, 1)).item()
    if check_grad == float('inf') or np.isnan(check_grad):
        import ipdb; ipdb.set_trace()
        
    return  grad_ord1, grad_ord2

def loss_calc_v2(y_true, y_pred, gain_type, sigma, N, device):
    # Normalize the loss with NDCG delta adopted from Microsoft paper

    rank_df = pd.DataFrame({"y": y_true, "doc": np.arange(y_true.shape[0])})
    rank_df = rank_df.sort_values("y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1


    pos_pairs_score_diff = 1.0 + torch.exp(-sigma * (y_pred - y_pred.t()))
    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).view(-1, 1)

    if gain_type == "exp2":
        gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
    elif gain_type == "identity":
        gain_diff = y_tensor - y_tensor.t()
    else:
        raise ValueError("NDCG_gain method not supported yet {}".format(ndcg_gain_in_train))

    rank_order_tensor = torch.tensor(rank_order, dtype=torch.float32, device=device).view(-1, 1)
    decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

    delta_ndcg = torch.abs(N * gain_diff * decay_diff)

    loss = torch.log(pos_pairs_score_diff) * delta_ndcg
    loss = torch.sum(loss, 1, keepdim=True)
        
    return  loss