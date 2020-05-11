import numpy as np
import pandas as pd
import torch

def grad_calc(y_true, y_pred, gain_type, sigma, device):

    # compute the rank order of each document
    rank_df = pd.DataFrame({"y": y_true, "doc": np.arange(y_true.shape[0])})
    rank_df = rank_df.sort_values("y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1


    # In my case y_pred = out
    pos_pairs_score_diff = 1.0/(1.0 + torch.exp(sigma * (y_pred - y_pred.t())))

    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).view(-1, 1)
    rel_diff = y_tensor - y_tensor.t()
    pos_pairs = (rel_diff > 0).type(precision)
    neg_pairs = (rel_diff < 0).type(precision)
    Sij = pos_pairs - neg_pairs
    if gain_type == "exp2":
        gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
    elif gain_type == "identity":
        gain_diff = y_tensor - y_tensor.t()
    else:
        raise ValueError("NDCG_gain method not supported yet {}".format(ndcg_gain_in_train))

    rank_order_tensor = torch.tensor(rank_order, dtype=torch.float32, device=device).view(-1, 1)
    decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

    delta_ndcg = torch.abs(N * gain_diff * decay_diff)

    grad_ord1 = -sigma * (0.5 * (1 - Sij) - pos_pairs_score_diff) * delta_ndcg
    grad_ord1 = torch.sum(grad_ord1, 1, keepdim=True)

    grad_ord2 = sigma*sigma*pos_pairs_score_diff*(1-pos_pairs_score_diff)*delta_ndcg
    grad_ord2 = torch.sum(grad_ord2s, 1, keepdim=True)

    assert grad_ord1.shape == y_pred.shape
    check_grad = torch.sum(grad_ord1, (0, 1)).item()
    if check_grad == float('inf') or np.isnan(check_grad):
        import ipdb; ipdb.set_trace()
        
    return  grad_ord1, grad_ord2