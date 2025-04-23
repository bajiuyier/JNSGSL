import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges.T]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score

class MultipleOptimizer():
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule

def normalize(mx, style='symmetric', add_loop=True, p=None):
    if style == 'row':
        if mx.is_sparse:
            return row_normalize_sp(mx)
        else:
            return row_nomalize(mx)
    elif style == 'symmetric':
        if mx.is_sparse:
            return normalize_sp_tensor_tractable(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)
    elif style == 'softmax':
        if mx.is_sparse:
            return torch.sparse.softmax(mx, dim=-1)
        else:
            return F.softmax(mx, dim=-1)
    elif style == 'row-norm':
        assert p is not None
        if mx.is_sparse:
            # TODO
            pass
        else:
            return F.normalize(mx, dim=-1, p=p)
    else:
        raise KeyError("The normalize style is not provided.")

def row_nomalize(mx):
    """Row-normalize dense tensor."""
    r_sum = mx.sum(1)
    r_inv = r_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx

def row_normalize_sp(mx):
    """Row-normalize sparse tensor."""
    adj = mx.coalesce()
    inv_sqrt_degree = 1. / (torch.sparse.sum(mx, dim=1).values() + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def normalize_sp_tensor_tractable(adj, add_loop=True):
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def normalize_tensor(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A

def normalize_sp_tensor(adj, add_loop=True):
    device = adj.device
    adj = sparse_tensor_to_scipy_sparse(adj)
    adj = normalize_sp_matrix(adj, add_loop)
    adj = scipy_sparse_to_sparse_tensor(adj).to(device)
    return adj

def normalize_sp_matrix(adj, add_loop=True):
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new

class Recorder:
    def __init__(self, patience=100, criterion=None):
        self.patience = patience
        self.criterion = criterion
        self.best_loss = 1e8
        self.best_metric = -1
        self.wait = 0

    def add(self, loss_val, metric_val):
        flag = False
        if self.criterion is None:
            flag = True
        elif self.criterion == 'loss':
            flag = loss_val < self.best_loss
        elif self.criterion == 'metric':
            flag = metric_val > self.best_metric
        elif self.criterion == 'either':
            flag = loss_val < self.best_loss or metric_val > self.best_metric
        elif self.criterion == 'both':
            flag = loss_val < self.best_loss and metric_val > self.best_metric
        else:
            raise NotImplementedError

        if flag:
            self.best_metric = metric_val
            self.best_loss = loss_val
            self.wait = 0
        else:
            self.wait += 1

        flag_earlystop = self.patience and self.wait >= self.patience

        return flag, flag_earlystop

class InnerProduct(nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, x, y=None, non_negative=False):
        if y is None:
            y = x
        adj = torch.matmul(x, y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj

class NonLinear(nn.Module):
    def __init__(self, non_linearity, i=None):
        super(NonLinear, self).__init__()
        self.non_linearity = non_linearity
        self.i = i

    def forward(self, adj):
        return apply_non_linearity(adj, self.non_linearity, self.i)

def convert_to_sparse_coo(sparse_tensor):
    row, col, value = sparse_tensor.coo()
    coo_indices = torch.stack([row, col], dim=0)
    sparse_coo_tensor = torch.sparse_coo_tensor(coo_indices, value, sparse_tensor.sizes())
    return sparse_coo_tensor

def accuracy(labels, logits):
    return np.sum(logits.argmax(1) == labels) / len(labels)

def apply_non_linearity(adj, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(adj * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(adj)
    elif non_linearity == 'none':
        return adj
    else:
        raise KeyError('Unsupported non-linearity.')

def scipy_sparse_to_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_tensor_to_scipy_sparse(sparse_tensor):
    sparse_tensor = sparse_tensor.cpu()
    row = sparse_tensor.coalesce().indices()[0].numpy()
    col = sparse_tensor.coalesce().indices()[1].numpy()
    values = sparse_tensor.coalesce().values().numpy()
    return sp.coo_matrix((values, (row, col)), shape=sparse_tensor.shape)

def get_edge_homophily(label, adj):
    # Convert adj to dense matrix
    adj_dense = adj.cpu().to_dense()
    # Remove self-loops: set diagonal to 0
    adj_dense.fill_diagonal_(0)
    # Count total edge weights excluding self-loops
    num_edge = adj_dense.sum()
    cnt = 0
    # Count homophilic edges
    for i, j in adj_dense.nonzero():
        if label[i] == label[j]:
            cnt += adj_dense[i, j]
    return cnt / num_edge
