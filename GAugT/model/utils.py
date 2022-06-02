from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import numpy as np
import scipy.sparse as sp


class MultipleOptimizer():
    """ a class that wraps multiple optimizers """

    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class CeilNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()

    @staticmethod
    def backward(ctx, g):
        return g


def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx


def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges.T]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score


def eval_node_cls(nc_logits, labels):
    """ evaluate node classification results """
    if len(labels.size()) == 2:
        preds = torch.round(torch.sigmoid(nc_logits))
        tp = len(torch.nonzero(preds * labels))
        tn = len(torch.nonzero((1-preds) * (1-labels)))
        fp = len(torch.nonzero(preds * (1-labels)))
        fn = len(torch.nonzero((1-preds) * labels))
        pre, rec, f1 = 0., 0., 0.
        if tp+fp > 0:
            pre = tp / (tp + fp)
        if tp+fn > 0:
            rec = tp / (tp + fn)
        if pre+rec > 0:
            fmeasure = (2 * pre * rec) / (pre + rec)
    else:
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        fmeasure = correct.item() / len(labels)
    return fmeasure


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    """ schedule the learning rate with the sigmoid function.
    The learning rate will start with near zero and end with near lr """
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    # range the factors to [0, 1]
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule


def col_normalization(features):
    """ column normalization for feature matrix """
    features = features.numpy()
    m = features.mean(axis=0)
    s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
    features -= m
    features /= s
    return torch.FloatTensor(features)
