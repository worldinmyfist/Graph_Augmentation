import sys
sys.path.insert(0, "./model")

import gc
from turtle import pos
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from GAug_model import GAug_model

import wandb

# VARIABLES
DEF_HIDDEN_SIZE = 128
DEF_EMB_SIZE = 32
DEF_N_LAYERS = 1
DEF_EPOCHS = 200
DEF_SEED = -1
DEF_LR = 1e-2
DEF_WT_DECAY = 5e-4
DEF_DROPOUT = 0.5
DEF_GAE = False
DEF_BETA = 0.5
DEF_TEMP = 0.2
DEF_WARMUP = 3
DEF_GNN_TYPE = 'gcn'
DEF_JKNET = False
DEF_ALPHA = 1
DEF_SAMPLE_TYPE = 'add_sample'
DEF_NORM_TYPE = 'row'


# BASE CLASS
class GAugBase(object):
    def __init__(self):
        self.features = None

    def preprocess(self):
        pass

    def fit(self):
        return None

    def normalize(self, norm_type):
        ''' Normalizes feature matrix '''
        if self.features is not None:
            if norm_type == 'row':
                self.features = F.normalize(self.features, p=1, dim=1)
            elif norm_type == 'col':
                self.features = self.col_normalization(self.features)

    @staticmethod
    def sample_edges(adj_matrix, edge_frac):
        ''' Samples edges and no-edges '''
        pos_edges = None
        neg_edges = None
        n_edges_sample = None

        adj_matrix = sp.csr_matrix(adj_matrix)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)

        # sample negative edges
        neg_edges = []
        added_edges = set()

        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if (i == j) or (adj_matrix[i, j] > 0) or ((i, j) in added_edges):
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)

        # sample positive edges
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]

        return pos_edges, neg_edges, n_edges_sample

    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        return eval_edge_pred(adj_pred, val_edges, edge_labels)

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        return eval_node_cls(nc_logits, labels)

    @staticmethod
    def get_lr_schedule(n_epochs, lr, warmup):
        return get_lr_schedule_by_sigmoid(n_epochs, lr, warmup)

    @staticmethod
    def col_normalization(features):
        return col_normalization(features)


class GAug(GAugBase):
    def __init__(
            self,
            adj_matrix,
            features,
            labels,
            tvt_nids,
            cuda=-1,
            hidden_size=DEF_HIDDEN_SIZE,
            emb_size=DEF_EMB_SIZE,
            n_layers=DEF_N_LAYERS,
            epochs=DEF_EPOCHS,
            seed=DEF_SEED,
            lr=DEF_LR,
            weight_decay=DEF_WT_DECAY,
            dropout=DEF_DROPOUT,
            gae=DEF_GAE,
            beta=DEF_BETA,
            temperature=DEF_TEMP,
            warmup=DEF_WARMUP,
            gnnlayer_type=DEF_GNN_TYPE,
            jknet=DEF_JKNET,
            alpha=DEF_ALPHA,
            sample_type=DEF_SAMPLE_TYPE,
            feat_norm=DEF_NORM_TYPE,
            verbose=True,
    ):
        super(GAug, self).__init__()

        # CUDA check
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda >= 0 else 'cpu')

        self.n_epochs = epochs
        self.learning_rate = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.gae = gae
        self.warmup = warmup
        self.feat_norm = feat_norm
        self.verbose = verbose

        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.preprocess(adj_matrix, features, labels, tvt_nids, gnnlayer_type)

        self.model = GAug_model(self.features.size(1),
                                hidden_size,
                                emb_size,
                                self.out_size,
                                n_layers,
                                F.relu,
                                dropout,
                                self.device,
                                gnnlayer_type,
                                temperature=temperature,
                                gae=gae,
                                jknet=jknet,
                                alpha=alpha,
                                sample_type=sample_type)

    def fit(self, pretrain_ep=200, pretrain_nc=20, wandb_log=False):
        '''
        Trains the model and returns test accuracy.
        '''

        model = self.model.to(self.device)
        adj = self.adj.to(self.device)
        adj_norm = self.adj_norm.to(self.device)
        adj_orig = self.adj_orig.to(self.device)
        labels = self.labels.to(self.device)
        features = self.features.to(self.device)

        adj_dim = self.adj_orig.shape[0]
        adj_sum = self.adj_orig.sum()

        norm_w = float(adj_dim ** 2) / (2 * (adj_dim ** 2 - adj_sum))
        pos_weight = torch.FloatTensor(
            [float(adj_dim ** 2 - adj_sum) / adj_sum]).to(self.device)

        if pretrain_ep:
            self.pretrain_ep_net(model, adj_norm, features,
                                 adj_orig, norm_w, pos_weight, pretrain_ep)

        if pretrain_nc:
            self.pretrain_nc_net(model, adj, features, labels, pretrain_nc)

        optimizers = MultipleOptimizer(
            torch.optim.Adam(model.ep_net.parameters(), lr=self.learning_rate),
            torch.optim.Adam(model.nc_net.parameters(), lr=self.learning_rate,
                             weight_decay=self.weight_decay)
        )

        if self.warmup:
            ep_lr_schedule = self.get_lr_schedule(
                self.n_epochs, self.learning_rate, self.warmup)

        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.
        cnt_patience = 0

        #########
        # TRAIN #
        #########

        for epoch in range(self.n_epochs):
            if self.warmup:
                optimizers.update_lr(0, ep_lr_schedule[epoch])

            model.train()

            nc_logits, adj_logits = model(adj_norm, adj_orig, features)

            # node classification loss
            nc_loss = nc_criterion(
                nc_logits[self.train_nid], labels[self.train_nid])

            # edge prediction loss
            ep_loss = norm_w * \
                F.binary_cross_entropy_with_logits(
                    adj_logits, adj_orig, pos_weight=pos_weight)

            # net loss
            loss = nc_loss + self.beta * ep_loss

            optimizers.zero_grad()
            loss.backward()
            optimizers.step()

            # validate
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(
                nc_logits_eval[self.val_nid], labels[self.val_nid])

            # test
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            test_acc = self.eval_node_cls(
                nc_logits_eval[self.test_nid], labels[self.test_nid])

            if self.verbose:
                print('Epoch [{:3}/{}] --- EP loss: {:.2f}, NC loss: {:.2f}, Val Acc: {:.2f}, Test Acc: {:.2f}'.format(
                    epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), val_acc, test_acc))

            if wandb_log:
                wandb_log_dict = {
                                     f'EP_Loss':ep_loss.item(),
                                     f'NC_Loss':nc_loss.item(),
                                     f'Val_Acc':val_acc,
                                     f'Test_Acc':test_acc,
                                 }
                wandb.log(wandb_log_dict)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                cnt_patience = 0
            else:
                cnt_patience += 1
                if cnt_patience == 100:
                    break
            ''''
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            test_acc = self.eval_node_cls(
                nc_logits_eval[self.test_nid], labels[self.test_nid])
            '''

        print("Test Accuracy: {:.3f}".format(test_acc))

        # release RAM and GPU memory
        del adj, features, labels, adj_orig
        torch.cuda.empty_cache()
        gc.collect()

        return test_acc

    def preprocess(self, adj_matrix, features, labels, tvt_nids, gnnlayer_type):
        '''
        Loads and preprocesses data.
        '''

        # features (torch.FloatTensor)
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        self.normalize(self.feat_norm)

        # original adj_matrix for training vgae (torch.FloatTensor)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        self.adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()

        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)

        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        if gnnlayer_type == 'gcn':
            self.adj = scipysp_to_pytorchsp(adj_norm)
        elif gnnlayer_type == 'gsage':
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix)
            adj_matrix_noselfloop = sp.coo_matrix(
                adj_matrix_noselfloop / adj_matrix_noselfloop.sum(1))
            self.adj = scipysp_to_pytorchsp(adj_matrix_noselfloop)
        elif gnnlayer_type == 'gat':
            self.adj = torch.FloatTensor(adj_matrix.todense())

        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels

        self.train_nid, self.val_nid, self.test_nid = tvt_nids[:3]

        # number of classes
        if len(self.labels.size()) == 1:
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)

        if self.labels.size(0) > 5000:
            # sample 1% for large graph
            edge_frac = 0.01
        else:
            # sample 10% for small graph
            edge_frac = 0.1

        pos_edges, neg_edges, n_edges_sample = self.sample_edges(
            adj_matrix, edge_frac)

        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)

    def pretrain_ep_net(self, model, adj, features, adj_orig, norm_w, pos_weight, n_epochs):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(model.ep_net.parameters(),
                                     lr=self.learning_rate)
        model.train()

        for epoch in range(n_epochs):
            adj_logits = model.ep_net(adj, features)
            loss = norm_w * \
                F.binary_cross_entropy_with_logits(
                    adj_logits, adj_orig, pos_weight=pos_weight)

            if not self.gae:
                mu = model.ep_net.mean
                lgstd = model.ep_net.logstd
                kl_divergence = 0.5 / \
                    adj_logits.size(0) * (1 + 2*lgstd - mu **
                                          2 - torch.exp(2*lgstd)).sum(1).mean()
                loss -= kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(
                adj_pred, self.val_edges, self.edge_labels)

            if self.verbose:
                print('Pretraining EPNet (Epoch [{:3}/{}]) --- Loss: {:.3f}, AUC: {:.2f}, AP Score: {:.2f}'.format(
                    epoch+1, n_epochs, loss.item(), ep_auc, ep_ap))

    def pretrain_nc_net(self, model, adj, features, labels, n_epochs):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(model.nc_net.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.
        for epoch in range(n_epochs):
            model.train()
            nc_logits = model.nc_net(adj, features)

            # losses
            loss = nc_criterion(
                nc_logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()

            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)

            val_acc = self.eval_node_cls(
                nc_logits_eval[self.val_nid], labels[self.val_nid])

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if self.verbose:
                print('Pretraining NCNet (Epoch [{:3}/{}]) --- Loss: {:.3f}, Val Acc: {:.2f}'.format(
                    epoch+1, n_epochs, loss.item(), val_acc))
