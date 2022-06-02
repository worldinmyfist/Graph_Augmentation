import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from GCN import *
from GNN import *
from GNN_JK import *
from VGAE import *
from utils import *

class GAug_model(nn.Module):
    def __init__(self, dim_feats, dim_h, dim_z, n_classes, n_layers, activation, dropout, device, gnnlayer_type, temperature=1, gae=False, jknet=False, alpha=1, sample_type='add_sample'):
        super(GAug_model, self).__init__()

        self.device = device
        self.temperature = temperature
        self.gnnlayer_type = gnnlayer_type
        self.alpha = alpha
        self.sample_type = sample_type

        # edge prediction network
        self.ep_net = VGAE(dim_feats, dim_h, dim_z, activation, gae=gae)

        if jknet == False:
            self.nc_net = GNN(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)
        else:
            self.nc_net = GNN_JK(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)

    def normalize_adjM(self, adj):
        if self.gnnlayer_type == 'gcn':
            adj.fill_diagonal_(1)
            # A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj
    
    def sample_adjM(self, adj_logits, adj_orig, alpha, change_frac):
        if self.sample_type == 'rand':
            #adj_new = self.sample_adj_random(adj_logits)
            adj_rand = torch.rand(adj_logits.size())
            adj_rand = adj_rand.triu(1)
            adj_rand = torch.round(adj_rand)
            adj_rand = adj_rand + adj_rand.T
            return adj_rand

        elif self.sample_type == 'edge':
            #adj_new = self.sample_adj_edge(adj_logits, adj_orig, self.alpha)
            adj = adj_orig.to_dense() if adj_orig.is_sparse else adj_orig
            n_edges = adj.nonzero().size(0)
            n_change = int(n_edges * change_frac / 2)
            # take only the upper triangle
            edge_probs = adj_logits.triu(1)
            edge_probs = edge_probs - torch.min(edge_probs)
            edge_probs = edge_probs / torch.max(edge_probs)
            adj_inverse = 1 - adj
            # get edges to be removed
            mask_rm = edge_probs * adj
            nz_mask_rm = mask_rm[mask_rm>0]
            if len(nz_mask_rm) > 0:
                n_rm = len(nz_mask_rm) if len(nz_mask_rm) < n_change else n_change
                thresh_rm = torch.topk(mask_rm[mask_rm>0], n_rm, largest=False)[0][-1]
                mask_rm[mask_rm > thresh_rm] = 0
                mask_rm = CeilNoGradient.apply(mask_rm)
                mask_rm = mask_rm + mask_rm.T
            # remove edges
            adj_new = adj - mask_rm
            # get edges to be added
            mask_add = edge_probs * adj_inverse
            nz_mask_add = mask_add[mask_add>0]
            if len(nz_mask_add) > 0:
                n_add = len(nz_mask_add) if len(nz_mask_add) < n_change else n_change
                thresh_add = torch.topk(mask_add[mask_add>0], n_add, largest=True)[0][-1]
                mask_add[mask_add < thresh_add] = 0
                mask_add = CeilNoGradient.apply(mask_add)
                mask_add = mask_add + mask_add.T
            # add edges
            adj_new = adj_new + mask_add
            return adj_new

        elif self.sample_type == 'add_round':
            #adj_new = self.sample_adj_add_round(adj_logits, adj_orig, self.alpha)
            edge_probs = adj_logits / torch.max(adj_logits)
            edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
            # sampling
            adj_sampled = RoundNoGradient.apply(edge_probs)
            # making adj_sampled symmetric
            adj_sampled = adj_sampled.triu(1)
            adj_sampled = adj_sampled + adj_sampled.T
            return adj_sampled

        elif self.sample_type == 'add_sample':
            if self.alpha != 1:
                #adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)
                edge_probs = adj_logits / torch.max(adj_logits)
                edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
                # sampling
                adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
                # making adj_sampled symmetric
                adj_sampled = adj_sampled.triu(1)
                adj_sampled = adj_sampled + adj_sampled.T
                return adj_sampled

            else:
                #adj_new = self.sample_adj(adj_logits)
                edge_probs = adj_logits / torch.max(adj_logits)
                # sampling
                adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
                # making adj_sampled symmetric
                adj_sampled = adj_sampled.triu(1)
                adj_sampled = adj_sampled + adj_sampled.T
                return adj_sampled

    def forward(self, adj, adj_orig, features):

        #layer 1
        adj_logits1 = self.ep_net(adj, features)
        adj_new1 = self.sample_adjM(adj_logits1, adj_orig, self.alpha, self.alpha)
        adj_new_normed1 = self.normalize_adjM(adj_new1)

        #layer 2
        adj_logits2 = self.ep_net(adj_logits1, features)
        adj_new2 = self.sample_adjM(adj_logits2, adj_new_normed1, self.alpha, self.alpha)
        adj_new_normed2 = self.normalize_adjM(adj_new2)

        nc_logits = self.nc_net(adj_new_normed2, features)
        
        return nc_logits, adj_logits2
