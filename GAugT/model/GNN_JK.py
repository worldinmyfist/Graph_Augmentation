import torch
import torch.nn as nn
from layers import *

class GNN_JK(nn.Module):
    """ GNN with JK design as a node classification model """
    def __init__(self, dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type='gcn'):
        super(GNN_JK, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'gat':
            gnnlayer = GATLayer
            heads = [8] * n_layers + [1]
            dim_h = int(dim_h / 8)
            activation = F.elu
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(dim_h*heads[i], dim_h, heads[i+1], activation, dropout))
        # output layer
        self.layer_output = nn.Linear(dim_h*n_layers*heads[-2], n_classes)

    def forward(self, adj, features):
        h = features
        hs = []
        for layer in self.layers:
            h = layer(adj, h)
            hs.append(h)
        # JK-concat design
        h = torch.cat(hs, 1)
        h = self.layer_output(h)
        return h
