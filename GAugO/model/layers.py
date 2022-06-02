import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x


class SAGELayer(nn.Module):
    """ one layer of GraphSAGE with gcn aggregator """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(SAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(input_dim, output_dim, bias=False)
        # self.linear_self = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        # using GCN aggregator
        if self.dropout:
            h = self.dropout(h)
        x = adj @ h
        x = self.linear_neigh(x)
        # x_neigh = self.linear_neigh(x)
        # x_self = self.linear_self(h)
        # x = x_neigh + x_self
        if self.activation:
            x = self.activation(x)
        # x = F.normalize(x, dim=1, p=2)
        return x


class GATLayer(nn.Module):
    """ one layer of GAT """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        self.n_heads = n_heads
        self.attn_l = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_r = nn.Linear(output_dim, self.n_heads, bias=False)
        self.attn_drop = nn.Dropout(p=0.6)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W # torch.Size([2708, 128])
        # calculate attentions, both el and er are n_nodes by n_heads
        el = self.attn_l(x)
        er = self.attn_r(x) # torch.Size([2708, 8])
        if isinstance(adj, torch.sparse.FloatTensor):
            nz_indices = adj._indices()
        else:
            nz_indices = adj.nonzero().T
        attn = el[nz_indices[0]] + er[nz_indices[1]] # torch.Size([13264, 8])
        attn = F.leaky_relu(attn, negative_slope=0.2).squeeze()
        # reconstruct adj with attentions, exp for softmax next
        attn = torch.exp(attn) # torch.Size([13264, 8]) NOTE: torch.Size([13264]) when n_heads=1
        if self.n_heads == 1:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1)), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn)
        else:
            adj_attn = torch.zeros(size=(adj.size(0), adj.size(1), self.n_heads), device=adj.device)
            adj_attn.index_put_((nz_indices[0], nz_indices[1]), attn) # torch.Size([2708, 2708, 8])
            adj_attn.transpose_(1, 2) # torch.Size([2708, 8, 2708])
        # edge softmax (only softmax with non-zero entries)
        adj_attn = F.normalize(adj_attn, p=1, dim=-1)
        adj_attn = self.attn_drop(adj_attn)
        # message passing
        x = adj_attn @ x # torch.Size([2708, 8, 128])
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        if self.n_heads > 1:
            x = x.flatten(start_dim=1)
        return x # torch.Size([2708, 1024])
