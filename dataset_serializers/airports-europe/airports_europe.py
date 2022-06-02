
import pickle
import csv
import os
from scipy.sparse import csr_matrix
import numpy as np
import torch

# PolBlogs
# URL: https://netset.telecom-paris.fr/datasets/polblogs.tar.gz

dataset_name = "airports_europe"
dataset_dir = "../../data/datasets/airports_europe"
graph_dir = "../../data/graphs"

dim = 399

# adjacency matrix
adj = csr_matrix((dim, dim), dtype=np.int8).toarray()

with open(os.path.join(dataset_dir, 'europe-airports.edgelist'), 'r') as f:
    lines = f.readlines()
    for row in lines:
        u, v = [int(str(x).strip()) for x in row.split()]
        adj[u][v] = 1
        adj[v][u] = 1

adj = csr_matrix(adj)

with open(os.path.join(graph_dir, '%s_adj.pkl' % dataset_name), 'wb') as f:
    pickle.dump(adj, f)


# feature matrix
feats = csr_matrix((dim, dim), dtype=np.int8).toarray()

for i in range(dim):
    feats[i][i] = 1     # identity matrix (https://github.com/danielzuegner/nettack/issues/3)

feats = csr_matrix(feats)

with open(os.path.join(graph_dir, '%s_features.pkl' % dataset_name), 'wb') as f:
    pickle.dump(feats, f)


# labels
labels = np.zeros(dim, dtype=np.int8)

with open(os.path.join(dataset_dir, 'labels-europe-airports.txt'), 'r') as f:
    lines = f.readlines()
    for row in lines:
        labels[int(row.split()[0])] = int(row.split()[1])

labels = torch.Tensor(labels)
with open(os.path.join(graph_dir, '%s_labels.pkl' % dataset_name), 'wb') as f:
    pickle.dump(labels, f)


# tvt_nids

# Train, Test, Val --> 10%, 30%, 60%
train_num = (dim * 10) // 100
val_num = (dim * 30) // 100
test_num = dim - train_num - val_num

train_nids = np.arange(0, 0+train_num)
val_nids = np.arange(train_num, train_num+val_num)
test_nids = np.arange(train_num+val_num, train_num+val_num+test_num)

tvt_nids = [train_nids, val_nids, test_nids]

with open(os.path.join(graph_dir, '%s_tvt_nids.pkl' % dataset_name), 'wb') as f:
    pickle.dump(tvt_nids, f)
