import os
import copy
import json
import wandb
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from models.GCN import GCN
from models.GAT import GAT
from models.GSAGE import GraphSAGE
from models.JKNet import JKNet

os.environ['CUDA_VISIBLE_DEVICES'],device =['0',0] if torch.cuda.is_available() else ['',None]

def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def sample_graph_det(adj, A, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj)
    orig_upper = sp.triu(adj, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        A_probs = np.array(A)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    new_adj = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj.shape)
    new_adj = new_adj + new_adj.T
    return new_adj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--wandb', dest='wandb_log', type=bool, default=False)
    parser.add_argument('--wandb-project', dest='wandb_project', type=str, default='tidl-project-GAugM')
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--parampath', type=str, default='../best_parameters.json')
    args = parser.parse_args()


    tvt_nids = load_pickle(os.path.join(args.datapath,'graphs',f"{args.dataset}_tvt_nids.pkl"))
    adj = load_pickle(os.path.join(args.datapath,'graphs',f"{args.dataset}_adj.pkl"))
    feats = load_pickle(os.path.join(args.datapath,'graphs',f"{args.dataset}_features.pkl"))
    labels = load_pickle(os.path.join(args.datapath,'graphs',f"{args.dataset}_labels.pkl"))
    feats = torch.FloatTensor(feats.toarray()) if sp.issparse(feats) else feats

    params = json.load(open(args.parampath, 'r'))['GAugM'][args.dataset][args.gnn]
    A = load_pickle(os.path.join(args.datapath,"edge_probabilities",f"{args.dataset}_graph_2_logits.pkl"))
    new_adj = sample_graph_det(adj, A, params['rm_pct'], params['add_pct'])

    hidden_size=128
    n_layers=1
    epochs=200
    lr=1e-2
    seed=-1

    if args.wandb_log:
        wandb.init(project=args.wandb_project,config={
                                                            f'model_type':'GAugO',
                                                            f'dataset':args.dataset,
                                                            f'gnn':args.gnn,
                                                            f'hidden_size':hidden_size,
                                                            f'n_layers':n_layers,
                                                            f'lr':lr,
                                                            f'epochs':epochs,
                                                            f'seed':seed,
                                                        })

    try:
        GNN = {'gcn':GCN, 'gsage':GraphSAGE, 'gat':GAT, 'jknet':JKNet}[args.gnn]
    except:
        raise Exception('Invalid gnn type specified')

    model = GNN(new_adj, new_adj, feats, labels, tvt_nids, 
                                    print_progress=True, 
                                    cuda=device, 
                                    hidden_size=128,
                                    n_layers=1,
                                    lr=1e-2,
                                    seed=-1,
                                    epochs=200)
    accuracy = model.fit()[0]
