# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from scipy.optimize import linear_sum_assignment
import random

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_graph_data(inputf, labelf, featuref):

    edgelist = []
    with open(inputf, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) >= 2:
                src, dst = int(tokens[0]), int(tokens[1])
                edgelist.append([src, dst])
    
    labels_array = np.load(labelf)
    labels = {i: int(labels_array[i]) for i in range(len(labels_array))}
    
    features = np.load(featuref)
    
    if len(edgelist) > 0:
        edgelist = np.array(edgelist).T
        edge_index = torch.tensor(edgelist, dtype=torch.long)
    else:
        n_nodes = features.shape[0]
        edge_index = torch.tensor([list(range(n_nodes)), list(range(n_nodes))], dtype=torch.long)
    
    x = torch.FloatTensor(features)
    y = torch.tensor([labels[i] for i in range(len(labels))], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    if edge_index.shape[1] > 0:
        data.edge_index, data.edge_weight = gcn_norm(
            data.edge_index, None, data.num_nodes, add_self_loops=True, dtype=data.x.dtype)
    else:
        data.edge_weight = None
    
    return data

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size, row_ind

class load_data(Dataset):
    def __init__(self, dataset, label):
        self.x = dataset
        self.y = label
        # data_mat.close()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
            torch.from_numpy(np.array(self.y[idx])), \
            torch.from_numpy(np.array(idx))