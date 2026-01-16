# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import Sequential, TAGConv
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import fowlkes_mallows_score
from utils import read_graph_data
from graph_construction import *
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cpu")

def dense_hoscpool(x, adj, s, mu=1, alpha=0.01, new_ortho=True, mask=None):

    EPS = 1e-15
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask
    # Output adjacency and feature matrices
    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Motif adj matrix - not sym. normalised second 2
    motif_adj = torch.mul(torch.matmul(adj, adj), adj)
    #motif_adj = torch.matmul(adj, adj)
    motif_out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), motif_adj), s)
    
    mincut_loss = ho_mincut_loss = 0
    # 1st order MinCUT loss
    if alpha < 1:
        diag_SAS = torch.einsum("ijj->ij", out_adj.clone())
        d_flat = torch.einsum("ijk->ij", adj.clone())
        d = _rank3_diag(d_flat)
        sds = torch.matmul(torch.matmul(s.transpose(1, 2), d), s)
        diag_SDS = torch.einsum("ijk->ij", sds) + EPS
        mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        mincut_loss = 1 / k * torch.mean(mincut_loss)

    # Higher order cut  second 2
    if alpha > 0:
        diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        d_flat = torch.einsum("ijk->ij", motif_adj)
        d = _rank3_diag(d_flat)
        diag_SDS = (torch.einsum(
            "ijk->ij", torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) +
                    1e-15)
        ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        ho_mincut_loss = 1 / k * torch.mean(ho_mincut_loss)

    # Combine ho and fo mincut loss.
    # We do not learn these coefficients yet
    hosc_loss = (1 - alpha) * mincut_loss + alpha * ho_mincut_loss

    # Orthogonality loss
    if mu == 0:
        ortho_loss = torch.tensor(0)
    else:
        if new_ortho:
            if s.shape[0] == 1:
                ortho_loss = ((-torch.sum(torch.norm(s, p="fro", dim=-2)) /
                               (num_nodes**0.5)) + k**0.5) / (k**0.5 - 1)
            elif mask != None:
                ortho_loss = sum([((-torch.sum(
                    torch.norm(
                        s[i][:mask[i].nonzero().shape[0]],
                        p="fro",
                        dim=-2,
                    )) / (mask[i].nonzero().shape[0]**0.5) + k**0.5) /
                                   (k**0.5 - 1)) for i in range(batch_size)
                                  ]) / float(batch_size)
            else:
                ortho_loss = sum(
                    [((-torch.sum(torch.norm(s[i], p="fro", dim=-2)) /
                       (num_nodes**0.5) + k**0.5) / (k**0.5 - 1))
                     for i in range(batch_size)]) / float(batch_size)
        else:
            # Orthogonality regularization.
            ss = torch.matmul(s.transpose(1, 2), s)
            i_s = torch.eye(k).type_as(ss)
            ortho_loss = torch.norm(
                ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
                i_s / torch.norm(i_s),
                dim=(-1, -2),
            )
            ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d + EPS)[:, None]
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, hosc_loss, mu * ortho_loss

def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


class GraphClusterNet(torch.nn.Module):
    def __init__(self, mp_units, mp_act, in_channels, n_clusters, mlp_units=[], mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            (TAGConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((TAGConv(mp_units[i], mp_units[i+1]), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
    
    def forward(self, x, edge_index, edge_weight, alpha=0.1):
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight) 
        # Cluster assignments (logits)
        s = self.mlp(x) 
        adj_r = self.dot_product_decode(x)
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = dense_hoscpool(x, adj, s, alpha=alpha)
        adj = adj.squeeze(0)
        loss_recon = F.binary_cross_entropy(adj_r.float(), adj.float())
        o_loss = o_loss + 0.1 * loss_recon
        return F.log_softmax(s, dim=-1), s, mc_loss, o_loss
    
    def dot_product_decode(self, Z):
        A_recon = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_recon

def graph_clustering(data, dataset_name, root_path, alpha=0.1):

    data = data.to(device)
    cluster_num = max(data.y) + 1
    
    model = GraphClusterNet([64, 64], "ELU", data.num_features, cluster_num, [32]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    best_ari = 0
    best_nmi = 0
    patience = 100
    best_epoch = 0
    
    for epoch in range(1, 10000):
        model.train()
        optimizer.zero_grad()
        _, _, mc_loss, o_loss = model(data.x, data.edge_index, data.edge_weight, alpha=alpha)
        loss = mc_loss + o_loss
        
        loss.backward()
        optimizer.step()
        
    
        if epoch % 10 == 0:
            model.eval()
            clust, s, _, _ = model(data.x, data.edge_index, data.edge_weight, alpha=alpha)
            cluster_pred = clust.max(1)[1].cpu()
            true_labels = data.y.cpu().numpy()
            
            ari = ari_score(true_labels, cluster_pred)
            nmi = nmi_score(true_labels, cluster_pred, average_method='arithmetic')
            
            if ari > best_ari:
                best_ari = ari
                best_nmi = nmi
                best_epoch = epoch
                patience = 50
                print(f"Epoch {epoch}: ARI: {ari:.4f}, NMI: {nmi:.4f}")
            else:
                patience -= 1
            
            if patience <= 0:
                print(f"early stop  epoch {epoch}")
                break
    
    print(f"best ARI: {best_ari:.4f}, sestNMI: {best_nmi:.4f}")
    return best_ari, best_nmi


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


if __name__ == "__main__":
    data_name  ='Adam'
       
    root_path = "../"
    all_results = []
    
    dataset_results = []
    feature_path = f"{root_path}/dataset/{data_name}/features.npy"
    reduced_features = np.load(feature_path)
    
    label_path = f"{root_path}dataset/{data_name}/label.npy"
    labels = np.load(label_path)
    
    #adj, edge_list = build_graph_from_features(reduced_features, labels, data_name, root_path)

    edge_list_path = f"{root_path}dataset/{data_name}/edge_list.txt"
    #label_path = f"{root_path}dataset/{data_name}/label.npy"
    feature_path = f"{root_path}dataset/{data_name}/features.npy"
    
    graph_data = read_graph_data(edge_list_path, label_path, feature_path)
    
    ari, nmi = graph_clustering(graph_data, data_name, root_path)
    result = {
        'dataset': data_name,
        'ari': ari,
        'nmi': nmi
    }
    dataset_results.append(result)
    all_results.append(result)
    
    print(f" ARI={ari:.4f}, NMI={nmi:.4f}")
    valid_results = [r for r in dataset_results if r['ari'] > 0]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['ari'])
        print(f"{data_name} best:  ARI={best_result['ari']:.4f}, NMI={best_result['nmi']:.4f}")
    
    
        

