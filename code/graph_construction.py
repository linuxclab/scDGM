# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def construct_graph(features, labels, method='cos', topk=10, gamma=1):
    
    if method == "rbf":
        dist = -(0.5 * gamma**2) * pairwise_distances(features, metric='euclidean') ** 2
        dist = np.exp(dist)
    elif method == 'heat':
        dist = -0.5 * pairwise_distances(features, metric='manhattan') ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features_binary = (features > 0).astype(float)
        dist = np.dot(features_binary, features_binary.T)
    elif method == 'corr':
        dist = -0.5 * pairwise_distances(features, metric='correlation') ** 2
        dist = np.exp(dist)
    else:
        dist = pairwise_distances(features, metric='euclidean')
        dist = 1 / (1 + dist) 
    
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    
    adj = np.zeros_like(dist)
    for i, neighbors in enumerate(inds):
        for j in neighbors:
            if j != i:  
                adj[i, j] = 1
    
    return adj

def normalize_adj(adj, self_loop=True, symmetry=True):

    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj
    
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    
    return norm_adj

def KNN_graph(aff_matrix, nr_of_knn):
    thres = np.sort(aff_matrix)[:, -nr_of_knn]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix

def build_graph_from_features(features, labels, dataset_name, root_path, topk=6, method='rbf'):
    adj = construct_graph(features, labels, method=method, topk=topk)
    
    norm_adj = normalize_adj(adj)
    
    knn_adj = KNN_graph(norm_adj, topk)
    
    edge_list = []
    for i in range(knn_adj.shape[0]):
        for j in range(knn_adj.shape[1]):
            if knn_adj[i, j] > 0 and i != j:
                edge_list.append([i, j])
    
    edge_list_path = f"../dataset/{dataset_name}/edge_list.txt"
    with open(edge_list_path, 'w') as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    print(f"edge: {len(edge_list)}")
    
    return knn_adj, edge_list

