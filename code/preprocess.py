# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import scipy as sp
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

def read_clean(data):

    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = data.astype(str)
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = {}
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=read_clean(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=read_clean(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], 
                                         exprs_handle["indices"][...],
                                         exprs_handle["indptr"][...]), 
                                        shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

def preprocess_data(filename):
    mat, obs, var, uns = read_data(filename, sparsify=False, skip_exprs=False)
    
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    
    return X, cell_label, obs, var


def normalize_scanpy(adata, highly_genes=2000, filter_min_counts=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    if highly_genes is not None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, 
                                   min_disp=0.5, n_top_genes=highly_genes, subset=True)
    
    sc.pp.scale(adata, max_value=10)
    return adata

def load_and_preprocess_data(data_path, dataset_name):

    X, cell_label, obs, var = preprocess_data(data_path)

    adata = sc.AnnData(X, obs=obs, var=var)
    
    print("normalize...")
    adata = normalize_scanpy(adata)
    
    feature_path = f"../dataset/{dataset_name}/feature.npy"
    np.save(feature_path, adata.X)
    
    label_path = f"../dataset/{dataset_name}/label.npy"
    np.save(label_path, cell_label)
    
    print(f"shape: {adata.X.shape}")
    return adata.X, cell_label

if __name__ == "__main__":
    data_path = "../dataset/Adam/data.h5"
    dataset_name = "Adam"
    X, labels = load_and_preprocess_data(data_path, dataset_name)