# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Linear
import numpy as np
from sklearn.cluster import KMeans
from utils import cluster_acc

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)
        
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h1)
        
        return x_bar, z

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def pretrain_ae(model, dataset, y, dataset_name, device, n_z=10):
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    best_acc = 0
    for epoch in range(200):
        model.train()
        total_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
       
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                x = torch.Tensor(dataset.x).to(device).float()
                x_bar, z = model(x)
                loss = F.mse_loss(x_bar, x)
                
                kmeans = KMeans(n_clusters=len(np.unique(y)), algorithm="elkan").fit(z.data.cpu().numpy())
                acc = cluster_acc(y, kmeans.labels_)[0]
                
                if acc > best_acc:
                    best_acc = acc
                    
                    model_path = f"../dataset/{dataset_name}/ae_model.pkl"
                    torch.save(model.state_dict(), model_path)
                    
                    feature_path = f"../dataset/{dataset_name}/features.npy"
                    np.save(feature_path, z.cpu().numpy())
                    

def autoencoder_dim_reduction(X, dataset_name, device, n_z=10):
    print("Reduction...")
    
    dataset = LoadDataset(X)
    n_input = X.shape[1]
    
    model = AE(
        n_enc_1=2000, n_enc_2=2000, n_enc_3=500,
        n_dec_1=500, n_dec_2=2000, n_dec_3=2000,
        n_input=n_input, n_z=n_z
    ).to(device)
    
    label_path = f"../dataset/{dataset_name}/label.npy"
    y = np.load(label_path)
    
    pretrain_ae(model, dataset, y, dataset_name, device, n_z)
    
    
    feature_path = f"../dataset/{dataset_name}/features.npy"
    reduced_features = np.load(feature_path)
    
    return reduced_features

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(f"../dataset/Adam/feature.npy")
    dataset_name = "Adam"
    features = autoencoder_dim_reduction(X, dataset_name, device)