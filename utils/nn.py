from sklearn.mixture import GaussianMixture

import torch
from torch import nn

from umap import UMAP



class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Encoder(784, 500, True),
                                     Encoder(500, 500, True),
                                     Encoder(500, 2000, True),
                                     Encoder(2000, 10, False))
        self.decoder = nn.Sequential(Decoder(10, 2000, True),
                                     Decoder(2000, 500, True),
                                     Decoder(500, 500, True),
                                     Decoder(500, 784, False))
            
    def forward(self, x):
        x  = self.encoder(x)
        gen = self.decoder(x)
        return x, gen


class N2D(nn.Module):
    def __init__(self, encoder, n_cluster):
        super().__init__()
        self.encoder = encoder
        self.n_cluster = n_cluster
        self.umap = None
        self.gmm = None
    
    def forward(self, x):
        x = self.encode(x)
        x = self.manifold(x)
        cluster = self.cluster(x)
        return cluster

    def encode(self, x):
        with torch.no_grad():
            x = self.encoder(x).cpu().numpy()
        return x

    def manifold(self, x):
        if self.umap is None:
            print('fit the UMAP ...')
            self.umap = UMAP(20, self.n_cluster).fit(x)
        x = self.umap.transform(x)
        return x

    def cluster(self, x):
        if self.gmm is None:
            print('fit the GMM ...')
            self.gmm = GaussianMixture(self.n_cluster).fit(x)
        prob = self.gmm.predict_proba(x)
        return prob
