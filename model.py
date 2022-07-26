import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale



def load_net(model, N, M, K, D, tau, dropout, items):
    if model == 'MultiDAE':
        return MultiDAE(M, D, dropout)
    elif model == 'MultiVAE':
        return MultiVAE(M, D, dropout)
    elif model == 'DisenVAE':
        return DisenVAE(M, K, D, tau, dropout)
    elif model == 'DisenEVAE':
        return DisenEVAE(M, K, D, tau, dropout, items)



def recon_loss(inputs, logits):
    return torch.mean(torch.sum(-logits * inputs, dim=1))

def kl_loss(mu, logvar):
    return torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1))



class MultiDAE(nn.Module):
    def __init__(self, M, D, dropout):
        super(MultiDAE, self).__init__()

        self.M = M
        self.H = D * 3
        self.D = D

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.D, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.M)
        )
        self.drop = nn.Dropout(dropout)
    

    def encode(self, X):
        X = self.drop(X)
        h = self.encoder(X)
        return h


    def decode(self, h):
        logits = self.decoder(h)
        logits = F.log_softmax(logits, dim=1)
        return logits


    def forward(self, X, A):
        h = self.encode(X)
        logits = self.decode(h)
        return logits, None, None, None, None, None
    

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits)
    


class MultiVAE(nn.Module):
    def __init__(self, M, D, dropout):
        super(MultiVAE, self).__init__()

        self.M = M
        self.H = D * 3
        self.D = D

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.D, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.M)
        )
        self.drop = nn.Dropout(dropout)
    

    def encode(self, X):
        X = self.drop(X)
        h = self.encoder(X)
        mu = h[:, :self.D]
        logvar = h[:, self.D:]
        return mu, logvar


    def decode(self, z):
        logits = self.decoder(z)
        logits = F.log_softmax(logits, dim=1)
        return logits


    def sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu
    

    def forward(self, X, A):
        mu, logvar = self.encode(X)
        z = self.sample(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, None, None, None
    
    
    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits) + anneal * kl_loss(X_mu, X_logvar)



class DisenVAE(nn.Module):
    def __init__(self, M, K, D, tau, dropout):
        super(DisenVAE, self).__init__()

        self.M = M
        self.H = D * 3
        self.D = D
        self.K = K
        self.tau = tau

        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.items = Parameter(torch.Tensor(self.M, self.D))
        self.cores = Parameter(torch.Tensor(self.K, self.D))
        self.drop = nn.Dropout(dropout)

        init.xavier_normal_(self.items)
        init.xavier_normal_(self.cores)


    def cluster(self):
        items = F.normalize(self.items, dim=1)      # M * D
        cores = F.normalize(self.cores, dim=1)      # K * D
        cates = torch.mm(items, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)             # M * K
        return items, cores, cates


    def encode(self, X, cates):
        n = X.shape[0]
        X = self.drop(X)
        X = X.view(n, 1, self.M) *  \
            cates.t().expand(n, self.K, self.M)     # n * K * M
        X = X.reshape(n * self.K, self.M)           # (n * K) * M
        h = self.encoder(X)                         # (n * K) * D * 2
        mu, logvar = h[:, :self.D], h[:, self.D:]   # (n * k) * D
        return mu, logvar


    def decode(self, z, items, cates):
        n = z.shape[0] // self.K
        z = F.normalize(z, dim=1)                   # (n * K) * D
        logits = torch.mm(z, items.t()) / self.tau  # (n * K) * M
        probs = torch.exp(logits)                   # (n * K) * M
        probs = torch.sum(probs.view(n, self.K, self.M) * \
                cates.t().expand(n, self.K, self.M), dim=1)
        logits = torch.log(probs)
        logits = F.log_softmax(logits, dim=1)
        return logits


    def sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu


    def forward(self, X, A):
        items, cores, cates = self.cluster()
        mu, logvar = self.encode(X, cates)
        z = self.sample(mu, logvar)
        logits = self.decode(z, items, cates)
        return logits, mu, logvar, None, None, None
    

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits) + anneal * kl_loss(X_mu, X_logvar)



class DisenEVAE(DisenVAE):
    def __init__(self, M, K, D, tau, dropout, items):
        super(DisenEVAE, self).__init__(M, K, D, tau, dropout)

        # change the feature from X to self.D dimensions
        items = PCA(n_components=self.D).fit_transform(items)
        # fit the xavier_normal distribution i.e. mu = 0, std = sqrt(2 / (fan_in + fan_out))
        items = scale(items, axis=1) * np.sqrt(2 / (M + D))
        # init the feature of cores
        cores = KMeans(n_clusters=self.K).fit(items).cluster_centers_

        self.items = Parameter(torch.Tensor(items))
        self.cores = Parameter(torch.Tensor(cores))


