import torch
import torch.nn as nn
import torch.nn.functional as F


def load_net(name, mdim, kfac, dfac, dropout, tau):
    if name == 'MultiDAE':
        return MultiDAE(mdim, kfac, dfac, dropout)
    elif name == 'MultiVAE':
        return MultiVAE(mdim, kfac, dfac, dropout)
    elif name == 'DisenVAE':
        return DisenVAE(mdim, kfac, dfac, dropout, tau)
    elif name == 'DisenSE':
        return DisenSE(mdim, kfac, dfac, dropout, tau)


class DisenSE(nn.Module):
    def __init__(self, mdim, kfac, dfac, dropout, tau):
        super(DisenSE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self, X, A):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return


class MultiDAE(nn.Module):
    def __init__(self, mdim, kfac, dfac, dropout):
        super(MultiDAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(mdim, 3 * dfac),
            nn.Tanh(),
            nn.Linear(3 * dfac, dfac),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(dfac, 3 * dfac),
            nn.Tanh(),
            nn.Linear(3 * dfac, mdim)
        )

        self.drop = nn.Dropout(dropout)
    
    def forward(self, X, A):
        X = F.normalize(X)
        X = self.drop(X)
        h = self.encoder(X)
        X_recon = self.decoder(h)
        X_recon = F.log_softmax(X_recon, dim=1)
        return X_recon, None, None, None, None, None
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return torch.mean(torch.sum(-X_recon * X, dim=1))
    

class MultiVAE(nn.Module):
    def __init__(self, mdim, kfac, dfac, dropout):
        super(MultiVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(mdim, 3 * dfac),
            nn.Tanh()
        )
        self.mu = nn.Linear(3 * dfac, dfac)
        self.var = nn.Linear(3 * dfac, dfac)

        self.decoder = nn.Sequential(
            nn.Linear(dfac, 3 * dfac),
            nn.Tanh(),
            nn.Linear(3 * dfac, mdim)
        )

        self.drop = nn.Dropout(dropout)
    
    def forward(self, X, A):
        X = F.normalize(X)
        X = self.drop(X)
        h = self.encoder(X)
        X_recon = self.decoder(h)
        X_recon = F.log_softmax(X_recon, dim=1)
        return X_recon, None, None, None, None, None
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return torch.mean(torch.sum(-X_recon * X, dim=1))


class DisenVAE(nn.Module):
    def __init__(self, mdim, kfac, dfac, dropout, tau):
        super(DisenVAE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self, X, A):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return
    