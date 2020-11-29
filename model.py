import torch
import torch.nn as nn


def load_net(name, mdim, kfac, dfac, tau):
    if name == 'MultiDAE':
        return MultiDAE(mdim, kfac, dfac)
    elif name == 'MultiVAE':
        return MultiVAE(mdim, kfac, dfac)
    elif name == 'DisenVAE':
        return DisenVAE(mdim, kfac, dfac, tau)
    elif name == 'DisenSE':
        return DisenSE(mdim, kfac, dfac, tau)


class DisenSE(nn.Module):
    def __init__(self, mdim, kfac, dfac, tau):
        super(DisenSE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self, X, A):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return


class MultiDAE(nn.Module):
    def __init__(self, mdim, kfac, dfac):
        super(MultiDAE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self, X, A):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return
    

class MultiVAE(nn.Module):
    def __init__(self, mdim, kfac, dfac):
        super(MultiVAE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self, X, A):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return


class DisenVAE(nn.Module):
    def __init__(self, mdim, kfac, dfac, tau):
        super(DisenVAE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self, X, A):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return
    