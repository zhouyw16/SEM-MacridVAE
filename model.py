import torch
import torch.nn as nn

class DisenSE(nn.Module):
    def __init__(self):
        super(DisenSE, self).__init__()

        self.fc = nn.Linear(10,10)
    
    def forward(self):
        return
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return


