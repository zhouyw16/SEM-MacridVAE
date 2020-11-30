import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter



def load_net(model, N, M, K, D, tau, dropout, items, cores):
    if model == 'MultiDAE':
        return MultiDAE(M, D, dropout)
    elif model == 'MultiVAE':
        return MultiVAE(M, D, dropout)
    elif model == 'DisenVAE':
        return DisenVAE(M, K, D, tau, dropout)
    elif model == 'DisenEVAE':
        return DisenEVAE(M, K, D, tau, dropout, items, cores)
    elif model == 'DisenSE':
        return DisenSE(N, M, K, D, tau, dropout)



class MultiDAE(nn.Module):
    def __init__(self, M, D, dropout):
        super(MultiDAE, self).__init__()

        self.M = M
        self.H = 3 * D
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
        X = F.normalize(X)
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
        recon_loss = torch.mean(torch.sum(-X_logits * X, dim=1))
        return recon_loss
    


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
        X = F.normalize(X)
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
        recon_loss = torch.mean(torch.sum(-X_logits * X, dim=1))
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(X_logvar) + X_mu ** 2 - 1 - X_logvar, dim=1))
        return recon_loss + anneal * kl_loss



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
        items = F.normalize(self.items, dim=1)
        cores = F.normalize(self.cores, dim=1)
        cates = torch.mm(items, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)
        return items, cores, cates


    def encode(self, X, cates):
        n = X.shape[0]
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
        items, _, cates = self.cluster()
        mu, logvar = self.encode(X, cates)
        z = self.sample(mu, logvar)
        logits = self.decode(z, items, cates)
        return logits, mu, logvar, None, None, None
    

    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        recon_loss = torch.mean(torch.sum(-X_logits * X, dim=1))
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(X_logvar) + X_mu ** 2 - 1 - X_logvar, dim=1))
        return recon_loss + anneal * kl_loss



class DisenEVAE(DisenVAE):
    def __init__(self, M, K, D, tau, dropout, items, cores):
        super(DisenEVAE, self).__init__(M, K, D, tau, dropout)

        self.items = items
        self.cores = cores



class DisenSE(nn.Module):
    def __init__(self, N, M, K, D, tau, dropout):
        super(DisenSE, self).__init__()

        self.N = N
        self.M = M
        self.H = D * 3
        self.D = D
        self.K = K
        self.tau = tau

        self.X_encoder = nn.Sequential(
            nn.Linear(self.M, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.A_encoder = nn.Sequential(
            nn.Linear(self.N, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.D * 2)
        )
        self.X_items = Parameter(torch.Tensor(self.M, self.D))
        self.X_cores = Parameter(torch.Tensor(self.K, self.D))
        self.A_users = Parameter(torch.Tensor(self.N, self.D))
        self.A_cores = Parameter(torch.Tensor(self.K, self.D))
        self.drop = nn.Dropout(dropout)

        init.xavier_normal_(self.X_items)
        init.xavier_normal_(self.X_cores)
        init.xavier_normal_(self.A_users)
        init.xavier_normal_(self.A_cores)


    def cluster(self):
        items = F.normalize(self.items, dim=1)
        cores = F.normalize(self.cores, dim=1)
        cates = torch.mm(items, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)
        return items, cores, cates


    def encode(self, X, cates):
        n = X.shape[0]
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
        X_items = F.normalize(self.X_items, dim=1)
        X_cores = F.normalize(self.X_cores, dim=1)
        X_cates = torch.mm(X_items, X_cores.t()) / self.tau
        X_cates = F.softmax(X_cates, dim=1)

        A_users = F.normalize(self.A_users, dim=1)
        A_cores = F.normalize(self.A_cores, dim=1)
        A_cates = torch.mm(A_users, A_cores.t()) / self.tau
        A_cates = F.softmax(A_cates, dim=1)

        return logits, mu, logvar, None, None, None
    
    def loss_fn(self, X, X_recon, X_mu, X_logvar,
                A, A_recon, A_mu, A_logvar, anneal):
        return