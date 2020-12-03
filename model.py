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
    elif model == 'DisenSR':
        return DisenSR(N, M, K, D, tau, dropout)



def recon_loss(inputs, logits):
    return torch.mean(torch.sum(-logits * inputs, dim=1))

def kl_loss(mu, logvar):
    return torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1))



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
    def __init__(self, M, K, D, tau, dropout, items, cores):
        super(DisenEVAE, self).__init__(M, K, D, tau, dropout)

        self.items = items
        self.cores = cores



class DisenSR(nn.Module):
    def __init__(self, N, M, K, D, tau, dropout):
        super(DisenSR, self).__init__()

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


    def X_cluster(self):
        items = F.normalize(self.X_items, dim=1)
        cores = F.normalize(self.X_cores, dim=1)
        cates = torch.mm(items, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)
        return items, cores, cates

    def A_cluster(self):
        users = F.normalize(self.A_users, dim=1)
        cores = F.normalize(self.A_cores, dim=1)
        cates = torch.mm(users, cores.t()) / self.tau
        cates = F.softmax(cates, dim=1)
        return users, cores, cates


    def X_encode(self, X, cates):
        n = X.shape[0]
        X = X.view(n, 1, self.M) *  \
            cates.t().expand(n, self.K, self.M)     # n * K * M
        X = X.reshape(n * self.K, self.M)           # (n * K) * M
        h = self.encoder(X)                         # (n * K) * D * 2
        mu, logvar = h[:, :self.D], h[:, self.D:]   # (n * k) * D
        return mu, logvar

    def A_encode(self, A, cates):
        n = A.shape[0]
        A = A.view(n, 1, self.N) *  \
            cates.t().expand(n, self.K, self.N)     # n * K * N
        A = A.reshape(n * self.K, self.N)           # (n * K) * N
        h = self.encoder(A)                         # (n * K) * D * 2
        mu, logvar = h[:, :self.D], h[:, self.D:]   # (n * k) * D
        return mu, logvar


    def X_decode(self, z, items, cates):
        n = z.shape[0] // self.K
        z = F.normalize(z, dim=1)                   # (n * K) * D
        logits = torch.mm(z, items.t()) / self.tau  # (n * K) * M
        probs = torch.exp(logits)                   # (n * K) * M
        probs = torch.sum(probs.view(n, self.K, self.M) * \
                cates.t().expand(n, self.K, self.M), dim=1)
        logits = torch.log(probs)
        logits = F.log_softmax(logits, dim=1)
        return logits

    def A_decode(self, z, items, cates):
        n = z.shape[0] // self.K
        z = F.normalize(z, dim=1)                   # (n * K) * D
        logits = torch.mm(z, items.t()) / self.tau  # (n * K) * M
        probs = torch.exp(logits)                   # (n * K) * M
        probs = torch.sum(probs.view(n, self.K, self.N) * \
                cates.t().expand(n, self.K, self.N), dim=1)
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
        X_items, _, X_cates = self.X_cluster()
        A_items, _, A_cates = self.A_cluster()
        X_mu, X_logvar = self.X_encode(X, X_cates)
        A_mu, A_logvar = self.A_encode(A, A_cates)
        X_z = self.sample(X_mu, X_logvar)
        A_z = self.sample(A_mu, A_logvar)

        # update X_z and A_z jointly
        X_z = X_z + A_z # * TODO
        A_z = A_z + X_z # * TODO

        X_logits = self.X_decode(X_z, X_items, X_cates)
        A_logits = self.A_decode(A_z, A_items, A_cates)

        return X_logits, X_mu, X_logvar, A_logits, A_mu, A_logvar
    
    def loss_fn(self, X, X_logits, X_mu, X_logvar,
                A, A_logits, A_mu, A_logvar, anneal):
        return recon_loss(X, X_logits) + anneal * kl_loss(X_mu, X_logvar) \
             + recon_loss(A, A_logits) + anneal * kl_loss(A_mu, A_logvar)


