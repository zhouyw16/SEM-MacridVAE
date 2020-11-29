
import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim

from data import load_sparse_data
from model import DisenSE


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True,
                    help='film-trust, ciao-dvd, etc.')
parser.add_argument('--mode', type=str, default='train',
                    help='train, test, visualize')
parser.add_argument('--seed', type=int, default=98765)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--kfac', type=int, default=7)
parser.add_argument('--dfac', type=int, default=100)
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu, cuda')
args = parser.parse_args()


if args.seed < 0:
    args.seed = int(time.time())
info = '%s-%gL-%dE-%dB-%gW-%gD-%gb-%dt-%dk-%dd-%ds' \
    % (args.data, args.lr, args.epochs, args.batch_size,
       args.weight_decay, args.dropout, args.beta,
       args.tau, args.kfac, args.dfac, args.seed)
print(info)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
    

dir = os.path.join('data', args.data)
n_users, n_items, tr_data, te_data, train_idx,    \
    valid_idx, test_idx, social_data = load_sparse_data(dir)


def train(net, train_idx, valid_idx):
    optimizer = optim.Adam(net.parameters(), lr=args.lr, 
                           weight_decay=args.weight_decay)
    criterion = net.loss_fn

    n_train = len(train_idx)
    n_batches = int(np.ceil(n_train / args.batch_size))
    update = 0
    anneals = 5 * n_batches

    best_n100 = 0.0
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        # train_idx = np.random.permutation(train_idx)

        t = time.time()
        for start_idx in range(0, n_train, args.batch_size):
            end_idx = min(start_idx + args.batch_size, n_train)
            X = tr_data[train_idx[start_idx: end_idx]]
            A = social_data[train_idx[start_idx: end_idx]]
            X = X.toarray()     # users-items matrix    TODO: cuda
            A = A.toarray()     # users-users matrix    TODO: cuda
            optimizer.zero_grad()
            X_recon, X_mu, X_logvar, A_recon, A_mu, A_logvar = net(X, A)
            anneal = min(args.beta, update / anneals)
            loss = criterion(X, X_recon, X_mu, X_logvar,
                             A, A_recon, A_mu, A_logvar, anneal)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            update += 1

        print('[%3d] loss: %.3f' % (epoch, running_loss / n_train), end='\t')
        n100, r20, r50 = test(net, valid_idx)
        if n100 > best_n100:
            best_n100 = n100
            torch.save(net.state_dict(),'disen_se.pkl')
        print('time: %.3f' % (time.time() - t))


def test(net, idx):
    net.eval()
    n_test = len(idx)
    n100s, r20s, r50s = [], [], []
    with torch.no_grad():
        for start_idx in range(0, n_test, args.batch_size):
            end_idx = min(start_idx + args.batch_size, n_test)
            X_tr  = tr_data[idx[start_idx: end_idx]]
            X_te  = te_data[idx[start_idx: end_idx]]
            A = social_data[idx[start_idx: end_idx]]
            X_tr = X_tr.toarray()
            X_te = X_te.toarray()
            A = A.toarray()
            X_tr_recon, _, _, _, _, _ = net(X_tr, A)

            # exclude X_tr_recon's samples from tr_data
            X_tr_recon[torch.nonzero(X_tr, as_tuple=True)] = float('-inf')

            n100s.append(ndcg_kth(X_tr_recon, X_te, k=100))
            r20s.append(recall_kth(X_tr_recon, X_te, k=20))
            r50s.append(recall_kth(X_tr_recon, X_te, k=50))
            
    n100s = torch.cat(n100s)
    r20s = torch.cat(r20s)
    r50s = torch.cat(r50s)

    print('ndcg@100: %.5f (±%.5f)' % (n100s.mean(), n100s.std() / np.sqrt(len(n100s))), end='\t')
    print('recall@20: %.5f (±%.5f)' % (r20s.mean(), r20s.std() / np.sqrt(len(r20s))), end='\t')
    print('recall@50: %.5f (±%.5f)' % (r50s.mean(), r50s.std() / np.sqrt(len(r50s))), end='\t')
    return n100s.mean(), r20s.mean(), r50s.mean()


def visualize():
    return


def ndcg_kth(outputs, labels, k=100):
    _, preds = torch.topk(outputs, k)                       # sorted top k index of outputs
    _, facts = torch.topk(labels, k)                        # min(k, labels.nnz(dim=1))
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    tp = 1.0 / torch.log2(torch.arange(2, k + 2).float())
    dcg = torch.sum(tp * labels[rows, preds], dim=1)
    idcg = torch.sum(tp * labels[rows, facts], dim=1)
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def recall_kth(outputs, labels, k=50):
    _, preds = torch.topk(outputs, k, sorted=False)         # top k index
    rows = torch.arange(labels.shape[0]).view(-1, 1)

    recall = torch.sum(labels[rows, preds], dim=1) \
           / torch.min(torch.Tensor([k]), torch.sum(labels, dim=1))
    recall[torch.isnan(recall)] = 0
    return recall


if args.mode == 'train':
    print('training ...')
    t = time.time()
    net = DisenSE()
    try:
        train(net, train_idx, valid_idx)
    except KeyboardInterrupt:
        print('terminate training...')
    print('train time: %.3f' % (time.time() - t))


if args.mode == 'test':
    print('testing ...')
    t = time.time()
    net = DisenSE()
    net.load_state_dict(torch.load('disen_se.pkl'))
    test(net, test_idx)
    print('test time: %.3f' % (time.time() - t))


if args.mode == 'visualize':
    print('visualizing...')
    visualize()