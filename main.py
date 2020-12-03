import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from data import load_data, load_cates
from embed import load_embed
from model import load_net


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True,
                    help='film-trust, ciao-dvd, etc.')
parser.add_argument('--model', type=str, default='DisenVAE',
                    help='MultiDAE, MultiVAE, DisenVAE, DisenVAE')
parser.add_argument('--mode', type=str, default='train',
                    help='train, test, visualize')
parser.add_argument('--seed', type=int, default=98765)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=800)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--kfac', type=int, default=7)
parser.add_argument('--dfac', type=int, default=200)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu, cuda')
args = parser.parse_args()


if args.seed < 0:
    args.seed = int(time.time())
info = '%s-%gL-%dE-%dB-%gW-%gD-%gb-%dk-%dd-%gt-%ds' \
    % (args.data, args.lr, args.epochs, args.batch_size,
       args.weight_decay, args.dropout, args.beta,
       args.kfac, args.dfac, args.tau, args.seed)
print(info)


np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device(args.device \
        if torch.cuda.is_available() else 'cpu')


dir = os.path.join('data', args.data)
n_users, n_items, tr_data, te_data, train_idx,  \
    valid_idx, test_idx, social_data = load_data(dir)
items_embed, cores_embed = load_embed(dir)      \
    if args.model == 'DisenEVAE' else None, None
net = load_net(args.model, n_users, n_items, args.kfac, args.dfac, 
               args.tau, args.dropout, items_embed, cores_embed)
net.to(device)


def train(net, train_idx, valid_idx):
    optimizer = optim.Adam(net.parameters(), lr=args.lr, 
                           weight_decay=args.weight_decay)
    criterion = net.loss_fn

    n_train = len(train_idx)
    n_batches = int(np.ceil(n_train / args.batch_size))
    update = 0
    anneals = 500 * n_batches

    best_n100 = 0.0
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_idx = np.random.permutation(train_idx)

        t = time.time()
        for start_idx in range(0, n_train, args.batch_size):
            end_idx = min(start_idx + args.batch_size, n_train)
            X = tr_data[train_idx[start_idx: end_idx]]
            X = torch.Tensor(X.toarray()).to(device)     # users-items matrix
            if social_data is not None:
                A = social_data[train_idx[start_idx: end_idx]]
                A = torch.Tensor(A.toarray()).to(device) # users-users matrix
            else:
                A = None
            optimizer.zero_grad()
            X_logits, X_mu, X_logvar, A_logits, A_mu, A_logvar = net(X, A)
            anneal = min(args.beta, update / anneals)
            loss = criterion(X, X_logits, X_mu, X_logvar,
                             A, A_logits, A_mu, A_logvar, anneal)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            update += 1

        print('[%3d] loss: %.3f' % (epoch, running_loss / n_batches), end='\t')
        n100, r20, r50 = test(net, valid_idx)
        if n100 > best_n100:
            best_n100 = n100
            torch.save(net.state_dict(), 'model/%s.pkl' % args.model)
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
            X_tr = torch.Tensor(X_tr.toarray()).to(device)
            X_te = torch.Tensor(X_te.toarray())
            if social_data is not None:
                A = social_data[train_idx[start_idx: end_idx]]
                A = torch.Tensor(A.toarray()).to(device)
            else:
                A = None
            X_tr_logits, _, _, _, _, _ = net(X_tr, A)

            # exclude X_tr_logits's samples from tr_data
            X_tr_logits[torch.nonzero(X_tr, as_tuple=True)] = float('-inf')
            X_tr_logits = X_tr_logits.cpu()

            n100s.append(ndcg_kth(X_tr_logits, X_te, k=100))
            r20s.append(recall_kth(X_tr_logits, X_te, k=20))
            r50s.append(recall_kth(X_tr_logits, X_te, k=50))
            
    n100s = torch.cat(n100s)
    r20s = torch.cat(r20s)
    r50s = torch.cat(r50s)

    print('ndcg@100: %.5f(±%.5f)' % (n100s.mean(), n100s.std() / np.sqrt(len(n100s))), end='\t')
    print('recall@20: %.5f(±%.5f)' % (r20s.mean(), r20s.std() / np.sqrt(len(r20s))), end='\t')
    print('recall@50: %.5f(±%.5f)' % (r50s.mean(), r50s.std() / np.sqrt(len(r50s))), end='\t')
    return n100s.mean(), r20s.mean(), r50s.mean()


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


def visualize(net, idx):
    net.eval()
    n_visual = len(idx)
    users = []
    with torch.no_grad():
        for start_idx in range(0, n_visual, args.batch_size):
            end_idx = min(start_idx + args.batch_size, n_visual)
            X = tr_data[idx[start_idx: end_idx]]
            X = torch.Tensor(X.toarray()).to(device)
            if social_data is not None:
                A = social_data[train_idx[start_idx: end_idx]]
                A = torch.Tensor(A.toarray()).to(device)
            else:
                A = None
            _, X_mu, _, _, _, _ = net(X, A)
            users.append(X_mu)
    
    users = torch.cat(users).detach().cpu()
    items = net.state_dict()['items'].detach().cpu()
    cores = net.state_dict()['cores'].detach().cpu()


    # align categories with prototypes
    cates_true = load_cates()
    cates_true = match_cores_cates(items, cores, cates_true)

    # nodes (items and users) prediction and ground truth
    nodes = torch.cat((items, users))
    print(nodes.shape)
    nodes_pred = torch.mm(nodes, cores)


    # plot pictures
    palette = np.array(
        [[238, 27 , 39 , 80],  # _0. Red
         [59 , 175, 81 , 80],  # _1. Green
         [255, 127, 38 , 80],  # _2. Orange
         [255, 129, 190, 80],  # _3. Pink
         [153, 153, 153, 80],  # _4. Gray
         [156, 78 , 161, 80],  # _5. Purple
         [35 , 126, 181, 80]], # _6. Blue
        dtype=np.float) / 255.0
    

        

    # TSNE

    # plot

    return


def match_cores_cates(self, items, cores, cates):
    '''
    items = embedding matrix [m, d]
    cores = embedding matrix [k, d]
    cates = one-hot matrix   [m, k] 
    '''
    # normalize items, cores
    items = F.normalize(items)
    cores = F.normalize(cores)
    
    # align categories with prototypes
    cates_centers = []
    for ki in args.kfac:
        cates_centers.append(torch.sum(items[cates_labels == ki], 
                                    dim=0, keepdim=True))
    cates_centers = torch.cat(cates_centers)
    cates_centers = F.normalize(cates_centers)
    cores_cates = torch.mm(cores, cates_labels.t())
    cores2cates = torch.argmax(cores_cates, dim=1)
    cates2cores = torch.argmax(cores_cates, dim=0)

    if len(set(cores2cates)) == args.k and  \
    len(set(cates2cores)) == args.k:
        for ki in args.kfac:
            if cores2cates[cates2cores[ki]] != ki:
                break
        else:
            return True, cates2cores
    return False, None 


if not os.path.exists('model'):
    os.mkdir('model')


if args.mode == 'train':
    print('training ...')
    t = time.time()
    try:
        train(net, train_idx, valid_idx)
    except KeyboardInterrupt:
        print('terminate training...')
    print('train time: %.3f' % (time.time() - t))


if args.mode == 'train' or args.mode == 'test':
    print('testing ...')
    t = time.time()
    net.load_state_dict(torch.load('model/%s.pkl' % args.model))
    test(net, test_idx)
    print('test time: %.3f' % (time.time() - t))


if args.mode == 'visualize':
    print('visualizing...')
    t = time.time()
    net.load_state_dict(torch.load('model/%s.pkl' % args.model))
    visualize(net, train_idx)
    print('test time: %.3f' % (time.time() - t))