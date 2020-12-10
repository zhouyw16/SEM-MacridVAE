import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 

from data import load_data, load_cates
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


dir = os.path.join('RecomData', args.data)
n_users, n_items, tr_data, te_data, train_idx,  \
    valid_idx, test_idx, items_embed, social_data = load_data(dir)
net = load_net(args.model, n_users, n_items, args.kfac, args.dfac, 
               args.tau, args.dropout, items_embed)
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

    users = F.normalize(users)  \
            .view(n_users, args.kfac, args.dfac)
    items = F.normalize(items)
    cores = F.normalize(cores)

    # align categories with prototypes
    items_cates = load_cates()
    items_cates = match_cores_cates(items, cores, items_cates)
    items_item, items_cate = items_cates.nonzero()

    # users and the categories they bought
    assert sparse.isspmatrix(tr_data) and sparse.isspmatrix(items_cates)
    t = time.time()
    users_cates = np.dot(tr_data[idx], items_cates)
    print(time.time() - t)
    t = time.time()
    users_cates_ = tr_data[idx] * items_cates
    print(time.time() - t)
    print((users_cates == users_cates_).all())
    users_user, users_cate = users_cates.nonzero()
    users = users[users_user, users_cate, :]

    # nodes (items and users) prediction and ground truth
    nodes = torch.cat((items, users))
    print(nodes.shape)
    nodes_pred = torch.argmax(torch.mm(nodes, cores.t()), dim=1)
    nodes_true = torch.cat(items_cate, users_cate)


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
    
    col_pred = palette[nodes_pred]
    col_true = palette[nodes_true]

    try:
        nodes_2d = np.load('run/tsne.npy')
    except:
        print('tsne...')
        nodes_kd = PCA(n_components=args.kfac).fit_transform(nodes) \
                   if args.dfac > args.kfac else nodes
        nodes_2d = TSNE(n_jobs=8).fit_transform(nodes_kd)
        np.save('run/tsne.npy')
    plot('tsne2d-nodes-pred', nodes_2d, col_pred)
    plot('tsne2d-nodes-gold', nodes_2d, col_true)
    plot('tsne2d-items-pred', nodes_2d[:n_items], col_pred[:n_items])
    plot('tsne2d-items-gold', nodes_2d[:n_items], col_true[:n_items])
    plot('tsne2d-users-pred', nodes_2d[n_items:], col_pred[n_items:])
    plot('tsne2d-users-gold', nodes_2d[n_items:], col_true[n_items:])


def match_cores_cates(items, cores, items_cates):
    '''
    align categories with prototypes 

    items = embedding matrix [m, d]
    cores = embedding matrix [k, d]
    items_cates = sparse one-hot matrix [m, k]
    '''
    cates = np.argmax(items_cates, axis=1)
    cates_centers = [torch.sum(items[cates == ki], dim=0, keepdim=True) 
                    for ki in range(args.kfac)]
    cates_centers = torch.cat(cates_centers)
    cates_centers = F.normalize(cates_centers)
    cores_cates = torch.mm(cores, cates_centers.t())
    cates2cores = torch.argmax(cores_cates, dim=1)
    cores2cates = torch.argmax(cores_cates, dim=0)

    print('cores: ', end='  ')
    print('%d' % ki for ki in range(args.kfac))
    print('cates: ', end='  ')
    print('%d' % cates2cores[ki] for ki in range(args.kfac))
    print()
    print('cates: ', end='  ')
    print('%d' % ki for ki in range(args.kfac))
    print('cores: ', end='  ')
    print('%d' % cores2cates[ki] for ki in range(args.kfac))

    if len(set(cores2cates)) == args.kfac and  \
       len(set(cates2cores)) == args.kfac:
        for ki in args.kfac:
            if cores2cates[cates2cores[ki]] != ki:
                break
        else:
            return items_cates[:, cates2cores]
    print('Some prototypes do not align well with categories.')
    exit()


def plot(fname, xy, color, marksz=2.0):
    plt.figure()
    plt.scatter(x=xy[:, 0], y=xy[:, 1], c=color, s=marksz)
    plt.savefig('%s.png' % fname)



if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('run'):
    os.mkdir('run')

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