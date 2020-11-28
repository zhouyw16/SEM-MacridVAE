
import os
import time
import argparse

import numpy as np
import torch

from data import load_data_dense, load_data_sparse
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
n_users, n_items, train_loader, valid_tr_loader,    \
    valid_te_loader, test_tr_loader, test_te_loader,\
    social_loader = load_data_sparse(dir, args.batch_size)


def train():
    return 


def valid():
    return 


def visualize():
    return


if args.mode == 'train':
    print('training ...')
    start = time.time()
    net = DisenSE()
    train(net, train_loader, valid_tr_loader, valid_te_loader, social_loader)
    finish = time.time()
    print('train time: %.4f' % (finish - start))


if args.mode == 'test':
    print('testing ...')
    start = time.time()
    net = DisenSE()
    net.load_state_dict(torch.load('disen_se.pkl'))
    valid(net, test_tr_loader, test_te_loader, social_loader)
    finish = time.time()
    print('train time: %.4f' % (finish - start))


if args.mode == 'visualize':
    print('visualizing...')
    visualize()