import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import torch
import torch.utils.data as Data


def load_data(dir):
    ''' 
    load data from preprocessed txt files
    '''
    dir = os.path.join(dir, 'prep')


    n_users = load_users_items(
        os.path.join(dir, 'users.txt')
    )

    n_items = load_users_items(
        os.path.join(dir, 'items.txt')
    )
    print('users: ', n_users, '\titems: ', n_items)

    train_data = load_train(
        os.path.join(dir, 'train.txt'), 
        n_items)

    valid_tr_data, valid_te_data = load_valid_test(
        os.path.join(dir, 'valid_tr.txt'), 
        os.path.join(dir, 'valid_te.txt'), 
        n_items)

    test_tr_data, test_te_data = load_valid_test(
        os.path.join(dir, 'test_tr.txt'), 
        os.path.join(dir, 'test_te.txt'), 
        n_items)

    try:
        social_data = load_social(
            os.path.join(dir, 'social.txt'),
            n_users
        )
    except:
        social_data = None
    
    return n_users, n_items, train_data, valid_tr_data, \
           valid_te_data, test_tr_data, test_te_data, social_data


def load_users_items(file):
    data = pd.read_csv(file)
    return len(data)


def load_train(file, n_items):
    df = pd.read_csv(file)
    users, items = df['user'], df['item']
    max_idx, min_idx = users.max(), users.min()

    assert min_idx == 0
    n_users = max_idx + 1

    data = sparse.csr_matrix((np.ones_like(users), (users, items)),
                            shape=(n_users, n_items), dtype=np.float)
    return data


def load_valid_test(tr_file, te_file, n_items):
    tr_df,  te_df  = pd.read_csv(tr_file), pd.read_csv(te_file)
    tr_users, tr_items = tr_df['user'], tr_df['item']
    te_users, te_items = te_df['user'], te_df['item']
    max_idx = max(tr_users.max(), te_users.max())
    min_idx = min(tr_users.min(), te_users.min())

    # map from [min, max] to [0, max - min]
    tr_users = tr_users - min_idx
    te_users = te_users - min_idx
    n_users = max_idx - min_idx + 1

    tr_data = sparse.csr_matrix((np.ones_like(tr_users), (tr_users, tr_items)),
                                shape=(n_users, n_items), dtype=np.float)
    te_data = sparse.csr_matrix((np.ones_like(te_users), (te_users, te_items)),
                                shape=(n_users, n_items), dtype=np.float)
    return tr_data, te_data


def load_social(file, n_users):
    df = pd.read_csv(file)
    trustor, trustee = df['trustor'], df['trustee']
    data = sparse.csr_matrix((np.ones_like(trustor), (trustor, trustee)),
                            shape=(n_users, n_users), dtype=np.float)
    return data


def load_data_dense(dir):
    '''
    load whole tr_ratings, te_ratings and social information
    '''
    n_users, n_items, train_data, valid_tr_data, valid_te_data, \
        test_tr_data, test_te_data, social_data = load_data(dir)
    
    tr_data = sparse.vstack([train_data, valid_tr_data, test_tr_data])
    te_data = sparse.vstack([train_data, valid_te_data, test_te_data])

    n_train = train_data.shape[0]
    n_valid = valid_tr_data.shape[0]
    n_test  = test_tr_data.shape[0]
    train_index = range(n_train)
    valid_index = range(n_train, n_train + n_valid)
    test_index  = range(n_train + n_valid, n_train + n_valid + n_test)

    return n_users, n_items, tr_data, te_data, \
        train_index, valid_index, test_index, social_data


class SparseDataset(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        index = torch.LongTensor(np.vstack((data.row, data.col)))
        value = torch.FloatTensor(data.data)
        shape = torch.Size(data.shape)
        data = torch.sparse.FloatTensor(index, value, shape)
        return data

def sparse_collate(batch):
    return torch.cat(batch, 0)

def sparse_loader(data, batch_size, shuffle=False):
    return Data.DataLoader(
        SparseDataset(data), 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=sparse_collate
    )

def load_data_sparse(dir, batch_size, shuffle=False):
    '''
    load torch dataloader
    '''
    n_users, n_items, train_data, valid_tr_data, valid_te_data, \
        test_tr_data, test_te_data, social_data = load_data(dir)

    train_loader    = sparse_loader(train_data,    batch_size, shuffle)
    valid_tr_loader = sparse_loader(valid_tr_data, batch_size, shuffle)
    valid_te_loader = sparse_loader(valid_te_data, batch_size, shuffle)
    test_tr_loader  = sparse_loader(test_tr_data,  batch_size, shuffle)
    test_te_loader  = sparse_loader(test_te_data,  batch_size, shuffle)
    social_loader   = sparse_loader(social_data,   batch_size, shuffle)

    return n_users, n_items, train_loader, valid_tr_loader, \
        valid_te_loader, test_tr_loader, test_te_loader, social_loader


if __name__ == "__main__":
    try:
        dir = sys.argv[1]
    except:
        print('please input a correct directory name.')
        exit()

    load_data_dense(dir)
    load_data_sparse(dir, 50)
