import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse


def load_data(dir):
    '''
    load whole tr_ratings, te_ratings and social information
    '''
    n_users, n_items, train_data, valid_tr_data, valid_te_data, \
        test_tr_data, test_te_data, embed_data, social_data = load_perp_data(dir)
    empty_data = sparse.csr_matrix(train_data.shape, dtype=float)

    tr_data = sparse.vstack([train_data, valid_tr_data, test_tr_data])
    te_data = sparse.vstack([empty_data, valid_te_data, test_te_data])

    n_train = train_data.shape[0]
    n_valid = valid_tr_data.shape[0]
    n_test  = test_tr_data.shape[0]
    train_idx = range(n_train)
    valid_idx = range(n_train, n_train + n_valid)
    test_idx  = range(n_train + n_valid, n_train + n_valid + n_test)

    return n_users, n_items, tr_data, te_data, train_idx,       \
           valid_idx, test_idx, embed_data, social_data


def load_perp_data(dir):
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
        embed_data = load_embed(
            os.path.join(dir, 'embed.npy')
        )
    except:
        embed_data = None

    try:
        social_data = load_social(
            os.path.join(dir, 'social.txt'),
            n_users
        )
    except:
        social_data = None

    return n_users, n_items, train_data, valid_tr_data, valid_te_data, \
           test_tr_data, test_te_data, embed_data, social_data


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
                            shape=(n_users, n_items), dtype=float)
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
                                shape=(n_users, n_items), dtype=float)
    te_data = sparse.csr_matrix((np.ones_like(te_users), (te_users, te_items)),
                                shape=(n_users, n_items), dtype=float)
    return tr_data, te_data


def load_social(file, n_users):
    df = pd.read_csv(file)
    trustors, trustees = df['trustor'], df['trustee']
    data = sparse.csr_matrix((np.ones_like(trustors), (trustors, trustees)),
                            shape=(n_users, n_users), dtype=float)
    return data


def load_embed(file):
    data = np.load(file)
    return data


def load_urls(dir):
    file = os.path.join(dir, 'prep', 'images.txt')
    data = np.loadtxt(file, dtype=np.str)
    return data


def load_cates(dir, n_items, k_cates):
    file = os.path.join(dir, 'prep', 'categorial.txt')
    df = pd.read_csv(file)
    items, cates = df['item'], df['cate']
    n_cates = np.max(cates) + 1
    k_cates = min(k_cates, n_cates)

    # create sparse matrix
    data = sparse.csr_matrix((np.ones_like(items), (items, cates)),
                            shape=(n_items, n_cates), dtype=float)
    # to dense matrix
    data = data.toarray()
    # choose top k categories with most items
    cates_id = np.argsort(np.sum(data, 0)[-k_cates:])
    data = data[:, cates_id]
    # make every item belong to unique category
    eps = 1e-6
    item_cate = [np.random.choice(k_cates, 1, p=(item_cates / item_cates.sum()))[0] \
                for item_cates in (data + eps)]
    data = np.eye(k_cates)[item_cate, :]

    assert np.min(np.sum(data, axis=1)) == 1
    assert np.max(np.sum(data, axis=1)) == 1
    return data


if __name__ == "__main__":
    try:
        dir = os.path.join('RecomData', sys.argv[1])
    except:
        print('please input a correct directory name.')
        exit()

    t = time.time()
    n_users, n_items, tr_data, te_data, train_idx, valid_idx, \
        test_idx, embed_data, social_data = load_data(dir)
    items_cates = load_cates(dir, n_items, 7)
    print('%.4fs' % (time.time() - t))
