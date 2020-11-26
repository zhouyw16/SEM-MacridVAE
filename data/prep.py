import os
import sys

import numpy as np
import pandas as pd


# input data directory
try:
    dir = sys.argv[1]
    os.chdir(os.path.join('.', dir, 'prep'))
except:
    print('please input a correct directory name.')
    exit()
print('%s:' % dir)


# read ratings and trusts if exists
ratings = pd.read_csv('ratings.txt')
try:
    trusts  = pd.read_csv('trusts.txt')
except:
    trusts = None


def info(df, name):
    '''
    print dataframe's infomation
    '''
    nnz = df.shape[0]
    names = df.columns.values
    rows = len(pd.unique(df[names[0]]))
    cols = len(pd.unique(df[names[1]]))
    sparsity = (nnz / (rows * cols)) * 100.0
    print('%s:\t%ss: %d\t%ss: %d\trecords: %d\tsparsity: %.4f%%' %
        (name, names[0], rows, names[1], cols, nnz, sparsity))

# reserve ratings >= 4.0
ratings = ratings[ratings['rating'] >= 4.0]
info(ratings, 'init   ratings')


def filter(df, name, min):
    '''
    filter the group whose size < min
    '''
    if min > 0:
        size = df.groupby(name).size()
        df = df[df[name].isin(size.index[size >= min])]
        return df


# keep the users with at least 5 items
ratings = filter(ratings, 'user', 5)
info(ratings, 'filter ratings')
print()


# save users' unique indices
users_id = pd.unique(ratings['user'])
pd.DataFrame({'id': users_id}).to_csv('users.txt', index=False)


# split train/valid/test users
np.random.seed(98765)
np.random.shuffle(users_id)
n_users = len(users_id)
n_test  = int(0.1 * n_users)
train_users = users_id[:(n_users - n_test * 2)]
valid_users = users_id[(n_users - n_test * 2):(n_users - n_test)]
test_users  = users_id[(n_users - n_test):]


# save items' unique indices (only train items)
train_ratings = ratings[ratings['user'].isin(train_users)]
items_id = pd.unique(train_ratings['item'])
pd.DataFrame({'id': items_id}).to_csv('items.txt', index=False)


# print train/valid/test ratings now
# NOTE: now there exist users with less than 5 items in valid/test ratings 
ratings = ratings[ratings['item'].isin(items_id)]
train_ratings = ratings[ratings['user'].isin(train_users)]
valid_ratings = ratings[ratings['user'].isin(valid_users)]
test_ratings  = ratings[ratings['user'].isin(test_users) ]
info(train_ratings, 'train  ratings')
info(valid_ratings, 'valid  ratings')
info(test_ratings , 'test   ratings')


def split(df, p_test=0.2):
    groups = df.groupby('user')
    tr, te = [], []

    np.random.seed(98765)
    for _, group in groups:
        n_items = len(group)
        if n_items >= 5:
            te_items = np.random.choice(group['item'], replace=False,
                                        size=int(p_test * n_items))
            tr.append(group[~group['item'].isin(te_items)])
            te.append(group[group['item'].isin(te_items)])
        else:
            tr.append(group)
    
    return pd.concat(tr), pd.concat(te)

# split valid/test data to training and testing part
valid_ratings_tr, valid_ratings_te = split(valid_ratings)
test_ratings_tr,  test_ratings_te  = split(test_ratings)


# mapping id to [0, len(id))
users_dict = dict((user_id, i) for (i, user_id) in enumerate(users_id))
items_dict = dict((item_id, i) for (i, item_id) in enumerate(items_id))

def savedf(df, name):
    '''
    mapping and save dataframe
    '''
    users = list(map(lambda id: users_dict[id], df['user']))
    items = list(map(lambda id: items_dict[id], df['item']))
    pd.DataFrame({'user': users, 'item': items})    \
      .sort_values(by='user')                       \
      .to_csv('%s.txt' % name, index=False)

savedf(train_ratings,    'train')
savedf(valid_ratings_tr, 'valid_tr')
savedf(valid_ratings_te, 'valid_te')
savedf(test_ratings_tr,  'test_tr')
savedf(test_ratings_te,  'test_te')