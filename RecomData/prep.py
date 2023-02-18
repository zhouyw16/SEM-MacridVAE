import os
import sys
import time

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
print()



ratings = pd.read_csv('ratings.txt')

def info(df, name):
    '''
    print dataframe's infomation
    '''
    nnz = df.shape[0]
    names = df.columns.values
    rows = len(pd.unique(df[names[0]]))
    cols = len(pd.unique(df[names[1]]))
    density = (nnz / (rows * cols)) * 100.0
    sparsity = (1.0 - (nnz / (rows * cols))) * 100.0
    print('%s:  \t%ss: %d\t%ss: %d\trecords: %d\tdensity: %.4f%%\tsparsity: %.4f%%' %
        (name, names[0], rows, names[1], cols, nnz, density, sparsity))

# reserve ratings >= 4.0
info(ratings, 'init   ratings')
ratings = ratings[ratings['rating'] >= 4.0]


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


# save users' unique indices
np.random.seed(98765)
users_id = pd.unique(ratings['user'])
np.random.shuffle(users_id)


# split train/valid/test users
n_users = len(users_id)
n_test  = int(0.1 * n_users)
train_users = users_id[:(n_users - n_test * 2)]
valid_users = users_id[(n_users - n_test * 2):(n_users - n_test)]
test_users  = users_id[(n_users - n_test):]


# save items' unique indices (only train items)
train_ratings = ratings[ratings['user'].isin(train_users)]
items_id = pd.unique(train_ratings['item'])


# exclude ratings with items not in items_id
ratings = ratings[ratings['item'].isin(items_id)]
info(ratings, 'final  ratings')
print()


# print train/valid/test ratings now
# NOTE: now there exist users with less than 5 items in valid/test ratings 
train_ratings = ratings[ratings['user'].isin(train_users)]
valid_ratings = ratings[ratings['user'].isin(valid_users)]
test_ratings  = ratings[ratings['user'].isin(test_users) ]
info(train_ratings, 'train  ratings')
info(valid_ratings, 'valid  ratings')
info(test_ratings , 'test   ratings')
print()


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


def save_ratings(df, name):
    '''
    mapping and save ratings
    '''
    users = list(map(lambda id: users_dict[id], df['user']))
    items = list(map(lambda id: items_dict[id], df['item']))
    pd.DataFrame({'user': users, 'item': items})    \
      .sort_values(by='user')                       \
      .to_csv('%s.txt' % name, index=False) 


pd.DataFrame({'id': users_id}).to_csv('users.txt', index=False)
pd.DataFrame({'id': items_id}).to_csv('items.txt', index=False)
save_ratings(train_ratings,    'train')
save_ratings(valid_ratings_tr, 'valid_tr')
save_ratings(valid_ratings_te, 'valid_te')
save_ratings(test_ratings_tr,  'test_tr')
save_ratings(test_ratings_te,  'test_te')


# def save_trust(df, name):
#     '''
#     mapping and save 
#     '''
#     trustors = list(map(lambda id: users_dict[id], df['trustor']))
#     trustees = list(map(lambda id: users_dict[id], df['trustee']))
#     pd.DataFrame({'trustor': trustors, 'trustee': trustees})    \
#       .sort_values(by='trustor')                                \
#       .to_csv('%s.txt' % name, index=False)   
      
# try:
#     trusts  = pd.read_csv('trusts.txt')
# except:
#     trusts = None

# if trusts is not None:
#     info(trusts, 'init   trusts ')

#     # filter records whose trustor or trustee is not in users
#     trusts = trusts[trusts['trustee'].isin(users_id)] 
#     trusts = trusts[trusts['trustor'].isin(users_id)]
#     info(trusts, 'filter trusts ')

#     # save trusts
#     save_trust(trusts, 'social')



# def save_category(df, name):
#     '''
#     mapping and save 
#     '''
#     items = list(map(lambda id: items_dict[id], df['item']))
#     cates = list(map(lambda id: cates_dict[id], df['cate']))
#     pd.DataFrame({'item': items, 'cate': cates})    \
#       .sort_values(by='item')                       \
#       .to_csv('%s.txt' % name, index=False)   
      
# try:
#     categories = pd.read_csv('categories.txt')
# except:
#     categories = None

# if categories is not None:
#     info(categories, 'init   categories')

#     # filter records whose item is not in items
#     categories = categories[categories['item'].isin(items_id)]
#     info(categories, 'filter categories')

#     # save categories' dictionary
#     cates_count = categories['cate'].value_counts().    \
#                   rename_axis('id').reset_index(name='count')
#     cates_dict = dict((cate_id, i) for (i, cate_id) in enumerate(cates_count['id']))
#     cates_count.to_csv('cates.txt', index=False)
#     print(cates_count)

#     # save categories
#     save_category(categories, 'categorial')


# try:
#     urls = pd.read_csv('urls.txt')
# except:
#     urls = None

# if urls is not None:
#     urls_dict = urls.set_index('item')['url'].to_dict()
#     images = []
#     for item_id in items_id:
#         if item_id in urls_dict:
#             images.append(urls_dict[item_id])
#         else:
#             images.append('')
#     np.savetxt('images.txt', images, fmt='%s')

# t = time.time()
# try:
#     features  = pd.read_csv('features.txt')
# except:
#     features = None
# print('read features: %.2fs' % (time.time() - t))

# if features is not None:
#     embeds = []
#     for item_id in items_id:
#         embeds.append(features[item_id])
#     np.save('embed.npy', embeds)

# try:
#     covers = pd.read_csv('covers.txt', keep_default_na=False)
# except:
#     covers = None

# if covers is not None:
#     from PIL import Image
#     import torch
#     from torchvision import models, transforms
    # covers_dict = covers.set_index('item')['cover'].to_dict()
    # cnt = 0
    # for item_id in items_id:
    #     if covers_dict[item_id]:
    #         try:
    #             with open('../image/%d.jpg' % item_id, 'rb') as image:
    #                 Image.open(image)
    #         except:
    #             print('cannot open %d' % item_id)
    #             cnt += 1
    #             os.system('wget %s -O ../image/%d.jpg' % (covers_dict[item_id], item_id))
    #     else:
    #         print('no url %d' % item_id)
    #         cnt += 1
    # print(len(items_id))
    # print(cnt)

    # model = models.alexnet(pretrained=True)
    # model.classifier = model.classifier[:-1]
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                           std=[0.229, 0.224, 0.225])
    # ])

    # covers_dict = covers.set_index('item')['cover'].to_dict()
    # cnt = 0
    # images = []
    # for item_id in items_id:
    #     try:
    #         with open('../image/%d.jpg' % item_id, 'rb') as file:
    #             image = Image.open(file)
    #             image = transform(image)
    #             images.append(image)
    #     except:
    #         cnt += 1
    #         image = Image.new('RGB', (256, 256), (np.random.randint(0, 256), 
    #                 np.random.randint(0, 256), np.random.randint(0, 256)))
    #         image = transform(image)
    #         images.append(image)
    # print(len(items_id))
    # print(cnt)
    # images = torch.stack(images)
    # model.eval()
    # with torch.no_grad():
    #     embeds = model(images)
    # np.save('embed.npy', embeds.numpy())