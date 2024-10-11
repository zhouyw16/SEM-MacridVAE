import os
import sys
import time

import numpy as np
import pandas as pd


# input data directory
try:
    dir = sys.argv[1]
    cut = int(sys.argv[2])
    os.chdir(os.path.join('.', dir, 'prep'))
except:
    print('please input a correct directory name and number')
    exit()
print('%s:' % dir)
print()


cates = pd.read_csv('cates.txt')
cates_delete = cates[cates['count'] < cut]
delete_cnt = cates_delete['count'].sum()
cates = cates[cates['count'] >= cut]
other_id = cates.shape[0]
cates = cates.append({'id': 'others', 'count': delete_cnt}, ignore_index=True)


categories = pd.read_csv('categorial.txt')
categories.loc[categories['cate'] >= other_id, 'cate'] = other_id
change_cnt = (categories['cate'] == other_id).sum()

if not os.path.exists('cate'): os.mkdir('cate')
cates.to_csv('cate/cates.txt', index=False)
categories.to_csv('cate/categorial.txt', index=False)

assert(delete_cnt == change_cnt)
