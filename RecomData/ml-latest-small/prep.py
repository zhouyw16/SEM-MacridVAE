import re
import time
# import requests
import pandas as pd


# read
ratings = pd.read_csv('ratings.csv', sep=',', header=0,
                      usecols=[0,1,2], names=['user','item','rating'])

# write
ratings.to_csv('prep/ratings.txt', index=False)


# categories
# movies = pd.read_csv('movies.csv', sep=',', header=0,
#                         usecols=[0,2], names=['item','cate'])
# categories = {'item':[], 'cate':[]}
# for index, item, cates in movies.itertuples():
#     categories['item'].append(item)
#     categories['cate'].append(cates.split('|')[0])
# pd.DataFrame(categories).to_csv('prep/categories.txt', index=False)
    



# link
# links = pd.read_csv('links.csv', sep=',', header=0,
#                     usecols=[0,2], names=['item', 'link'])
# covers = {'item': [], 'cover': []}
# pattern = re.compile('data-src="//(.*?)"')
# for item, link in zip(links['item'], links['link']):
#     print(item)
#     try:
#         url = 'https://www.themoviedb.org/movie/%d' % (link)
#         html = requests.get(url)
#         cover = pattern.search(html.text).group(1)
#     except:
#         print(item, link)
#         cover = ''
#     covers['item'].append(item)
#     covers['cover'].append(cover)
#     time.sleep(0.1)
# pd.DataFrame(covers).to_csv('prep/covers.txt', index=False)


# links = pd.read_csv('links.csv', sep=',', header=0,
#                     usecols=[0,2], names=['item', 'link'])
# covers = pd.read_csv('prep/covers.txt', keep_default_na=False)
# covers_ = {'item': [], 'cover': []}
# pattern = re.compile('data-src="//(.*?)"')
# for item, link, cover in zip(links['item'], links['link'], covers['cover']):
#     if not cover:
#         print(item)
#         try:
#             url = 'https://www.themoviedb.org/movie/%d' % (link)
#             html = requests.get(url)
#             cover_ = pattern.search(html.text).group(1)
#         except:
#             print(item, link)
#             cover_ = ''
#         time.sleep(0.1)
#     else:
#         cover_ = cover
#     covers_['item'].append(item)
#     covers_['cover'].append(cover_)
# pd.DataFrame(covers_).to_csv('prep/covers.txt', index=False)