import re
import pandas as pd


file = open('Douban_br.sql', encoding='utf-8')
lines = file.readlines()
file.close()

ratings = {'user': [], 'item': [], 'rating': []}
p1 = re.compile('INSERT INTO `user_br` VALUES \(\'(\d+)\', \'.*?\', \'{(.*?)}\', \'.*?\'\)')
p2 = re.compile('u\\\\\'(\d+)\\\\\': u\\\\\'(\d)\\\\\'')
for line in lines:
    m1 = p1.search(line)
    if m1:
        rs = m1.group(2).split(', ')
        for r in rs:
            m2 = p2.search(r)
            ratings['user'].append(m1.group(1))
            ratings['item'].append(m2.group(1))
            ratings['rating'].append(m2.group(2))
pd.DataFrame(ratings).to_csv('prep/ratings.txt', index=False)