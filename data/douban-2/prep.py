import re
import pandas as pd


file = open('douban_20100129_1108.sql', encoding='utf-8')
lines = file.readlines()
file.close()

ratings = {'user': [], 'item': [], 'rating': []}
p = re.compile('\(\d+,(\d+),(\d+),\'watched\',\'.*?\',\'.*?\',\'.*?\',(\d)\)')
for line in lines:
    m = p.search(line)
    if m:
        ratings['user'].append(m.group(1))
        ratings['item'].append(m.group(2))
        ratings['rating'].append(m.group(3))
pd.DataFrame(ratings).to_csv('prep/ratings.txt', index=False)


trusts = {'trustor': [], 'trustee': []}
p = re.compile('\(\d+,(\d+),(\d+),\'friends\'\)')
for line in lines:
    m = p.search(line)
    if m:
        trusts['trustor'].append(m.group(1))
        trusts['trustee'].append(m.group(2))
pd.DataFrame(trusts).to_csv('prep/trusts.txt', index=False)