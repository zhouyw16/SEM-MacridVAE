import pandas as pd


# pd.read_csv() cannot read lines with special characters
file = open('MovieRatings.txt', encoding='utf-8')
lines = file.readlines()
file.close()
ratings = {'user': [], 'item': [], 'rating': []}
for line in lines:
    record = line.split()
    ratings['user'].append(record[0])
    ratings['item'].append(record[1])
    ratings['rating'].append(record[2])
pd.DataFrame(ratings).to_csv('prep/ratings.txt', index=False)


trusts  = pd.read_csv('relation.txt', sep='\t', header=None, 
                      usecols=[0,1], names=['trustor', 'trustee'])
trusts.to_csv('prep/trusts.txt', index=False)