import pandas as pd


# read
ratings = pd.read_csv('movie-ratings.txt', sep=',', header=None, 
                      usecols=[0,1,4], names=['user', 'item', 'rating'])
trusts  = pd.read_csv('trusts.txt', sep=',', header=None, 
                      usecols=[0,1], names=['trustor', 'trustee'])

# write
ratings.to_csv('prep/ratings.txt', index=False)
trusts.to_csv('prep/trusts.txt', index=False)