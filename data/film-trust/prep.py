import pandas as pd


# read
ratings = pd.read_csv('ratings.txt', sep=' ', header=None,
                      names=['user', 'item', 'rating'])
trusts  = pd.read_csv('trust.txt', sep=' ', header=None, 
                      usecols=[0,1], names=['trustor', 'trustee'])

# write
ratings.to_csv('prep/ratings.txt', index=False)
trusts.to_csv('prep/trusts.txt', index=False)