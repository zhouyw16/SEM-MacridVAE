import pandas as pd


# read
ratings = pd.read_csv('ratings.dat', sep='::', header=None, engine='python',
                      usecols=[0,1,2], names=['user', 'item', 'rating'])

# write
ratings.to_csv('prep/ratings.txt', index=False)