import pandas as pd


# read
ratings = pd.read_csv('ratings.csv', sep=',', header=0,
                      usecols=[0,1,2], names=['user', 'item', 'rating'])

# write
ratings.to_csv('prep/ratings.txt', index=False)