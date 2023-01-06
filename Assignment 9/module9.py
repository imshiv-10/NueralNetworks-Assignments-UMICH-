from imdb import Imdb
imdb=Imdb()
imdb._load(num_words=10000)
imdb.train(epochs=20)
imdb.evaluate()


#%%
from reuters import Reuters
reuters=Reuters()
reuters._load(num_words=10000)
reuters.train(epochs=30)
reuters.evaluate()
#%%
# Boston Housing price analysis

from boston_housing import Boston_Housing
boston_housing=Boston_Housing()
boston_housing._load()
boston_housing.train(epochs=30)
boston_housing.evaluate()