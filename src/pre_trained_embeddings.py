from keras.layers import Embedding
from keras.models import Sequential
import simple_gem
import numpy as np

# our vocabulary
vocab = ['a', 'man', 'ran']
word2ix = {w: i for i, w in enumerate(vocab)}

# glov dictionary
glove_840B = simple_gensim.KeyedVectors.load_word2vec_format('/scratch/glove.840B.300d.txt')


glov_subset = np.array([
  glove_840B[w] if w in glove_840B else np.zeros(300)
  for w in vocab
])

# save this:
np.save(glov_subset, 'glov_subset.npy')

# load from saved
#glov_subset = np.load('glov_subset.npy')

# embedding initializer
def glove_init(shape, dtype=None):
  return glov_subset

# an example of a model in keras with pre-trained embeddings
model = Sequential([
  Embedding(len(word2ix), 300, embeddings_initializer=glove_init, trainable=False)
])

