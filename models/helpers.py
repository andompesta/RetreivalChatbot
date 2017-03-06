import array
import numpy as np
import tensorflow as tf
from collections import defaultdict


def build_initial_embedding_matrix(vocab_dict, glove_dict, glove_embedding, embedding_dim):
    '''
    Function used to load the Glove word2vec embedding as input for the word present both in our embedding and in the Glove embedding
    :param vocab_dict:      our dataset vocabulary2word_idx
    :param glove_dict:      Glove vocabulary2word_idx should be the same of the vocab_dict
    :param glove_embedding: Glove embedding
    :param embedding_dim:   Glove embedding dimension
    :return:
    '''
    initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
    for word, glove_word_idx in glove_dict.items():
        word_idx = vocab_dict.get(word)
        initial_embeddings[word_idx, :] = glove_embedding[glove_word_idx]
    return initial_embeddings