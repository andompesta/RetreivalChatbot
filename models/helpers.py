import numpy as np
import tensorflow as tf
import utils.IO_data as IO_data


def get_id_feature(features, key, len_key):
    ids = features[key]
    ids_len = features[len_key]
    return ids, ids_len

def get_embeddings(hparams):
    '''
    generate initial word embeddings according to the hparams flags
    :param hparams: flags
    :return: get or create the varaible word_embedding
    '''
    if hparams.glove_path and hparams.vocab_path:           # using Glove embedding
        tf.logging.info("Loading Glove embeddings...")
        vocab_array, vocab_dict = IO_data.load_vocab(hparams.vocab_path)
        glove_embedding, glove_dict = IO_data.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        initializer = build_initial_embedding_matrix(vocab_dict, glove_dict, glove_embedding, hparams.embedding_dim)
        return tf.get_variable("word_embeddings",
                               initializer=initializer)
    else:                                                   # using random init
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)
        return tf.get_variable("word_embeddings",
                               shape=[hparams.vocab_size, hparams.embedding_dim],
                               initializer=initializer)


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