import tensorflow as tf
import pandas as pd
import array
from collections import defaultdict
import numpy as np
TEXT_FEATURE_SIZE = 160

def create_csv_batch_iter(file_name, path='../data', batch_size=10000):
    '''
    Read the csv input file into batched pandas dataframe
    :param file_name: file name
    :param path: path where is saved the file
    :param batch_size: batch dimention for reading the csv file
    :return: generator which return a batch at a time
    '''
    for df in pd.read_csv(path + '/' + file_name, chunksize=batch_size):
        yield df

def create_row_iter(batch_iter, func=lambda x:x):
    '''
    read each line of the input batches and apply a function
    :param batch_iter: gererator of the input batches
    :param func: function to apply
    :return: generator of each row
    '''
    for batch in batch_iter:
        for index, row in batch.iterrows():
            yield func(row)

def load_vocab(file_name, path='./data'):
    '''
    load the vocabulary build during the data preprocessing phase
    :param file_name: file name saved during the vocabulary creation
    :param path: path to the dir where the vocabulary is saved
    :return: list of vocab, dictionary word2idx
    '''
    vocab = None
    with open(path + '/' + file_name) as f:
        vocab = f.read().splitlines()
    dct = defaultdict(int)
    for idx, word in enumerate(vocab):
        dct[word] = idx
    return [vocab, dct]

def load_glove_vectors(file_name, vocab, path='./data'):
    '''
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    :param file_name: name of the file to load
    :param vocab: vocabulary build during the preprocessing phase
    :param path: path to the dir containing the file
    '''
    dct = {}
    vectors = array.array('d')  # create an empty array of double
    current_idx = 0
    with open(path + '/' + file_name, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:  # if the word is in my vocabulary
                dct[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1
        word_dim = len(entries)
        num_vectors = len(dct)
        tf.logging.info("Found {} out of {} vectors in Glove".format(num_vectors, len(vocab)))
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]

def get_feature_columns(mode):
    '''
    read the feature of the input example according to the running mode (label for training, distractor for eval)
    :param mode: running mode
    :return:
    '''
    feature_columns = []

    feature_columns.append(tf.contrib.layers.real_valued_column(column_name="context", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(column_name="utterance", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(column_name="context_len", dimension=1, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(column_name="utterance_len", dimension=1, dtype=tf.int64))

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        # During training we have a label feature
        feature_columns.append(tf.contrib.layers.real_valued_column(column_name="label", dimension=1, dtype=tf.int64))

    if mode == tf.contrib.learn.ModeKeys.EVAL:
        # During evaluation we have distractors
        for i in range(9):
            feature_columns.append(tf.contrib.layers.real_valued_column(column_name="distractor_{}".format(i), dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
            feature_columns.append(tf.contrib.layers.real_valued_column(column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))
    return set(feature_columns)


def create_input_fn(mode, input_files, batch_size, num_epochs):
    '''
    read the input file accodring to the mode which is running the model
    :param mode: running mode
    :param input_files: files to read
    :param batch_size: batch size
    :param num_epochs: number of epocs to repead the datasets, if None it never stop
    :return:
    '''
    def read_and_decode(file_name_queue):
        '''
        Decode a single sequence example
        :param file_name_queue: queue of the files to read
        :return: tensor containing the contex_feature and sequence_fature of a single example
        '''
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                                                           context_features={
                                                                               'context_len': tf.FixedLenFeature([], dtype=tf.int64),
                                                                               'utterance_len': tf.FixedLenFeature([], dtype=tf.int64),
                                                                               'label': tf.FixedLenFeature([], dtype=tf.int64),
                                                                           },
                                                                           sequence_features={
                                                                               "context": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                                               "utterance": tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                                           })
        return context_parsed, sequence_parsed

    def input_fn():
        with tf.name_scope('input'):
            file_name_queue = tf.train.string_input_producer(input_files, num_epochs=num_epochs)
            context_parsed, sequence_parsed = read_and_decode(file_name_queue)      # decode each example
            feature_map = dict(context_parsed, **sequence_parsed)                           # construct a feature map dictionary

            feature_map_batch = tf.train.batch(    # BATCH THE DATA
                tensors=feature_map,
                batch_size=batch_size,
                dynamic_pad=True)                                # dimanic size of the batch


        # This is an ugly hack because of a current bug in tf.learn
        # During evaluation TF tries to restore the epoch variable which isn't defined during training
        # So we define the variable manually here
        targets = feature_map_batch.pop("label")
        targets = tf.expand_dims(targets, 1)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            tf.get_variable(
                "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
                initializer=tf.constant(0, dtype=tf.int64))
        # elif mode == tf.contrib.learn.ModeKeys.EVAL:
        #     targets = tf.split(targets, num_or_size_splits=10, axis=0)        # needed for the evaluation of recall@_k
        #     targets = tf.concat(targets, axis=1)

        return feature_map_batch, targets
    return input_fn