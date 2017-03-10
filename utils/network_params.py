import tensorflow as tf
from collections import namedtuple


TESTING_EXAMPLE_SIZE = 10           # during testing/validation every example has 1 positive utterance and 9 negative distractors


# Model Parameters
tf.flags.DEFINE_integer("vocab_size", 91620, "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", 'glove.6B.100d.txt', "file name to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", 'vocabulary_my.txt', "file name to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16 * TESTING_EXAMPLE_SIZE, "Batch size during evaluation")          # every batch have to be of 16 * 10. Every testing example have to be composed by 1 positive and 10 negative example
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "embedding_dim",
        "eval_batch_size",
        "learning_rate",
        "optimizer",
        "rnn_dim",
        "vocab_size",
        "glove_path",
        "vocab_path"
    ])

def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        glove_path=FLAGS.glove_path,
        vocab_path=FLAGS.vocab_path,
        rnn_dim=FLAGS.rnn_dim)