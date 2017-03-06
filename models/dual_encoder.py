import tensorflow as tf
import numpy as np
from models import helpers
import utils.IO_data as IO_data

FLAGS = tf.flags.FLAGS


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
        initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_embedding, hparams.embedding_dim)
        return tf.get_variable("word_embeddings",
                               initializer=initializer)
    else:                                                   # using random init
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)
        return tf.get_variable("word_embeddings",
                               shape=[hparams.vocab_size, hparams.embedding_dim],
                               initializer=initializer)


def dual_encoder_model(hparams, mode, context, context_len, utterance, utterance_len, targets):
    # Logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object

    # Initialize embedidngs randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(hparams)

    # Embed the context and the utterance, convert the word_idx to the word_embedding
    context_embedded = tf.nn.embedding_lookup(embeddings_W, context, name="embed_context")
    utterance_embedded = tf.nn.embedding_lookup(embeddings_W, utterance, name="embed_utterance")


    # Build the RNN
    with tf.variable_scope("rnn") as vs:    # define a scope where to share all the variable of the rnn network
        # We use an LSTM Cell
        cell = tf.contrib.rnn.LSTMCell(
            hparams.rnn_dim,
            forget_bias=2.0,            # init value of the ferget baias. it define the scale of forgetting at the beginning of the training
            use_peepholes=True,         # allow the usage of weighted peephole connections
            state_is_tuple=True)

        # Run the utterance and context through the RNN
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,
                                                    tf.concat([context_embedded, utterance_embedded], axis=0),
                                                    sequence_length=tf.concat([context_len, utterance_len], axis=0),
                                                    dtype=tf.float32)
        encoding_context, encoding_utterance = tf.split(rnn_states.h, num_or_size_splits=2, axis=0)

    with tf.variable_scope("prediction") as vs:
        M = tf.get_variable("M",
                            shape=[hparams.rnn_dim, hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())

        # "Predict" a  response: c * M
        generated_response = tf.matmul(encoding_context, M)             # execute a linear classification
        generated_response = tf.expand_dims(generated_response, 2)      # add a dimention at the end of the shape, generate_response.shape => [batch_size, rnn_dim, 1]
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)

        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response, encoding_utterance, transpose_a=True)
        logits = tf.squeeze(logits, [2])        # remove 3D dimention

        # Apply sigmoid to convert logits to probabilities
        probs = tf.sigmoid(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")

    return probs, mean_loss
