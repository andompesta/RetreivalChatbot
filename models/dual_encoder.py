import tensorflow as tf
import numpy as np
from models import helpers
import utils.IO_data as IO_data

FLAGS = tf.flags.FLAGS



def dual_encoder_model(hparams, mode, context, context_len, utterance, utterance_len, targets):
    # Initialize embedidngs randomly or with pre-trained vectors if available
    embeddings_W = helpers.get_embeddings(hparams)
    tf.summary.histogram('embeddings', embeddings_W)


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
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,                                                               # rnn_output[?, sequence_length, RNN_dim] => output of the rnn at each timestamp(for each word), rnn_state[?, RNN_DIM] => the last state for each example(sentence)
                                                    tf.concat([context_embedded, utterance_embedded], axis=0),          # encode all the context and then all the utterance of the current batch
                                                    sequence_length=tf.concat([context_len, utterance_len], axis=0),    # specify the length of each embdding to early stop
                                                    dtype=tf.float32)
        encoding_context, encoding_utterance = tf.split(rnn_states.h, num_or_size_splits=2, axis=0)                     # since we have the same amount of context and utterance, split the final output in half will give us the encoding of the contexts and of the utterances

    with tf.variable_scope("prediction") as vs:
        M = tf.get_variable("M",
                            shape=[hparams.rnn_dim, hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())

        # "Predict" a  response: c * M
        generated_response = tf.matmul(encoding_context, M)             # execute a linear classification
        # expand_dim used to guarantee the correct broadcasting during the computation of the logits
        generated_response = tf.expand_dims(generated_response, 2)      # add a dimention at the end of the shape, generate_response.shape => [batch_size, rnn_dim, 1]
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)      # add a dimention at the end of the shape, encoding_utterance.shape => [batch_size, rnn_dim, 1]

        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response, encoding_utterance, transpose_a=True)
        logits = tf.squeeze(logits, [2])        # remove 3D dimention,  logits.shape => [batch_size, 1]

        # Apply sigmoid to convert logits to probabilities
        probs = tf.sigmoid(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        # targets = tf.expand_dims(targets, 0)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")

    return probs, mean_loss
