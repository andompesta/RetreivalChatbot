import tensorflow as tf
import sys

def get_id_feature(features, key, len_key):
    ids = features[key]
    ids_len = features[len_key]
    return ids, ids_len

def create_train_op(loss, hparams):
    '''
    Function used to train the model
    :param loss: loss function to evaluate the error
    :param hparams: hiper-parameters used to configure the optimizer
    :return: training updates
    '''
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,                                          # loss function used
        global_step=tf.contrib.framework.get_global_step(), # number of batches seen so far
        learning_rate=hparams.learning_rate,                # learning rate
        clip_gradients=10.0,                                # clip gradient to a max value
        optimizer=hparams.optimizer)                        # optimizer used
    return train_op


def create_model_fn(hparams, model_impl):
    '''
    Function used to create the model according different implementations and usage mode
    :param hparams: hiper-parameters used to configure the model
    :param model_impl: implementation of the model used, have to use the same interface to inject a different model
    :return: probabilities of the predicted class, value of the loss function, operation to execute the training
    '''
    def model_fn(features, targets, mode):
        context, context_len = get_id_feature(features, "context", "context_len")
        utterance, utterance_len = get_id_feature(features, "utterance", "utterance_len")

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                utterance,
                utterance_len,
                targets)
            train_op = create_train_op(loss, hparams)
            return probs, loss, train_op

        if mode == tf.contrib.learn.ModeKeys.INFER:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                utterance,
                utterance_len,
                None)
            return probs, 0.0, None

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                utterance,
                utterance_len,
                targets)

            split_probs = tf.split(probs, num_or_size_splits=10, axis=0)    # split the probabilities between the first positive utterance and the following 9 negative distractors
            shaped_probs = tf.concat(split_probs, axis=1)                   # matrix with shape(?, 10) for each example we have the probability of each response

            # Add summaries
            tf.summary.histogram("eval_correct_probs_hist", split_probs[0])
            tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))
            tf.summary.histogram("eval_incorrect_probs_hist", split_probs[1])
            tf.summary.scalar("eval_incorrect_probs_average", tf.reduce_mean(split_probs[1]))

            return shaped_probs, loss, None
    return model_fn
