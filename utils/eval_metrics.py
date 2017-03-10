import tensorflow as tf
import functools
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples




def create_evaluation_metrics():
    '''
    Create a dictionary containing the eval metrics
    '''
    def custom_streaming_sparse_recall_at_k(predictions, labels, k):
        '''
        customize my evaluation metric. preditctions has the shape [batch_size, num_class] -> [16, 10]
        so we have to change the labels shape [160, 1] -> [16, 10]

        :param predictions: shaped probability for each class
        :param labels: unshape lebels
        :param k: to k classes
        :return: recall_at_k metrics for evaluation
        '''
        split_labs = tf.split(labels, num_or_size_splits=10, axis=0)
        shaped_labels = tf.concat(split_labs, axis=1)
        return tf.contrib.metrics.streaming_sparse_recall_at_k(predictions, shaped_labels, k)

    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            custom_streaming_sparse_recall_at_k, k=k))
    return eval_metrics
