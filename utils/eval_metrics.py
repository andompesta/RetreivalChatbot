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
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k, k=k))
    return eval_metrics
