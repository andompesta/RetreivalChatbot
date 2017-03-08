import time
from tensorflow.contrib.learn import MetricSpec
import functools
import tensorflow as tf
import model

import utils.IO_data as IO_data
import utils.network_params as network_params
from models.dual_encoder import dual_encoder_model
from utils.eval_metrics import create_evaluation_metrics

TIMESTAMP = int(time.time())

tf.flags.DEFINE_string("input_dir", './data', "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", './runs_{}'.format(TIMESTAMP), "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
tf.flags.DEFINE_float("memory_fraction", 0., "fraction of gpu memory allocated")
FLAGS = tf.flags.FLAGS


TRAIN_FILE = FLAGS.input_dir + '/train_my.tfrecords'
VALIDATION_FILE =  FLAGS.input_dir + '/validation_my.tfrecords'

tf.logging.set_verbosity(FLAGS.loglevel)

def main():
    hparams = network_params.create_hparams()

    model_fn = model.create_model_fn(           # create the model
        hparams,
        model_impl=dual_encoder_model)

    estimator = tf.contrib.learn.Estimator(     # creation of the estimator for our model_fn functions
        model_fn=model_fn,                      # model function
        model_dir=FLAGS.model_dir,              # directory to save the model paramters, graphs, etc.
        config=tf.contrib.learn.RunConfig(gpu_memory_fraction=FLAGS.memory_fraction))   # specify the ammount of memory to use for the GPU

    input_fn_train = IO_data.create_input_fn(   # ensure that the model receive the data in the correct format for training
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        input_files=[TRAIN_FILE],
        batch_size=hparams.batch_size,
        num_epochs=FLAGS.num_epochs)            # Integer specifying the number of times to read through the dataset.

    input_fn_eval = IO_data.create_input_fn(    # ensure that the model receive the data in the correct format for evaluation
        mode=tf.contrib.learn.ModeKeys.EVAL,
        input_files=[VALIDATION_FILE],
        batch_size=hparams.eval_batch_size,
        num_epochs=1)

    eval_metrics = create_evaluation_metrics()

    eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(     # creation of the monitor which evaluate the model every eval_every step during training
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

    estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])

if __name__ == '__main__':
    main()
