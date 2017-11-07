# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=line-too-long
"""Evaluation for Domain Separation Networks (DSNs)."""
# pylint: enable=line-too-long
import math

import numpy as np
import tensorflow as tf

import ast
import losses
import models

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_integer('target_num', 0,
                            'Choose the ith subject as target data.')

tf.app.flags.DEFINE_string('master', '',
                           'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('checkpoint_dir', 'da/',
                           'Directory where the model was written to.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'da/',
    'Directory where we should write the tf summaries to.')

tf.app.flags.DEFINE_integer('num_examples', 3000, 'Number of test examples.')

tf.app.flags.DEFINE_string('basic_tower', 'dann_eeg',
                           'The basic tower building block.')

tf.app.flags.DEFINE_bool('enable_precision_recall', False,
                         'If True, precision and recall for each class will '
                         'be added to the metrics.')


t_file_name = 'data/source%d' % (FLAGS.target_num)
t_file = open(t_file_name)

def get_target(bs):
  X = []
  Y = []
  for i in range(bs):
    for line in t_file:
      line = ast.literal_eval(line.strip())
      X.append(line[:-1])
      tmp = line[-1]
      if tmp==-1:
        tmp=2
      Y.append(tmp)
      break
  X = np.array(X)
  Y = np.array(Y)
  Y = tf.one_hot(Y, 3)
  X = X.astype('float32')
  if X.shape[0] == bs:
    return X,Y
  else:
    t_file.seek(0)
    return get_target(bs)

def quaternion_metric(predictions, labels):
  params = {'batch_size': FLAGS.batch_size}
  logcost = losses.log_quaternion_loss_batch(predictions, labels, params)
  return slim.metrics.streaming_mean(logcost)


def angle_diff(true_q, pred_q):
  angles = 2 * (
      180.0 /
      np.pi) * np.arccos(np.abs(np.sum(np.multiply(pred_q, true_q), axis=1)))
  return angles




def main(_):
  g = tf.Graph()
  with g.as_default():
    # Load the data.
    Xt, Yt = get_target(FLAGS.batch_size)
    num_classes = 3

    # Define the model:
    with tf.variable_scope('towers'):
      basic_tower = getattr(models, FLAGS.basic_tower)
      predictions, endpoints = basic_tower(
          Xt,
          num_classes=num_classes,
          is_training=False)
    metric_names_to_values = {}

    # Define the metrics:
    
    accuracy = tf.contrib.metrics.streaming_accuracy(
        tf.argmax(predictions, 1), tf.argmax(Yt, 1))

    predictions = tf.argmax(predictions, 1)
    labels = tf.argmax(Yt, 1)
    metric_names_to_values['Accuracy'] = accuracy

    if FLAGS.enable_precision_recall:
      for i in xrange(num_classes):
        index_map = tf.one_hot(i, depth=num_classes)
        name = 'PR/Precision_{}'.format(i)
        metric_names_to_values[name] = slim.metrics.streaming_precision(
            tf.gather(index_map, predictions), tf.gather(index_map, labels))
        name = 'PR/Recall_{}'.format(i)
        metric_names_to_values[name] = slim.metrics.streaming_recall(
            tf.gather(index_map, predictions), tf.gather(index_map, labels))

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
        metric_names_to_values)

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
      op = tf.summary.scalar(metric_name, metric_value)
      op = tf.Print(op, [metric_value], metric_name)
      summary_ops.append(op)

    # This ensures that we make a single pass over all of the data.
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))

    # Setup the global step.
    slim.get_or_create_global_step()
    slim.evaluation.evaluation_loop(
        FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        summary_op=tf.summary.merge(summary_ops))


if __name__ == '__main__':
  tf.app.run()
