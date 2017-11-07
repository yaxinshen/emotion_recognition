"""Training for DSNs"""
from __future__ import division

import tensorflow as tf
import numpy as np

import ast

import dsn

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_integer('target_num', 0,
                            'Choose the ith subject as target data.')

tf.app.flags.DEFINE_string('train_log_dir', 'da/',
                           'Directory where to write event logs.')

tf.app.flags.DEFINE_string(
    'layers_to_regularize', 'fc3',
    'Comma-separated list of layer names to use MMD regularization on.')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'The learning rate')

tf.app.flags.DEFINE_float('alpha_weight', 0,
                          'The coefficient for scaling the reconstruction '
                          'loss.')

tf.app.flags.DEFINE_float(
    'beta_weight', 0,
    'The coefficient for scaling the private/shared difference loss.')

tf.app.flags.DEFINE_float(
    'gamma_weight', 0,
    'The coefficient for scaling the shared encoding similarity loss.')

tf.app.flags.DEFINE_float(
    'weight_decay', 1e-6,
    'The coefficient for the L2 regularization applied for all weights.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 5,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 30,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None,
    'The maximum number of gradient steps. Use None to train indefinitely.')


tf.app.flags.DEFINE_float('decay_rate', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_integer('decay_steps', 5000, 'Learning rate decay steps.')

tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum value.')

tf.app.flags.DEFINE_string('decoder_name', 'small_decoder',
                           'The decoder to use.')
tf.app.flags.DEFINE_string('encoder_name', 'default_encoder',
                           'The encoder to use.')

################################################################################
# Flags that control the architecture and losses
################################################################################
tf.app.flags.DEFINE_string(
    'similarity_loss', 'dann_loss',
    'The method to use for encouraging the common encoder codes to be '
    'similar, one of "grl", "mmd", "corr".')

tf.app.flags.DEFINE_string('recon_loss_name', 'sum_of_pairwise_squares',
                           'The name of the reconstruction loss.')

tf.app.flags.DEFINE_string('basic_tower', 'dann_eeg',
                           'The basic tower building block.')

t_file_name = 'data/n%d' % (FLAGS.target_num)
t_file = open(t_file_name)

s_file_name = 'data/source%d' % (FLAGS.target_num)
s_file = open(s_file_name)

def get_source(bs):
  X = []
  Y = []
  for i in range(bs):
    for line in s_file:
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
    s_file.seek(0)
    return get_source(bs)
    
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

def main(_):
  model_params = {
      'layers_to_regularize': FLAGS.layers_to_regularize,
      'alpha_weight': FLAGS.alpha_weight,
      'beta_weight': FLAGS.beta_weight,
      'gamma_weight': FLAGS.gamma_weight,
      'recon_loss_name': FLAGS.recon_loss_name,
      'decoder_name': FLAGS.decoder_name,
      'encoder_name': FLAGS.encoder_name,
      'weight_decay': FLAGS.weight_decay,
      'batch_size': FLAGS.batch_size,
  }
  
  g = tf.Graph()
  with g.as_default():
    Xs, Ys = get_source(FLAGS.batch_size)
    Xt, Yt = get_target(FLAGS.batch_size)

    slim.get_or_create_global_step()
    dsn.create_model(
        Xs,
        Ys,
        Xt,
        Yt,
        FLAGS.similarity_loss,
        model_params,
        basic_tower_name=FLAGS.basic_tower)
    
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        slim.get_or_create_global_step(),
        FLAGS.decay_steps,
        FLAGS.decay_rate,
        staircase=True,
        name='learning_rate')
    
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('total_loss', tf.losses.get_total_loss())
    
    #opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    opt = tf.train.AdamOptimizer(learning_rate)
    tf.logging.set_verbosity(tf.logging.INFO)
    #run training
    loss_tensor = slim.learning.create_train_op(
        slim.losses.get_total_loss(),
        #tf.losses.get_total_loss(),
        opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True)
    slim.learning.train(
        train_op=loss_tensor,
        logdir=FLAGS.train_log_dir,
        master="",
        is_chief=1,
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run()
