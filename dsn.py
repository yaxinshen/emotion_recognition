"""Functions to create a DSN model and add the different losses to it.

Specifically, in this file we define the:
  - Shared Encoding Similarity Loss Module, with:
    - The Gradient Reversal (Domain-Adversarial) method
  - Difference Loss Module
  - Reconstruction Loss Module
  - Task Loss Module
"""
from functools import partial

import tensorflow as tf

import losses
import models
import utils

slim = tf.contrib.slim

#create model
def create_model(source_input, source_labels, target_input,
                 target_labels, similarity_loss, params, basic_tower_name):

  network = getattr(models, basic_tower_name)
  #num_classes = 3
  network = partial(network,num_classes=3)
  
  source_endpoints = add_task_loss(source_input, source_labels, network, params)
  
  with tf.variable_scope('towers', reuse=True):
    target_logits, target_endpoints = network(
        target_input, weight_decay=params['weight_decay'], prefix='target')
  
  target_accuracy = utils.accuracy(
      tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
  
  tf.summary.scalar('eval/Target accuracy', target_accuracy)
  
  source_shared = source_endpoints[params['layers_to_regularize']]
  target_shared = target_endpoints[params['layers_to_regularize']]  
  
  add_similarity_loss(similarity_loss, source_shared, target_shared, params)
  
  add_autoencoders(
      source_input,
      source_shared,
      target_input,
      target_shared,
      params=params,)

def add_similarity_loss(method_name,
                        source_samples,
                        target_samples,
                        params,
                        scope=None):
  """Adds a loss encouraging the shared encoding from each domain to be similar.

  Args:
    method_name: the name of the encoding similarity method to use. `dann_loss'
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    params: a dictionary of parameters. Expecting 'gamma_weight'.
    scope: optional name scope for summary tags.
  Raises:
    ValueError: if `method_name` is not recognized.
  """    
  weight = params['gamma_weight']
  method = getattr(losses,method_name)
  method(source_samples, target_samples, weight, scope)


def add_reconstruction_loss(recon_loss_name, data, recons, weight, domain):
  if recon_loss_name == 'sum_of_pairwise_squares':
    loss_fn = tf.contrib.losses.mean_pairwise_squared_error
  elif recon_loss_name == 'sum_of_squares':
    loss_fn = tf.contrib.losses.mean_squared_error

  loss = loss_fn(recons, slim.flatten(data), weight)
  assert_op = tf.Assert(tf.is_finite(loss), [loss])
  with tf.control_dependencies([assert_op]):
    tf.summary.scalar('losses/%s Recon Loss' % domain, loss)
    #try add
    #tf.losses.add_loss(loss)

def add_autoencoders(source_data, source_shared, target_data, target_shared, params):

  def normalize_input(data):
    data -= tf.reduce_min(data)
    return data / tf.reduce_max(data)
   
  def concat_operation(shared_repr, private_repr):
    return shared_repr + private_repr
   
  concat_layer = params['layers_to_regularize']
  
  difference_loss_weight = params['beta_weight']
  recon_loss_weight = params['alpha_weight']
  recon_loss_name = params['recon_loss_name']
  
  decoder_name = params['decoder_name']
  encoder_name = params['encoder_name']
  
  dim = source_data.shape[1]
  code_size = source_shared.get_shape().as_list()[-1]
  print('dim')
  print(dim)
  print('code_size')
  print(code_size)
  weight_decay = params['weight_decay']
  
  encoder_fn = getattr(models, encoder_name)
  
  with tf.variable_scope('source_encoder'):
    source_endpoints = encoder_fn(
        source_data, code_size, weight_decay=weight_decay)

  with tf.variable_scope('target_encoder'):
    target_endpoints = encoder_fn(
        target_data, code_size, weight_decay=weight_decay)
  
  decoder_fn = getattr(models, decoder_name)
  
  decoder = partial(
      decoder_fn,
      dim=dim,
      weight_decay = weight_decay)
  
  source_private = source_endpoints[concat_layer]
  target_private = target_endpoints[concat_layer]
  with tf.variable_scope('decoder'):
    source_recons = decoder(concat_operation(source_shared, source_private))

  with tf.variable_scope('decoder', reuse=True):
    source_private_recons = decoder(
        concat_operation(tf.zeros_like(source_private), source_private))
    source_shared_recons = decoder(
        concat_operation(source_shared, tf.zeros_like(source_shared)))

  with tf.variable_scope('decoder', reuse=True):
    target_recons = decoder(concat_operation(target_shared, target_private))
    target_shared_recons = decoder(
        concat_operation(target_shared, tf.zeros_like(target_shared)))
    target_private_recons = decoder(
        concat_operation(tf.zeros_like(target_private), target_private)) 
  
  losses.difference_loss(
      source_private,
      source_shared,
      weight=difference_loss_weight,
      name='Source')
  losses.difference_loss(
      target_private,
      target_shared,
      weight=difference_loss_weight,
      name='Target')

  add_reconstruction_loss(recon_loss_name, source_data, source_recons,
                          recon_loss_weight, 'source')
  add_reconstruction_loss(recon_loss_name, target_data, target_recons,
                          recon_loss_weight, 'target')  

def add_task_loss(source_input, source_labels, basic_tower, params):
  with tf.variable_scope('towers'):
    source_logits, source_endpoints = basic_tower(
        source_input, weight_decay=params['weight_decay'], prefix='Source')
  #print(source_labels.shape)
  #print(source_logits.shape)
  classification_loss = tf.losses.softmax_cross_entropy(
      source_labels, source_logits)
  #try add
  #tf.losses.add_loss(classification_loss)
  tf.summary.scalar('losses/classification_loss', classification_loss)
  return source_endpoints            
