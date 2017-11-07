from functools import partial
import tensorflow as tf

import grl_op_grads  # pylint: disable=unused-import
import grl_op_shapes  # pylint: disable=unused-import
import grl_ops
import utils
slim = tf.contrib.slim


################################################################################
# SIMILARITY LOSS
################################################################################
def dann_loss(source_samples, target_samples, weight, scope=None):
  """Adds the domain adversarial (DANN) loss.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """ 
  with tf.variable_scope('dann'):
    batch_size = tf.shape(source_samples)[0]
    samples = tf.concat(axis=0, values=[source_samples, target_samples])
    samples = slim.flatten(samples)

    domain_selection_mask = tf.concat(
        axis=0, values=[tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))])

    # Perform the gradient reversal and be careful with the shape.
    #print samples.dtype
    grl = grl_ops.gradient_reversal(samples)
    grl = tf.reshape(grl, (-1, samples.get_shape().as_list()[1]))

    grl = slim.fully_connected(grl, 100, scope='fc1')
    logits = slim.fully_connected(grl, 1, activation_fn=None, scope='fc2')

    domain_predictions = tf.sigmoid(logits)

  domain_loss = tf.losses.log_loss(
      domain_selection_mask, domain_predictions, weights=weight)

  domain_accuracy = utils.accuracy(
      tf.round(domain_predictions), domain_selection_mask)

  assert_op = tf.Assert(tf.is_finite(domain_loss), [domain_loss])
  with tf.control_dependencies([assert_op]):
    tag_loss = 'losses/domain_loss'
    tag_accuracy = 'losses/domain_accuracy'
    if scope:
      tag_loss = scope + tag_loss
      tag_accuracy = scope + tag_accuracy
    #try add
    #tf.losses.add_loss(domain_loss)
    tf.summary.scalar(tag_loss, domain_loss)
    tf.summary.scalar(tag_accuracy, domain_accuracy)

  return domain_loss
  

################################################################################
# DIFFERENCE LOSS
################################################################################
def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  """Adds the difference loss between the private and shared representations.

  Args:
    private_samples: a tensor of shape [num_samples, num_features].
    shared_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the incoherence loss.
    name: the name of the tf summary.
  """
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)

  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)

  correlation_matrix = tf.matmul(
      private_samples, shared_samples, transpose_a=True)

  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')

  tf.summary.scalar('losses/Difference Loss {}'.format(name),
                                       cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
    tf.losses.add_loss(cost)
