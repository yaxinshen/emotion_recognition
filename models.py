"""Contains different architectures for the different DSN parts.

We define here the modules that can be used in the different parts of the DSN
model.
- shared encoder (dsn_cropped_linemod, dann_xxxx)
- private encoder (default_encoder)
- decoder (large_decoder, gtsrb_decoder, small_decoder)
"""
import tensorflow as tf

#from models.domain_adaptation.domain_separation
import utils

slim = tf.contrib.slim


################################################################################
# PRIVATE ENCODERS
################################################################################
def default_encoder(X, code_size, weight_decay=0.0):
  """Encodes the given images to codes of the given size.

  Args:
    X: a tensor of size [batch_size, dim].
    code_size: the number of hidden units in the code layer of the classifier.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    end_points: the code of the input.
  """
  end_points = {}
  with slim.arg_scope(
      [slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
      ):
    end_points['fc1'] = slim.fully_connected(X, 256, scope='fc1')
    end_points['fc2'] = slim.fully_connected(
        end_points['fc1'], 128, scope='fc2')
    end_points['fc3'] = slim.fully_connected(
        end_points['fc2'], 100, scope='fc3')
    
  return end_points


################################################################################
# DECODERS
################################################################################
def small_decoder(codes,
                  dim,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size.

  Args:
    codes: a tensor of size [batch_size, code_size].
    dim: orignal size
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
      [slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
      ):
    net = slim.fully_connected(codes, 128, scope='fc1')
    net = slim.fully_connected(net, 256, scope='fc2')
    net = slim.fully_connected(net, dim, scope='fc3')
    
  return net


################################################################################
# SHARED ENCODERS
################################################################################
def dann_eeg(X,
             weight_decay=0.0,
             prefix='model',
             num_classes=3,
             **kwargs):
  """Creates a eeg model.

   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    X: the DE eeg, a tensor of size [batch_size, 310].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """
  end_points = {}

  with slim.arg_scope(
      [slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
      ):
    end_points['fc1'] = slim.fully_connected(X, 200, scope='fc1')
    end_points['fc2'] = slim.fully_connected(
        end_points['fc1'], 200, scope='fc2')
    end_points['fc3'] = slim.fully_connected(
        end_points['fc2'], 100, scope='fc3')
    end_points['fc4'] = slim.fully_connected(
        end_points['fc3'], 100, scope='fc4')
      
  logits = slim.fully_connected(
      end_points['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, end_points