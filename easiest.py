#%matplotlib inline

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cPickle as pkl
import ast
from sklearn.datasets import make_moons, make_blobs
from sklearn.decomposition import PCA
from collections import defaultdict

from utils import *

train_file_name = 'data/Train.txt'
test_file_name = 'data/Test.txt'

train_file = open(train_file_name)
test_file = open(test_file_name)

curnum = 0
batch_size = 128


t_file_name = 'data/n%d' % (0)
t_file = open(t_file_name)

s_file_name = 'data/source%d' % (0)
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
  if X.shape[0] == bs:
    return X,Y
  else:
    t_file.seek(0)
    return get_target(bs)


def get_t_all():
    X = []
    Y = []
    size = 0
    test_file.seek(0)
    for line in t_file:
        line = ast.literal_eval(line.strip())
        X.append(line[:-1])
        Y.append(line[-1])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def build_model():
    X = tf.placeholder(tf.float32, [None, 310], name='X') # Input data
    Y_ind = tf.placeholder(tf.int32, [None], name='Y_ind')  # Class index
    
    Y = tf.one_hot(Y_ind, 3)
    
    # Feature extractor
    W0 = weight_variable([310, 256])
    b0 = bias_variable([256])
    fc0 = tf.nn.relu(tf.matmul(X, W0) + b0)
    
    W1 = weight_variable([256, 128])
    b1 = bias_variable([128])
    fc1 = tf.nn.relu(tf.matmul(fc0, W1) + b1)
    
    W2 = weight_variable([128, 100])
    b2 = bias_variable([100])
    fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
    
    W3 = weight_variable([100, 100])
    b3 = bias_variable([100])
    fc3 = tf.nn.relu(tf.matmul(fc2, W3) + b3)
    
    W4 = weight_variable([150, 100])
    b4 = bias_variable([100])
    F = tf.nn.relu(tf.matmul(fc3, W4) + b4, name = 'feature')
    
    W4l = weight_variable([100, 3])
    b4l = bias_variable([3])
    p_logit = tf.matmul(F, W4l) + b4l
    p = tf.nn.softmax(p_logit)
    p_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = p_logit)
    
    pred_loss = tf.reduce_sum(p_loss, name = 'pred_loss')
    
    pred_train_op = tf.train.AdamOptimizer(0.001).minimize(pred_loss, name='pred_train_op')
    
    # Evaluation
    p_acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(p, 1)), tf.float32), name='p_acc')

build_model()
sess = tf.InteractiveSession()  

def train_and_evaluate(sess, train_op_name, train_loss_name, grad_scale=None, num_batches=5000, verbose=True):
    
    
    # Get output tensors and train op
    p_acc = sess.graph.get_tensor_by_name('p_acc:0')
    train_loss = sess.graph.get_tensor_by_name(train_loss_name + ':0')
    train_op = sess.graph.get_operation_by_name(train_op_name)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    #saver.restore(sess, 'model/e-dann.model-2500')
    
    ckpt = tf.train.get_checkpoint_state('./model_e')
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("checkpoint not found!!")
        
    
    for i in range(num_batches):
        
        X0, y0 = get_sbatch(batch_size)
        X1, y1 = get_tbatch(batch_size)

        _, loss, pa = sess.run([train_op, train_loss, p_acc],
                                   feed_dict={'X:0': X0, 'Y_ind:0': y0})

        if verbose and i % 50 == 0:
              
            pat = sess.run([d_acc, p_acc], feed_dict={'X:0': X1, 'Y_ind:0': y1})
            print 'loss: %f, s_a: %f, t_a: %f' % (loss, pa, pat)   
            if verbose and i % 200 == 0:
                path = saver.save(sess, 'model/e-dann.model', global_step=i)
                #print(path)
      
   
    Xt,Yt = get_t_all()
    Xs,Ys = get_sbatch(20000) 
    #Xt,Yt = get_tbatch(40000)     
    # Get final accuracies on whole dataset
    das, pas = sess.run([d_acc, p_acc], feed_dict={'X:0': Xs, 'Y_ind:0': Ys, 
                            'D_ind:0': np.zeros(Xs.shape[0], dtype=np.int32), 'train:0': False, 'l:0': 1.0})
    dat, pat = sess.run([d_acc, p_acc], feed_dict={'X:0': Xt, 'Y_ind:0': Yt,
                            'D_ind:0': np.ones(Xt.shape[0], dtype=np.int32), 'train:0': False, 'l:0': 1.0})

    print 'Source domain: ', das
    print 'Source class: ', pas
    print 'Target domain: ', dat
    print 'Target class: ', pat  
train_and_evaluate(sess, 'pred_train_op', 'pred_loss')
#train_and_evaluate(sess, 'dann_train_op', 'total_loss')
