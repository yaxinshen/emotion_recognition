from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
discriminant_analysis, random_projection)

t_file_name = 'data/n%d' % (0)
t_file = open(t_file_name)

s_file_name = 'data/source%d' % (0)
s_file = open(s_file_name)

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

data, label = get_target(1000)
n_samples, n_features = data.shape

def plot_embbedding(X, title=None):
  X_min, X_max = np.min(X,0), np.max(X,0)
  X = (X-X_min) / (X_max-X_min)
  plt.figure()
  ax = plt.subplot(111)
  for i in range(X.shape[0]):
    plt.text(X[i], str(label[i]),
    color = plt.cm.Set1(label[i]/3.),
    fontdict = {'weight': 'bold', 'size': 9})
  
