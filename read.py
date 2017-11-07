"""
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np 

data = sio.loadmat('EEG_X.mat')
data = data['X']
label = sio.loadmat('EEG_Y.mat')
label = label['Y']

for i in range(15):
  out_file = 'data/n%d' % (i)
  X = np.array(data[0][i])
  Y = np.array(label[0][i])
  F = np.hstack((X,Y))
  np.random.shuffle(F)
  print F.shape
  with open(out_file, "w") as f:
    for line in F:
      f.write('[')
      for l in line:
        f.write(str(l))
        f.write(',')
      f.write(']\n')
"""
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np 

data = sio.loadmat('EEG_X.mat')
data = data['X']
label = sio.loadmat('EEG_Y.mat')
label = label['Y']
for i in range(15):
  out_file = 'data/source%d' % (i)
  F = np.zeros(311)
  for j in range(15):
    if j==i:
      continue
    X = np.array(data[0][j])
    Y = np.array(label[0][j])
    Ftmp = np.hstack((X,Y))
    F = np.vstack((F,Ftmp))
  F = np.delete(F, 0, axis=0)
  np.random.shuffle(F)
  print F.shape
  with open(out_file, "w") as f:
    for line in F:
      f.write('[')
      for l in line:
        f.write(str(l))
        f.write(',')
      f.write(']\n')
