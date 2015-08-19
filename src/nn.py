from __future__ import division
__author__ = 'Vladimir Iglovikov'

#read train

import graphlab as gl
from lasagne.nonlinearities import softmax
from sklearn.metrics import roc_auc_score


import sys
#For now I will only work with first person

train = gl.SFrame('../data/train_1')

#validation set will be 8th series

# train = train[train['subj'] == 1]

#let's prepare human being train
def create_y(x):
  temp = [x['HandStart'], x['FirstDigitTouch'], x['BothStartLoadPhase'], x['LiftOff'], x['Replace'], x['BothReleased']]
  if sum(temp) == 0:
    return 6
  else:
    return temp.index(1)

train['y'] = train.apply(lambda x: create_y(x))

hold = train[train['series'] == 8]
training = train[train['series'] != 8]

features = ['Fp1',
            'Fp2',
            'F7',
            'F3',
            'Fz',
            'F4',
            'F8',
            'FC5',
            'FC1',
            'FC2',
            'FC6',
            'T7',
            'C3',
            'Cz',
            'C4',
            'T8',
            'TP9',
            'CP5',
            'CP1',
            'CP2',
            'CP6',
            'TP10',
            'P7',
            'P3',
            'Pz',
            'P4',
            'P8',
            'PO9',
            'O1',
            'Oz',
            'O2',
            'PO10']

print X.shape
X = training[features].to_dataframe()
print X.shape

X_test = hold[features].to_dataframe()


from sklearn.metrics import roc_auc_score
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
import numpy as np
from lasagne import layers
from lasagne.layers import DropoutLayer
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd
__author__ = 'Vladimir Iglovikov'

def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)



net1 = NeuralNet(
      layers=[  # three layers: one hidden layer
          ('input', layers.InputLayer),
          # ('dropout1', DropoutLayer),
          ('hidden0', layers.DenseLayer),
          # ('dropout2', DropoutLayer),
          # ('hidden1', layers.DenseLayer),
          ('output', layers.DenseLayer),
          ],
      # layer parameters:
      input_shape=(None, training.shape[1]),
      # dropout1_p=0.1,
      hidden0_num_units=100,  # number of units in hidden layer
      # dropout2_p=0.3,
      # hidden1_num_units=400,  # number of units in hidden layer
      # output_nonlinearity=None,  # output layer uses identity function
      output_num_units=7,  # 1 target values
      outout_nonlinearity=softmax,

      # optimization method:
      update=nesterov_momentum,
      # update_learning_rate=0.001,
      # update_momentum=0.9,
      update_momentum=theano.shared(float32(0.9)),

      eval_size=0.2,
      max_epochs=100,  # we want to train this many epochs
      update_learning_rate=theano.shared(float32(0.03)),
      verbose=1,
      regression=True,  # flag to indicate we're dealing with regression problem
      # on_epoch_finished=[
      #                 AdaptiveVariable('update_learning_rate', start=0.001, stop=0.00001),
      #                 AdjustVariable('update_momentum', start=0.9, stop=0.999),
      #                 EarlyStopping(),
      #             ]
      )

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
X_test = scaler.transform(X_test)
y = training['y'].astype(np.int32)

y_test = hold[['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']]

net1.fit(X, y)

prediction = net1.predict_proba(X_test)

def score(y_test, prediction):
  result = []
  result += [roc_auc_score(y_test['HandStart'], prediction[:, 0])]
  result += [roc_auc_score(y_test['FirstDIgitTouch'], prediction[:, 1])]
  result += [roc_auc_score(y_test['BothStartLoadPhase'], prediction[:, 2])]
  result += [roc_auc_score(y_test['LiftOff'], prediction[:, 3])]
  result += [roc_auc_score(y_test['Replace'], prediction[:, 4])]
  result += [roc_auc_score(y_test['BothReleased'], prediction[:, 5])]
  return np.mean(result)

print score(y_test, prediction[:6, ])

