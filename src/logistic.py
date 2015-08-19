from __future__ import division
__author__ = 'Vladimir Iglovikov'

#read train

import graphlab as gl
from lasagne.nonlinearities import softmax
from sklearn.metrics import roc_auc_score


import sys
#For now I will only work with first person

train = gl.SFrame('../data/train_1')

print train.shape
#validation set will be 8th series

# train = train[train['subj'] == 1]

#let's prepare human being train
# def create_y(x):
#   temp = [x['HandStart'], x['FirstDigitTouch'], x['BothStartLoadPhase'], x['LiftOff'], x['Replace'], x['BothReleased']]
#   if sum(temp) == 0:
#     return 6
#   else:
#     return temp.index(1)
#
# train['y'] = train.apply(lambda x: create_y(x))

print train.shape

hold = train[train['series'] == 8]
print 'hold'
print hold.shape

training = train[train['series'] != 8]
print 'training'
print training.shape

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

print training.shape
# X = training[features]
# print X.shape
#
# X_test = hold[features]

from sklearn.metrics import roc_auc_score

import numpy as np
# from sklearn.preprocessing import StandardScaler
import math
import pandas as pd


clf = gl.logistic_classifier.create(training, target='HandStart', features=features)

prediction = clf.predict(hold, output_type='probability')

print roc_auc_score(hold['HandStart'], prediction)
#
# y_test = hold[['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']].to_dataframe()
#
# net1.fit(X, y)
#
# prediction = net1.predict_proba(X_test)
#
# def score(y_test, prediction):
#   result = []
#   result += [roc_auc_score(y_test['HandStart'], prediction[:, 0])]
#   result += [roc_auc_score(y_test['FirstDIgitTouch'], prediction[:, 1])]
#   result += [roc_auc_score(y_test['BothStartLoadPhase'], prediction[:, 2])]
#   result += [roc_auc_score(y_test['LiftOff'], prediction[:, 3])]
#   result += [roc_auc_score(y_test['Replace'], prediction[:, 4])]
#   result += [roc_auc_score(y_test['BothReleased'], prediction[:, 5])]
#   return np.mean(result)
#
# print score(y_test, prediction[:6, ])
#
