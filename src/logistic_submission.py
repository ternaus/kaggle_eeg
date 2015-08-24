from __future__ import division
__author__ = 'Vladimir Iglovikov'
'''
This script uses logistic regression on the raw data to do prediction using Logistic regression
'''


#read train

import graphlab as gl
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import sys
#For now I will only work with first person

submission = []

for subject in range(1, 13):
  print 'subject = ', subject
  train = gl.SFrame('../data/train_{subject}'.format(subject=subject))
  test = gl.SFrame('../data/test_{subject}'.format(subject=subject))


  temp = gl.SFrame()

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
              'PO10',
              'time',
              'subj',
              'series']

  temp['id'] = test['id']

  for target in ['HandStart',
                 'FirstDigitTouch',
                 'BothStartLoadPhase',
                 'LiftOff',
                 'Replace',
                 'BothReleased']:

    clf = gl.logistic_classifier.create(train, target=target, features=features, validation_set=None)

    temp[target] = clf.predict(test, output_type='probability')

  submission += [temp.to_dataframe()]

submission = pd.concat(submission)

print 'save submission to file'
submission.to_csv('predictions/LG_raw.csv', index=False)