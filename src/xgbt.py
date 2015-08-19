from __future__ import division
__author__ = 'Vladimir Iglovikov'

#read train

import graphlab as gl

#For now I will only work with first person

train = gl.SFrame('../data/train.csv')

#validation set will be 8th series

train = train[train['subj'] == 1]
hold = train[train['series'] == 8]
training = train[train['series'] != 8]

print hold.column_names()