from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
Let's try to create global train
'''

import graphlab as gl
train_result = []

for subj in range(1, 13):
  for series in range(1, 9):
    print (subj, series)
    data = gl.SFrame('../data/train/subj{subj}_series{series}_data.csv'.format(subj=subj, series=series))
    events = gl.SFrame('../data/train/subj{subj}_series{series}_events.csv'.format(subj=subj, series=series))
    print 'data.shape', data.shape
    print 'events.shape', events.shape

    temp = data.join(events, on='id')
    temp['subj'] = subj
    temp['series'] = series
    print temp.shape
    train_result += [temp]

train_new = train_result[0]
for i in range(1, len(train_result)):
    train_new = train_new.append(train_result[i])

train_new['time'] = train_new.apply(lambda x: int(x['id'].split('_')[-1]))

train_new.save('../data/train.csv')