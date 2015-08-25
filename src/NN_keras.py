from __future__ import division
import graphlab as gl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
np.random.seed(42)

train = gl.SFrame('../data/train_1')

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


X = train[features].to_dataframe()
# print X.shape
#
X_test = hold[features].to_dataframe()

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

score = []

for target in ['HandStart',
               'FirstDigitTouch',
               'BothStartLoadPhase',
               'LiftOff',
               'Replace',
               'BothReleased']:

  dims = X.shape[1]
  print(dims, 'dims')

  print("Building model...")

  model = Sequential()
  model.add(Dense(dims, 512, init='glorot_uniform'))
  model.add(PReLU((512,)))
  model.add(BatchNormalization((512,)))
  model.add(Dropout(0.5))

  model.add(Dense(512, 512, init='glorot_uniform'))
  model.add(PReLU((512,)))
  model.add(BatchNormalization((512,)))
  model.add(Dropout(0.5))

  model.add(Dense(512, 512, init='glorot_uniform'))
  model.add(PReLU((512,)))
  model.add(BatchNormalization((512,)))
  model.add(Dropout(0.5))

  model.add(Dense(512, 2, init='glorot_uniform'))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer="adam")

  print("Training model...")

  model.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.2)

  proba = model.predict_proba(X_test)
  # print clf.summary()
  y = np.array(train[target]).astype(np.int32)

  prediction = model.predict_proba(X_test)[:, 0]
  y_true = list(hold[target])
  score += [roc_auc_score(y_true, prediction)]

print np.mean(score)