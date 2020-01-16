#!/usr/bin/python
# # -*- coding=utf-8 -*-

import random
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Activation,Dropout,Dense,LSTM,Conv1D,MaxPool1D,Flatten
from common_func import loss_history,evaluate_method,read_data
from keras import optimizers
from sklearn.model_selection import KFold

#read train data
np.random.seed(6)
train_x, train_y_1D = read_data.read_data('train_data_yongxin.csv')
test_x, test_y_1D = read_data.read_data('test_data_yongxin.csv')
train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)

train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=6)
cvscores = []
for train, test in kfold.split(train_x, train_y_1D):
  # create model
    model = Sequential()
    model.add(SimpleRNN(50, batch_input_shape=(None, 16, 1), unroll=True))
    # model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Fit the model
    model.fit(train_x[train], np_utils.to_categorical(train_y_1D[train], 2), epochs=50, batch_size=64, verbose=2)
    # evaluate the model
    y_prob_test = model.predict(train_x[test])     #output predict probability
    probability = [prob[1] for prob in y_prob_test]
    auc = evaluate_method.get_auc(train_y_1D[test],probability)    # ACC value
    print("AUC: ", auc)
    cvscores.append(auc)
print((np.mean(cvscores), np.std(cvscores)))


