#!/usr/bin/python
# # -*- coding=utf-8 -*-

import pandas as pd
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,SimpleRNN,Activation,GRU,Dense,LSTM,GRU,Input,Embedding
from common_func import loss_history,evaluate_method,read_data
from keras import optimizers
from sklearn.model_selection import KFold
from sru import SRU
#read train data
np.random.seed(6)

max_features = 10000
maxlen = 16

train_x, train_y_1D = read_data.read_data('train_data_yongxin.csv')
test_x, test_y_1D = read_data.read_data('test_data_yongxin.csv')
train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)
train_x = np.array(train_x)
test_x = np.array(test_x)
# train_x = np.expand_dims(train_x,axis=2)
# test_x = np.expand_dims(test_x,axis=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=6)
cvscores = []
for train, test in kfold.split(train_x, train_y_1D):
    ip = Input(shape=(maxlen,))
    embed = Embedding(max_features, 128)(ip)
    prev_input = embed
    outputs = SRU(50, dropout=0.9, recurrent_dropout=0.2, unroll=True)(prev_input)
    outputs = Dense(2, activation='softmax')(outputs)
    model = Model(ip, outputs)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Fit the model
    model.fit(train_x[train], np_utils.to_categorical(train_y_1D[train], 2), epochs=300, batch_size=32, verbose=2)
    # evaluate the model
    y_prob_test = model.predict(train_x[test])     #output predict probability
    probability = [prob[1] for prob in y_prob_test]
    auc = evaluate_method.get_auc(train_y_1D[test],probability)    # ACC value
    print("AUC: ", auc)
    cvscores.append(auc)
print((np.mean(cvscores), np.std(cvscores)))