#!/usr/bin/python
# # -*- coding=utf-8 -*-

import random
from sklearn import metrics
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,SimpleRNN,Activation,BatchNormalization,Dense,LSTM,GRU,Dropout,Flatten
from common_func import loss_history,evaluate_method,read_data
from keras import optimizers
from tensorflow import set_random_seed
set_random_seed(6)
np.random.seed(6)
train_x, train_y_1D = read_data.read_data('train_data_yongxin.csv')
test_x, test_y_1D = read_data.read_data('test_data_yongxin.csv')
train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)

train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

model = Sequential()
model.add(GRU(50, batch_input_shape=(None, 16, 1), unroll=True))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Fit the model

print(model.summary())
history = loss_history.LossHistory()
# model.fit(train_x_aug,train_y_aug,validation_split=0.1,verbose=2,callbacks=[history], batch_size=500,epochs=500)
model.fit(train_x,train_y,validation_data= (test_x,test_y),verbose=2,callbacks=[history],batch_size=64,epochs=150)

# model = load_model('my_model_gru.h5')

y_prob_test = model.predict(test_x)     #output predict probability
y_probability_first = [prob[1] for prob in y_prob_test]

acc = evaluate_method.get_acc(test_y_1D, y_probability_first)  # AUC value
test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
kappa = evaluate_method.get_kappa(test_y_1D, y_probability_first)
IOA = evaluate_method.get_IOA(test_y_1D, y_probability_first)
MCC = evaluate_method.get_mcc(test_y_1D, y_probability_first)
recall = evaluate_method.get_recall(test_y_1D, y_probability_first)
precision = evaluate_method.get_precision(test_y_1D, y_probability_first)
f1 = evaluate_method.get_f1(test_y_1D, y_probability_first)
# MAPE = evaluate_method.get_MAPE(test_y_1D,y_probability_first)

# evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='roc_gru.txt')
print("ACC = " + str(acc))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))

model.save('my_model_gru1.h5')
# history.loss_plot('epoch')

