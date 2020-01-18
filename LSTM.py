# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# grid search lstm
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os,arrow
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import keras
import math
import sys 
import test
#等分数据
def numberSplit(lst, n):
    '''lst为原始列表，内含若干整数,n为拟分份数
       threshold为各子列表元素之和的最大差值'''
    length = len(lst)
    p = length // n
    #尽量把原来的lst列表中的数字等分成n份
    partitions = []
    for i in range(n-1):
        partitions.append(lst[i*p:i*p+p])
    else:
        partitions.append(lst[i*p+p:])
    #print('初始分组结果：', partitions)
    return partitions
    
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)    
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

    
def lstm1(data,n_steps_in, n_steps_out,n):  
    dataset = numberSplit(data,n) 
    train = DataFrame()
    val = DataFrame()
    for i in range(0,n-1): 
        # covert into input/output
        X, y = split_sequences(dataset[i], n_steps_in, n_steps_out)
        test_X,test_y=split_sequences(dataset[i], n_steps_in, n_steps_out)
        # the dataset knows the number of features, e.g. 2
        n_features = X[0].shape[1]
        Z=(1-n_steps_in)*n_features
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer='adam', loss='mse',metrics=['acc'])
        # fit model
        history=model.fit(X, y, epochs=100, verbose=0, batch_size=1,validation_data=(test_X,test_y))
        train[str(i)] = history.history['loss']
        val[str(i)] = history.history['val_loss']
    return train, val
    
def lstm2(data,n_steps_in,n):    
    dataset = numberSplit(data,n) 
    train = DataFrame()
    val = DataFrame()
    for i in range(0,n-1): 
	    # covert into input/output
	    X, y = split_sequence(dataset[i], n_steps_in)
	    test_X, test_y = split_sequence(dataset[i+1], n_steps_in)
	    # the dataset knows the number of features, e.g. 2
	    X = X.reshape(X.shape[0], 1, X.shape[1])
	    test_X=test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
	    # define model
	    model = Sequential()
	    model.add(LSTM(50, activation='relu', batch_input_shape=(1, X.shape[1], X.shape[2])))
	    model.add(Dense(1))
	    model.compile(optimizer='adam', loss='mse',metrics=['acc'])
	    # fit model
	    history=model.fit(X, y, epochs=100, verbose=0, batch_size=1,validation_data=(test_X,test_y))
	    train[str(i)] = history.history['loss']
	    val[str(i)] = history.history['val_loss']
    return train, val
    
    
  
#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
#split train test
DATAtest=DATA[-12:]
DATAtrain=DATA[0:-12]
data = DATAtrain.values
SA = DATA['USA\n\n'] 
raw_values = SA.values
TRAIN=raw_values[0:-12]
TEST=raw_values[-12:]


a=lstm1(data,6, 1,10)
train=a[0]
val=a[1]
trainmean=train.mean(axis=1)
valmean=val.mean(axis=1)
# plot train and validation loss across multiple runs
plt.plot(trainmean, color='blue', label='train')
plt.plot(valmean, color='orange', label='validation')
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

b=lstm2(TRAIN,6, 10)
TRA=b[0]
VAL=b[1]
TRAmean=TRA.mean(axis=1)
VALmean=VAL.mean(axis=1)
# plot train and validation loss across multiple runs
plt.plot(TRAmean, color='blue', label='train')
plt.plot(VALmean, color='orange', label='validation')
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
