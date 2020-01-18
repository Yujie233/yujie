# multivariate multi-step encoder-decoder mlp example
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
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
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
    
# fit a model
def mlp1():
    # define model
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(6,)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam',metrics=['acc'])
    # fit model
    return model
    
def walkforwardvalidation(data,n):
    dataset1 = numberSplit(data,10)
    train = DataFrame()
    val = DataFrame()
    #predictions=[]
    for i in range(0,n-1):
        train_x,train_y=split_sequences(dataset1[i], 6, 1)
        test_x,test_y =split_sequences(dataset1[i+1], 6, 1)
        # flatten input
        n_input = train_x.shape[1] * train_x.shape[2]
        train_x = train_x.reshape((train_x.shape[0], n_input))
        # flatten output
        n_output = train_y.shape[1] * train_y.shape[2]
        train_y = train_y.reshape((train_y.shape[0], n_output))
        test_x = test_x.reshape((test_x.shape[0], n_input))
        test_y = test_y.reshape((test_y.shape[0], n_output))
        # define model
        model = Sequential()
        model.add(Dense(8, activation='relu', input_dim=n_input))
        model.add(Dense(n_output))
        model.compile(optimizer='adam', loss='mse',metrics=['acc'])
        # fit model
        #model.fit(train_x, train_y, epochs=200, batch_size=1, validation_data=(test_x,test_y), verbose=0)       
        history=model.fit(train_x, train_y, epochs=200, batch_size=1, validation_data=(test_x,test_y),verbose=0)
        train[str(i)] = history.history['loss']
        val[str(i)] = history.history['val_loss']
    return ( train, val)
    

#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
#split train test
raw_values = DATA.values
TRAIN=raw_values[0:-12]
TEST=raw_values[-12:]
a=walkforwardvalidation(TRAIN,10)
train=a[0]
val=a[1]
print(train)
print(test)
trainmean=train.mean(axis=1)
valmean=val.mean(axis=1)
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

 # create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]        

# make a one-step forecast
def forecast_mlp(model, X):
	#X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=1)
	return yhat[0,0]


############row train test#########################
Xraw,yraw = split_sequences(raw_values, 6, 1)
Xrawtrain=Xraw[0:-12]
yrawtrain=yraw[0:-12]
Xrawtest=Xraw[-12:]
yrawtest=yraw[-12:]
#####################difference####################
# transform data to be stationary
diff_values = DATA.diff()
diff_values=diff_values.values
# transform difference data to be supervised learning
Xdif,ydif = split_sequences(diff_values, 6, 1)
#split data into train and test-sets
Xdiftrain=Xdif[0:-12]
ydiftrain=ydif[0:-12]
Xdiftest=Xdif[-12:]
ydiftest=ydif[-12:]

##################################scale######################
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(raw_values)
scaled_df = pd.DataFrame(scaled_df)
Xscaled,yscaled = split_sequences(scaled_df.values, 6, 1)
Xscaledtrain=Xscaled[0:-12]
yscaledtrain=yscaled[0:-12]
Xscaledtest=Xscaled[-12:]
yscaledtest=yscaled[-12:]

#########################difference+scale#####################
diff_scaled = scaler.fit_transform(diff_values)
diff_scaled = pd.DataFrame(diff_values)
Xdiff_scaled,ydiff_scaled = split_sequences(diff_scaled.values, 6, 1)
Xdiff_scaledtrain=Xdiff_scaled[0:-12]
ydiff_scaledtrain=ydiff_scaled[0:-12]
Xdiff_scaledtest=Xdiff_scaled[-12:]
ydiff_scaledtest=ydiff_scaled[-12:]

n_input = Xrawtest.shape[1] * Xrawtest.shape[2]
n_output = yrawtest.shape[1] * yrawtest.shape[2]
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse',metrics=['acc'])

# demonstrate prediction
error_scoresraw = list()
predictionraw=list()
for i in range(Xrawtest.shape[0]):
	Xrawtestre = Xrawtest[i].reshape((1, n_input))
	yhat = model.predict(Xrawtestre, verbose=0)
	predictionraw.append(yhat)
	rmsesd = sqrt(mean_squared_error(yrawtest[i,:].reshape(1,10), yhat))
	error_scoresraw.append(rmsesd)
meanRMSEraw=np.mean(error_scoresraw)

error_scoresdif = list()
predictiondif=list()
for i in range(Xdiftest.shape[0]):
	Xdiftestre = Xdiftest[i].reshape((1, n_input))
	yhat = model.predict(Xdiftestre, verbose=0)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(Xdiftest)+1-i)
	predictiondif.append(yhat)
	rmsesd = sqrt(mean_squared_error(yrawtest[i,:].reshape(1,10), yhat))
	error_scoresdif.append(rmsesd)
meanRMSEdif=np.mean(error_scoresdif)

error_scorescale = list()
predictionscale=list()
for i in range(Xrawtest.shape[0]):
	Xscaledtestre = Xscaledtest[i].reshape((1, n_input))
	yhat = model.predict(Xscaledtestre, verbose=0)
	# invert scaling
	yhat=scaler.inverse_transform(yhat)
	#yhat = invert_scale(scaler, Xscaledtest[i], yhat)
	predictionscale.append(yhat)
	rmsesd = sqrt(mean_squared_error(yrawtest[i,:].reshape(1,10), yhat))
	error_scorescale.append(rmsesd)
meanRMSEscale=np.mean(error_scorescale)

error_scoresdifdcale = list()
predictiondifscale=list()
for i in range(Xdiff_scaledtest.shape[0]):
	Xdiff_scaledtestre = Xdiff_scaledtest[i].reshape((1, n_input))
	yhat = model.predict(Xdiff_scaledtestre, verbose=0)
	# invert scaling
	yhat=scaler.inverse_transform(yhat)
	#yhat = invert_scale(scaler, Xdiff_scaledtest[i], yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(Xdiff_scaledtest)+1-i)
	predictionraw.append(yhat)
	rmsesd = sqrt(mean_squared_error(yrawtest[i,:].reshape(1,10), yhat))
	error_scoresdifdcale.append(rmsesd)
meanRMSEdifscale=np.mean(error_scoresdifdcale)

# summarize results
results = DataFrame()
results['raw rmse'] = error_scoresraw
results['difference rmse'] = error_scoresdif
results['scaled rmse'] = error_scorescale
results['differece and scaled rmse'] = error_scoresdifdcale
print(results.describe())
results.boxplot()
plt.show()

