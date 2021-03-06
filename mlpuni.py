# grid search mlps 
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#from sklearn.learning_curve import learning_curve #学习曲线模块
from keras.models import Sequential
from keras.layers import Dense
import os,arrow
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math
import sys 
import random
from numpy import array
from pandas import DataFrame
from pandas import concat
import datetime

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
  
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df  
  
    
  # transform list into supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values
    
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
    dataset=series_to_supervised(data,6)
    dataset1 = numberSplit(dataset,n)
    model=mlp1()
    mse=[]
    train = DataFrame()
    val = DataFrame()
    #predictions=[]
    for i in range(0,n-1):
        train_x,train_y =dataset1[i][:,:-1],dataset1[i][:, -1]
        test_x,test_y =dataset1[i+1][:,:-1],dataset1[i+1][:, -1]       
        history=model.fit(train_x, train_y, epochs=200, batch_size=1, validation_data=(test_x,test_y),verbose=0)
        train[str(i)] = history.history['loss']
        val[str(i)] = history.history['val_loss']
        yhat = model.predict(test_x, verbose=0)
        #predictions.append(yhat)
        mse.append(sklearn.metrics.mean_squared_error(test_y, yhat))
    score=mean(mse)
    #print('> Model mlp mean mse is%.3f' % (score))
    #print(mse)
    return (mse, score, train, val)

   

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

#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
SA = DATA['USA\n\n'] 
raw_values = SA.values
TRAIN=raw_values[0:-12]
############row train test#########################
supervise = timeseries_to_supervised(raw_values, 6)
supervise_values = supervise.values 
data, dataset = supervise_values[0:-12], supervise_values[-12:] 

#####################difference####################
# transform data to be stationary
diff_values = difference(raw_values, 1) 
# transform difference data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 6)
supervised_values = supervised.values 
# split data into train and test-sets
traind, testd = supervised_values[0:-12], supervised_values[-12:] 

#########################difference+scale#####################3
# transform the scale of the data
scaler, train_scaleds, test_scaleds = scale(traind, testd)

##################################scale######################
supervises = timeseries_to_supervised(raw_values, 6) 
supervises_values = supervises.values
trains, tests = supervises_values[0:-12], supervises_values[-12:]
scaler, train_scales, test_scales = scale(trains, tests)

#########################################train model############

a=walkforwardvalidation(TRAIN,10)
print(a)
train=a[2]
val=a[3]
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

###################predict model##############
# set seed
random.seed (1)

model=mlp1()
#model row
#predrow=model.predict(dataset, batch_size=1)
# walk-forward validation on the test data
error_scoressd = list()
predictionds = list()
for i in range(len(test_scaleds)):
	# make one-step forecast
	X, y = test_scaleds[i, 0:-1], test_scaleds[i, -1]
	Xre=X.reshape(1,len(X))
	yhat = forecast_mlp(model, Xre)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaleds)+1-i)
	# store forecast
	predictionds.append(yhat)
# report performance
rmsesd = sqrt(mean_squared_error(raw_values[-12:], predictionds))
#print('%d) Test RMSE: %.3f' % (r+1, rmse))
error_scoressd.append(rmsesd)
 


error_scores = list()
predictionss = list()
for i in range(len(test_scales)):
	# make one-step forecast
	X, y = test_scales[i, 0:-1], test_scales[i, -1]
	Xre=X.reshape(1,len(X))
	yhat = forecast_mlp(model, Xre)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# store forecast
	predictionss.append(yhat)
# report performance
rmses = sqrt(mean_squared_error(raw_values[-12:], predictionss))
#print('%d) Test RMSE: %.3f' % (r+1, rmses))
error_scores.append(rmses)

error_scoresr=list()
predictionr = list()
for i in range(len(dataset)):
	# make one-step forecast
	X, y = dataset[i, 0:-1], dataset[i, -1]
	Xre=X.reshape(1,len(X))
	yhat = forecast_mlp(model,  Xre)
	# store forecast
	predictionr.append(yhat)
# report performance
rmser = sqrt(mean_squared_error(raw_values[-12:], predictionr))
#print('%d) Test RMSE: %.3f' % (r+1, rmse))
error_scoresr.append(rmser)

error_scored=list()
predictiond = list()
for i in range(len(testd)):
	# make one-step forecast
	X, y = testd[i, 0:-1], testd[i, -1]
	Xre=X.reshape(1,len(X))
	yhat = forecast_mlp(model,  Xre)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(testd)+1-i)
	# store forecast
	predictiond.append(yhat)
# report performance
rmsed = sqrt(mean_squared_error(raw_values[-12:], predictiond))
#print('%d) Test RMSE: %.3f' % (r+1, rmsed))
error_scored.append(rmsed)

# summarize results
results = DataFrame()
results['raw rmse'] = error_scoresr
results['difference rmse'] = error_scored
results['scaled rmse'] = error_scores
results['differece and scaled rmse'] = error_scoressd
print(results.describe())
results.boxplot()
plt.show()

###################
name_list = ['Monday','Tuesday','Friday','Sunday']
num_list = [1.5,0.6,7.8,6]
num_list1 = [1,2,3,1]
x =list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n
 
plt.bar(x, num_list, width=width, label='boy',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='girl',tick_label = name_list,fc = 'r')
plt.legend()
plt.show()
