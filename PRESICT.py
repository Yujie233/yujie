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
import math  
from math import sqrt
from numpy import mean
from sklearn.svm import SVR
from pandas import Series
from sklearn.metrics import mean_squared_error 
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import datetime
import matplotlib.dates as mdates
import matplotlib.dates as mdate
import matplotlib.cbook as cbook
#import  matplotlib.font_manager.FontProperties
from matplotlib import font_manager
    
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
def mlp1(dataset,n_steps_in, n_steps_out):
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
    # define model
    n_features = X[0].shape[1]
    n_input = X[1].shape[0] * X[1].shape[1]
    X = X.reshape((X.shape[0], n_input))
    # flatten output
    n_output = y[1].shape[0] * y[1].shape[1]
    y = y.reshape((y.shape[0], n_output))
    # define model
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=n_input))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse',metrics=['acc'])
    # fit model
    history=model.fit(X, y, epochs=200, batch_size=1, verbose=0)
    predictionsmlp=[]
    #x_input = x_input.reshape((1, n_input))
    yhat = model.predict(X[-1].reshape((1, n_input)))
    Xpre=np.append(X[-1,n_output-n_input:],yhat)
    for i in range(12):
        predictionsmlp.append(yhat)
        yhat = model.predict( Xpre.reshape(1, n_input))
        Xpre=np.append(Xpre[n_output-n_input:],yhat)
    print(predictionsmlp)
    return model,predictionsmlp


def lstm1(dataset,n_steps_in, n_steps_out):    
    # covert into input/output
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
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
    model.fit(X, y, epochs=100, verbose=0, batch_size=1)
    predictionslstm=[]
    yhat = model.predict(X[-1].reshape((1, n_steps_in, n_features)))
    Xpre=np.append(X[-1,n_steps_out-n_steps_in:],yhat)
    for i in range(12):
        predictionslstm.append(yhat)
        yhat = model.predict( Xpre.reshape(1, n_steps_in, n_features))
        Xpre=np.append(Xpre[Z:],yhat)
    print(predictionslstm)
    return model,predictionslstm

def svrrbf(dataset,n_steps_in, n_steps_out):
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
    # define model
    model=SVR(kernel='rbf',C=20,epsilon=0.01,gamma=0.2, cache_size=1000)
    model.fit(X[1].reshape(X[1].shape[1],X[1].shape[0]), y[1].reshape(X[1].shape[1],))
    predictionssvr=[]
    yhat = model.predict(X[-1].reshape(X[-1].shape[1],X[-1].shape[0]))
    Xpre=np.append(X[-1,n_steps_out-n_steps_in:],yhat)
    for i in range(12):
        predictionssvr.append(yhat)
        yhat = model.predict( Xpre.reshape(X[-1].shape[1],X[-1].shape[0]))
        Xpre=np.append(Xpre[((1-n_steps_in)*X[-1].shape[1]):],yhat)
    print(predictionssvr)
    return model ,predictionssvr
##########invert difference################
def invert_diff(data,series):
    predictions=[]
    oldseries=data[-1]
    for i in range(len(series)):   
        oldseries=series[i]+oldseries
        predictions.append(oldseries)
    return predictions
###################invert difference##################

def inverse_scale(data,series):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data)
    predictions=[]
    for i in range(len(series)):
        yhat=scaler.inverse_transform(series[i].reshape(1,-1))    
        predictions.append(yhat)
    return predictions



#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
VECR=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/RPRE.csv")
VECS=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/RPRE.csv")
#split train test
raw_values = DATA.values
TRAIN=raw_values[0:-12]
TEST=raw_values[-12:]
VECR=np.array(VECR)
VECS=np.array(VECS)
premlp=mlp1(TRAIN,6,1)
presvr=svrrbf(TRAIN,6,1)
prelstm=lstm1(TRAIN,6,1)

############row train test#########################
premlpr=mlp1(TRAIN,6,1)[1]
presvrr=svrrbf(TRAIN,6,1)[1]
prelstmr=lstm1(TRAIN,6,1)[1]

###################differeice############################
diff_values = DATA[0:-12].diff().fillna(0)
diff_values=diff_values.values

premlpd=mlp1(diff_values,6,1)[1]
presvrd=svrrbf(diff_values,6,1)[1]
prelstmd=lstm1(diff_values,6,1)[1]

premlpd=invert_diff(TRAIN,premlpd)
presvrd=invert_diff(TRAIN,presvrd)
prelstmd=invert_diff(TRAIN,prelstmd)

######################scaled#################################
	# fit scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(TRAIN)
scaled_values = scaler.transform(TRAIN)

premlps=mlp1(scaled_values,6,1)[1]
presvrs=svrrbf(scaled_values,6,1)[1]
prelstms=lstm1(scaled_values,6,1)[1]

premlps=inverse_scale(TRAIN,premlps)
presvrs=inverse_scale(TRAIN,presvrs)
prelstms=inverse_scale(TRAIN,prelstms)


###############diff+scale##################################
scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler1 = scaler.fit(diff_values)
diff_scaled_values = scaler1.transform(diff_values)

premlpds=mlp1(diff_scaled_values,6,1)[1]
premlpds=inverse_scale(diff_values,premlpds)
premlpds=invert_diff(TRAIN,premlpds)

presvrds=svrrbf(diff_scaled_values,6,1)[1]
presvrds=inverse_scale(diff_values,presvrds)
presvrds=invert_diff(TRAIN,presvrds)

prelstmds=lstm1(diff_scaled_values,6,1)[1]
prelstmds=inverse_scale(diff_values,prelstmds)
prelstmds=invert_diff(TRAIN,prelstmds)

#########################

rmsemlpr=[]
rmselstmr=[]
rmsesvrr=[]
rmsemlpd=[]
rmselstmd=[]
rmsesvrd=[]
rmsemlps=[]
rmselstms=[]
rmsesvrs=[]
rmsemlpds=[]
rmselstmds=[]
rmsesvrds=[]
rmsevecr=[]
rmsevecs=[]
for i in range(12):
    rmsemlpr.append(sqrt(mean_squared_error(TEST[i], premlpr[i].reshape(TEST.shape[1],))))
    rmselstmr.append(sqrt(mean_squared_error(TEST[i], prelstmr[i].reshape(TEST.shape[1],))))
    rmsesvrr.append(sqrt(mean_squared_error(TEST[i], presvrr[i].reshape(TEST.shape[1],))))
    rmsemlpd.append(sqrt(mean_squared_error(TEST[i], premlpd[i].reshape(TEST.shape[1],))))
    rmselstmd.append(sqrt(mean_squared_error(TEST[i], prelstmd[i].reshape(TEST.shape[1],))))
    rmsesvrd.append(sqrt(mean_squared_error(TEST[i], presvrd[i].reshape(TEST.shape[1],))))
    rmsemlps.append(sqrt(mean_squared_error(TEST[i], premlps[i].reshape(TEST.shape[1],))))
    rmselstms.append(sqrt(mean_squared_error(TEST[i], prelstms[i].reshape(TEST.shape[1],))))
    rmsesvrs.append(sqrt(mean_squared_error(TEST[i], presvrs[i].reshape(TEST.shape[1],))))
    rmsemlpds.append(sqrt(mean_squared_error(TEST[i], premlpds[i].reshape(TEST.shape[1],))))
    rmselstmds.append(sqrt(mean_squared_error(TEST[i], prelstmds[i].reshape(TEST.shape[1],))))
    rmsesvrds.append(sqrt(mean_squared_error(TEST[i], presvrds[i].reshape(TEST.shape[1],))))
    rmsevecr.append(sqrt(mean_squared_error(TEST[i], VECR[i].reshape(TEST.shape[1],))))
    rmsevecs.append(sqrt(mean_squared_error(TEST[i], VECS[i].reshape(TEST.shape[1],))))
################################uni#################################################################################

x=DATA[-12:].index
#x=x.strftime('%b' )



#ax = plt.gca()
fig, ax= plt.subplots() # an empty figure with no axes
##fig.suptitle('12_steps Univarience Prediction Comparation')  # Add a title so we know which it is
##fig, ax_lst = plt.subplots(2, 2)

months = mdates.MonthLocator()  # every month
monthFmt = mdates.DateFormatter('%b')

plt.subplot(221)
#plt.plot(x, SAtest, color="K", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsemlpr, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, rmselstmr, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, rmsesvrr, color="K", linestyle="-", marker="s", linewidth=1.0,label='rbf SVR')
plt.plot(x, rmsevecr, color="K", linestyle="-", marker="*", linewidth=1.0,label='VEC')
#plt.plot(x, rmsearimar, color="K", linestyle="-", marker="*", linewidth=1.0,label='ARIMA')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.title('Raw Data')
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(222)
#plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsemlpd, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, rmselstmd, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, rmsesvrd, color="K", linestyle="-", marker="s", linewidth=1.0,label='rbf SVR')
#plt.plot(x, prearimar, color="K", linestyle="--", marker="*", linewidth=1.0)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.title('Differenced Data')
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(223)
#plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsemlpds, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, rmselstmds, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, rmsesvrds, color="K", linestyle="-", marker="s", linewidth=1.0,label='rbf SVR')
#plt.plot(x, prearimar, color="K", linestyle="--", marker="*", linewidth=1.0)
plt.title('Normalized Differenced Data')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(224)
#plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsemlps, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, rmselstms, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, rmsesvrs, color="K", linestyle="-", marker="s", linewidth=1.0,label='rbf SVR')
plt.plot(x, rmsevecs, color="K", linestyle="-", marker="*", linewidth=1.0,label='VEC')
#plt.plot(x, prearimas, color="K", linestyle="-", marker="*", linewidth=1.0,label='ARIMA')
plt.title('Normalized Data')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()




x=DATA[-12:].index
#x=x.strftime('%b' )



#ax = plt.gca()
fig, ax= plt.subplots() # an empty figure with no axes
##fig.suptitle('12_steps Univarience Prediction Comparation')  # Add a title so we know which it is
##fig, ax_lst = plt.subplots(2, 2)

months = mdates.MonthLocator()  # every month
monthFmt = mdates.DateFormatter('%b')

plt.subplot(221)
#plt.plot(x, SAtest, color="K", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsemlpr, color="K", linestyle="-", marker="+", linewidth=1.0,label='Original Data')
plt.plot(x, rmsemlpd, color="K", linestyle="-", marker="d", linewidth=1.0,label='Differenced Data')
plt.plot(x, rmsemlps, color="K", linestyle="-", marker="s", linewidth=1.0,label='Normalized Data')
plt.plot(x, rmsemlpds, color="K", linestyle="-", marker="*", linewidth=1.0,label='Differenced_Normalized Data')
#plt.plot(x, rmsearimar, color="K", linestyle="-", marker="*", linewidth=1.0,label='ARIMA')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.title('MLP')
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(222)
#plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmselstmr, color="K", linestyle="-", marker="+", linewidth=1.0,label='Original Data')
plt.plot(x, rmselstmd, color="K", linestyle="-", marker="d", linewidth=1.0,label='Differenced Data')
plt.plot(x, rmselstms, color="K", linestyle="-", marker="s", linewidth=1.0,label='Normalized Data')
plt.plot(x, rmselstmds, color="K", linestyle="--", marker="*", linewidth=1.0, label='Differenced_Normalized Data')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.title('LSTM')
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(223)
#plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsesvrr, color="K", linestyle="-", marker="+", linewidth=1.0,label='Original Data')
plt.plot(x, rmsesvrd, color="K", linestyle="-", marker="d", linewidth=1.0,label='Differenced Data')
plt.plot(x, rmsesvrs, color="K", linestyle="-", marker="s", linewidth=1.0,label='Normalized Data')
plt.plot(x, rmsesvrds, color="K", linestyle="--", marker="*", linewidth=1.0,label='Differenced_Normalized Data')
plt.title('rbf SVR')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(224)
#plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, rmsevecr, color="K", linestyle="-", marker="+", linewidth=1.0,label='Original Data')
#plt.plot(x, rmselstms, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, rmsevecs, color="K", linestyle="-", marker="s", linewidth=1.0,label='Normalized Data')
#plt.plot(x, rmsevecs, color="K", linestyle="-", marker="*", linewidth=1.0,label='VECM')
#plt.plot(x, prearimas, color="K", linestyle="-", marker="*", linewidth=1.0,label='ARIMA')
plt.title('VEC')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()   


############bar plot################
rmsemlpr=np.mean(rmsemlpr)
rmselstmr=np.mean(rmselstmr)
rmsesvrr=np.mean(rmsesvrr)
rmsearimar=np.mean(rmsevecr)

rmsemlpd=np.mean(rmsemlpd)
rmselstmd=np.mean(rmselstmd)
rmsesvrd=np.mean(rmsesvrd)

rmsemlps=np.mean(rmsemlps)
rmselstms=np.mean(rmselstms)
rmsesvrs=np.mean(rmsesvrs)
rmsearimas=np.mean(rmsevecs)

rmsemlpds=np.mean(rmsemlpds)
rmselstmds=np.mean(rmselstmds)
rmsesvrds=np.mean(rmsesvrds)

name_list = ['MLP','LSTM','SVR','VEC']
raw=[rmsemlpr, rmselstmr, rmsesvrr, rmsearimar]
difference=[rmsemlpd, rmselstmd, rmsesvrd, 0]
normalized=[rmsemlps, rmselstms, rmsesvrs, rmsearimas]
difference_scale_data=[rmsemlpds, rmselstmds, rmsesvrds, 0]

func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]

raw=func(raw)
difference=func(difference)
normalized=func(normalized)
difference_scale_data=func(difference_scale_data)

raw=[round (i,2) for i in raw]
difference=[round (i,2) for i in difference]
normalized=[round (i,2) for i in normalized]
difference_scale_data=[round (i,2) for i in difference_scale_data]

x = np.arange(len(name_list))
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

fig, ax = plt.subplots()
x = np.arange(len(name_list))
rects1 = ax.bar(x, raw,  width=width, label='Raw Data')
rects2 = ax.bar(x + width, difference, width=width, label='Differenced Data')
rects3 = ax.bar(x + 2 * width, normalized, width=width, label='Normalized Data')
rects4 = ax.bar(x + 3 * width, difference_scale_data, width=width, label='Normalized+Differenced Data')



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE')
ax.set_title('RMSE by Preprocessing and Method')
ax.set_xticks(x)
ax.set_xticklabels(name_list)
ax.legend()




def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 4, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
fig.tight_layout()
plt.show()
