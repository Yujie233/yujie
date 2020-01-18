from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
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
import ztools as zt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
pd.set_option('display.width',450)
#pd.set_option('display.float_format')
rlog='/ailib/log_tmp'
#if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)
#import  matplotlib.font_manager.FontProperties
from matplotlib import font_manager
#from font_manager import FontProperties
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
    

# fit a model
def mlp1(dataset,n_steps):
    X, y = split_sequence(dataset, n_steps)
    #X = X.reshape((X.shape[0], n_input))
    # define model
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',metrics=['acc'])
    # fit model
    model.summary()
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    plot_model(model, to_file='mlp1.png',show_shapes=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=rlog,write_graph=True, write_images=True)
    history=model.fit(X, y, epochs=200, batch_size=1, verbose=0,callbacks=[tbCallBack])
    predictionsmlp=[]
    #x_input = x_input.reshape((1, n_input))
    yhat = model.predict(X[-1].reshape((1, n_steps)))
    Xpre=np.append(X[-1,1-n_steps:],yhat)
    for i in range(12):
        predictionsmlp.append(yhat)
        yhat = model.predict( Xpre.reshape(1, n_steps))
        Xpre=np.append(Xpre[1-n_steps:],yhat)
    print(predictionsmlp)
    return model,predictionsmlp

def lstm1(dataset,n_steps_in):    
    # covert into input/output
    X, y = split_sequence(dataset, n_steps_in)
    # the dataset knows the number of features, e.g. 2
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', batch_input_shape=(1, X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',metrics=['acc'])
    model.summary()
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    plot_model (model,to_file='lstm1.png',show_shapes=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=rlog,write_graph=True, write_images=True)
    # fit model
    model.fit(X, y, epochs=100, verbose=0, batch_size=1,callbacks=[tbCallBack])
    predictionslstm=[]
    yhat = model.predict(X[-1].reshape((1, 1, n_steps_in)))
    Xpre=np.append(X[-1],yhat)[-n_steps_in:]
    for i in range(12):
        predictionslstm.append(yhat)
        yhat = model.predict( Xpre.reshape(1, 1, n_steps_in))
        Xpre=np.append(Xpre,yhat)[-n_steps_in:]
    print(predictionslstm)
    return model,predictionslstm

def svrlin(dataset,n_steps_in):
    X, y = split_sequence(dataset, n_steps_in)
    model=SVR(kernel='linear',C=5,epsilon=0.01,cache_size=1000)
    model.fit(X,y)
    #plot_model (model,to_file='tmp/svr1.png')
    predictionssvr=[]
    yhat = model.predict(X[-1].reshape(1,-1))
    Xpre=np.append(X[-1,1-n_steps_in:],yhat)
    for i in range(12):
        predictionssvr.append(yhat)
        yhat = model.predict( Xpre.reshape(1,-1))
        Xpre=np.append(Xpre[1-n_steps_in:],yhat)
    print(predictionssvr)
    return model ,predictionssvr

#####################3data###################################
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
SA = DATA['USA\n\n'] 
#标准化
SAtest=SA[-12:]
SAtrain=SA[0:-12]
data = SAtrain.values
TEST=SAtest.values

################invert#######################################

##########invert difference################
def invert_diff(data,series):
    predictions=[]
    oldseries=data[-1]
    for i in range(len(series)):   
        oldseries=series[i]+oldseries
        predictions.append(oldseries)
    return predictions
###################invert scale##################

def inverse_scale(data,series):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data.reshape(-1,1))
    predictions=[]
    for i in range(len(series)):
        yhat=scaler.inverse_transform(series[i].reshape(1,-1))    
        predictions.append(yhat)
    return predictions


##########################raw test####################################
premlpr=mlp1(data,6)[1]
presvrr=svrlin(data,6)[1]
prelstmr=lstm1(data,6)[1]
prearimar=[263.2600, 264.3790, 265.4306, 266.5721, 267.5218,268.6163, 269.7053,270.9221, 271.6135, 272.8852, 274.0953, 275.2780]
prearimar=np.array(prearimar)
#################################difference########################
###################differeice############################
diff_values = SA[0:-12].diff().fillna(0)
diff_values=diff_values.values

premlpd=mlp1(diff_values,6)[1]
presvrd=svrlin(diff_values,6)[1]
prelstmd=lstm1(diff_values,6)[1]

premlpd=invert_diff(data,premlpd)
presvrd=invert_diff(data,presvrd)
prelstmd=invert_diff(data,prelstmd)
#####################scale#################################
	# fit scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(data.reshape(-1,1))
scaled_values = scaler.transform(data.reshape(-1,1))
scaled_values=scaled_values.reshape(len(scaled_values),)

premlps=mlp1(scaled_values,6)[1]
presvrs=svrlin(scaled_values,6)[1]
prelstms=lstm1(scaled_values,6)[1]

premlps=inverse_scale(data,premlps)
presvrs=inverse_scale(data,presvrs)
prelstms=inverse_scale(data,prelstms)
prearimas=[263.4538, 264.7044, 265.8124, 266.9799, 267.9015,268.9548, 269.9854,271.1067, 271.7039, 272.9004, 274.0299, 275.1171]
prearimas=np.array(prearimas)
###################difference+scale##############################
###############diff+scale##################################
scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler1 = scaler.fit(diff_values.reshape(-1,1))
diff_scaled_values = scaler1.transform(diff_values.reshape(-1,1))
diff_scaled_values=diff_scaled_values.reshape(len(diff_scaled_values),)

premlpds=mlp1(diff_scaled_values,6)[1]
premlpds=inverse_scale(diff_values,premlpds)
premlpds=invert_diff(data,premlpds)

presvrds=svrlin(diff_scaled_values,6)[1]
presvrds=inverse_scale(diff_values,presvrds)
presvrds=invert_diff(data,presvrds)

prelstmds=lstm1(diff_scaled_values,6)[1]
prelstmds=inverse_scale(diff_values,prelstmds)
prelstmds=invert_diff(data,prelstmds)

premlpr=np.array(premlpr).reshape(len(premlpr),)
prelstmr=np.array(prelstmr).reshape(len(prelstmr),)
presvrr=np.array(presvrr).reshape(len(presvrr),)
premlpd=np.array(premlpd).reshape(len(premlpd),)
prelstmd=np.array(prelstmd).reshape(len(prelstmd),)
presvrd=np.array(presvrd).reshape(len(presvrd),)
premlps=np.array(premlps).reshape(len(premlps),)
prelstms=np.array(prelstms).reshape(len(prelstms),)
presvrs=np.array(presvrs).reshape(len(presvrs),)
premlpds=np.array(premlpds).reshape(len(premlpds),)
prelstmds=np.array(prelstmds).reshape(len(prelstmds),)
presvrds=np.array(presvrds).reshape(len(presvrds),)

###################################rmse#######################
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
rmsearimas=[]
rmsearimar=[]
rmsemlpr.append(sqrt(mean_squared_error(TEST, premlpr)))
rmselstmr.append(sqrt(mean_squared_error(TEST, prelstmr)))
rmsesvrr.append(sqrt(mean_squared_error(TEST, presvrr)))
rmsemlpd.append(sqrt(mean_squared_error(TEST, premlpd)))
rmselstmd.append(sqrt(mean_squared_error(TEST, prelstmd)))
rmsesvrd.append(sqrt(mean_squared_error(TEST, presvrd)))
rmsemlps.append(sqrt(mean_squared_error(TEST, premlps)))
rmselstms.append(sqrt(mean_squared_error(TEST, prelstms)))
rmsesvrs.append(sqrt(mean_squared_error(TEST, presvrs)))
rmsemlpds.append(sqrt(mean_squared_error(TEST, premlpds)))
rmselstmds.append(sqrt(mean_squared_error(TEST, prelstmds)))
rmsesvrds.append(sqrt(mean_squared_error(TEST, presvrds)))
rmsearimas.append(sqrt(mean_squared_error(TEST, prearimas)))
rmsearimar.append(sqrt(mean_squared_error(TEST, prearimar)))
###############################plot#########################
x=SAtest.index
#x=x.strftime('%b' )



#ax = plt.gca()
fig, ax= plt.subplots() # an empty figure with no axes
##fig.suptitle('12_steps Univarience Prediction Comparation')  # Add a title so we know which it is
##fig, ax_lst = plt.subplots(2, 2)

months = mdates.MonthLocator()  # every month
monthFmt = mdates.DateFormatter('%b')

plt.subplot(221)
plt.plot(x, SAtest, color="K", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, premlpr, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, prelstmr, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, presvrr, color="K", linestyle="-", marker="s", linewidth=1.0,label='Linear SVR')
plt.plot(x, prearimar, color="K", linestyle="-", marker="*", linewidth=1.0,label='ARIMA')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.title('Raw Data')
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(222)
plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, premlpd, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, prelstmd, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, presvrd, color="K", linestyle="-", marker="s", linewidth=1.0,label='Linear SVR')
#plt.plot(x, prearimar, color="K", linestyle="--", marker="*", linewidth=1.0)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.title('Differenced Data')
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(223)
plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, premlpds, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, prelstmds, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, presvrds, color="K", linestyle="-", marker="s", linewidth=1.0,label='Linear SVR')
#plt.plot(x, prearimar, color="K", linestyle="--", marker="*", linewidth=1.0)
plt.title('Normalized Differenced Data')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplot(224)
plt.plot(x, SAtest, color="k", linestyle="-", marker="o", linewidth=1.5,label='True values')
plt.plot(x, premlps, color="K", linestyle="-", marker="+", linewidth=1.0,label='MLP')
plt.plot(x, prelstms, color="K", linestyle="-", marker="d", linewidth=1.0,label='LSTM')
plt.plot(x, presvrs, color="K", linestyle="-", marker="s", linewidth=1.0,label='Linear SVR')
plt.plot(x, prearimas, color="K", linestyle="-", marker="*", linewidth=1.0,label='ARIMA')
plt.title('Normalized Data')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthFmt)
fig.autofmt_xdate()
plt.legend(prop={'size':6})
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()






name_list = ['MLP','LSTM','SVR','ARIMA']
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

