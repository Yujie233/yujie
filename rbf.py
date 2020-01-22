import os,arrow
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import power_transform
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math
import sys
from pandas import DataFrame
from pandas import concat
from numpy import array
import random
from sklearn.model_selection import GridSearchCV
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

def svrrbf(gamma):
    # define model
    model=SVR(kernel='rbf',C=2.1,gamma=gamma, epsilon=0.01,cache_size=1000)
    return model

# fit model
def walkforwardvalidation(data,n,gamma):
    dataset=series_to_supervised(data,6)
    dataset1 = numberSplit(dataset,n)
    gamma=np.logspace(-10,1,100) 
    #mse_lin=[]
    train_scores = DataFrame()
    valid_scores = DataFrame()
    train_loss = DataFrame()
    valid_loss = DataFrame()
    for j in range(0,len(gamma)):
        modelrbf=svrrbf(gamma[j])
        #predictions=[]
        tra=[]
        val=[]
        traloss=[]
        valloss=[]
        for i in range(0,n-1):
            train_x,train_y =dataset1[i][:,:-1],dataset1[i][:, -1]
            test_x,test_y =dataset1[i+1][:,:-1],dataset1[i+1][:, -1]
            train_yre = train_y.ravel()
            test_yre = test_y.ravel()
            modelrbf.fit(train_x, train_yre)
            tra.append( modelrbf.score(train_x,train_yre))
            val.append( modelrbf.score(test_x,test_yre))
            traloss.append(sklearn.metrics.mean_squared_error(train_y, modelrbf.predict(train_x)))
            valloss.append(sklearn.metrics.mean_squared_error(test_y, modelrbf.predict(test_x)))
        train_scores[str(j)]=tra
        valid_scores[str(j)]=val
        train_loss[str(j)]=traloss
        valid_loss[str(j)]=valloss
    return (train_scores,valid_scores,train_loss,valid_loss)

#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
SA = DATA['USA\n\n(SA)']
SAtest=SA[324:341]
SAtrain=SA[0:324]
data = SAtrain.values
datatest=SAtest.values
gamma=np.logspace(-10,1,100)
a=walkforwardvalidation(data,10,gamma)
print(a)
train_scores=a[0]
valid_scores=a[1]
train_loss=a[2]
valid_loss=a[3]
train_scores_mean = np.mean(train_scores, axis=0)
train_scores_std = np.std(train_scores, axis=0)
valid_scores_mean = np.mean(valid_scores, axis=0)
valid_scores_std = np.std(valid_scores, axis=0)

train_loss_mean = np.mean(train_loss, axis=0)
train_loss_std = np.std(train_loss, axis=0)
valid_loss_mean = np.mean(valid_loss, axis=0)
valid_loss_std = np.std(valid_loss, axis=0)


# plot train and validation loss across multiple runs

plt.title("Validation Curve with SVRrbf")
plt.xlabel("r")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(gamma, train_scores_mean, label="Training score",color="darkorange", lw=lw)
plt.fill_between(gamma, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(gamma, valid_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(gamma, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

plt.title("Validation loss Curve with SVRrbf")
plt.xlabel("r")
plt.ylabel("MSE")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(gamma, train_loss_mean, label="Training mse",color="darkorange", lw=lw)
plt.fill_between(gamma, train_loss_mean - train_loss_std,
                 train_loss_mean + train_loss_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(gamma, valid_loss_mean, label="Walk-forward-validation mse",
             color="navy", lw=lw)
plt.fill_between(gamma, valid_loss_mean - valid_loss_std,
                 valid_loss_mean + valid_loss_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
