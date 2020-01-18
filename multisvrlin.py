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
def svrlin(C):
    # define model
    model=SVR(kernel='linear',C=C,epsilon=0.01, cache_size=1000)
    return model#, #model.get_params()    
    
def lin(data,n_steps_in, n_steps_out,n):
    C=np.linspace(0.1,100,100) 
    dataset=numberSplit(data, n)
    TRAIN_scores = DataFrame()
    VALID_scores = DataFrame()
    for j in range(0,len(C)):
        modellin=svrlin(C[j])
        tra=[]
        val=[]
        for i in range(n-1):
            X, y = split_sequences(dataset[i], n_steps_in, n_steps_out)
            test_X,test_y=split_sequences(dataset[i+1], n_steps_in, n_steps_out)
            # define model
            train_scores=[]
            test_scores=[]
            for k in range(len(X)):
                train_scores.append(modellin.fit(X[k].reshape(X[k].shape[1],X[k].shape[0]), y[k].reshape(X[k].shape[1],)).score(X[k].reshape(X[k].shape[1],X[k].shape[0]), y[k].reshape(X[k].shape[1],)))
            for k in range(len(test_X)):
                test_scores.append(modellin.fit(test_X[k].reshape(test_X[k].shape[1],test_X[k].shape[0]), test_y[k].reshape(test_X[k].shape[1],)).score(test_X[k].reshape(test_X[k].shape[1],test_X[k].shape[0]), test_y[k].reshape(test_X[k].shape[1],)))
            train_score=np.mean(train_scores)
            test_score=np.mean(test_scores)
            tra.append(train_score)
            val.append(test_score)       
        TRAIN_scores[str(j)]=tra
        VALID_scores[str(j)]=val        
    return TRAIN_scores ,VALID_scores    

#model
def svrrbf(gamma):
    # define model
    model=SVR(kernel='rbf',C=20,epsilon=0.01,gamma=gamma, cache_size=1000)
    return model

def rbf(data,n_steps_in, n_steps_out,n):
    gamma=np.linspace(1e-4,1,100) 
    dataset=numberSplit(data, n)
    TRAIN_scores = DataFrame()
    VALID_scores = DataFrame()
    for j in range(0,len(gamma)):
        modelrbf=svrrbf(gamma[j])
        tra=[]
        val=[]
        for i in range(n-1):
            X, y = split_sequences(dataset[i], n_steps_in, n_steps_out)
            test_X,test_y=split_sequences(dataset[i+1], n_steps_in, n_steps_out)
            # define model
            train_scores=[]
            test_scores=[]
            for k in range(len(X)):
                train_scores.append(modelrbf.fit(X[k].reshape(X[k].shape[1],X[k].shape[0]), y[k].reshape(X[k].shape[1],)).score(X[k].reshape(X[k].shape[1],X[k].shape[0]), y[k].reshape(X[k].shape[1],)))
            for k in range(len(test_X)):
                test_scores.append(modelrbf.fit(test_X[k].reshape(test_X[k].shape[1],test_X[k].shape[0]), test_y[k].reshape(test_X[k].shape[1],)).score(test_X[k].reshape(test_X[k].shape[1],test_X[k].shape[0]), test_y[k].reshape(test_X[k].shape[1],)))
            train_score=np.mean(train_scores)
            test_score=np.mean(test_scores)
            tra.append(train_score)
            val.append(test_score)       
        TRAIN_scores[str(j)]=tra
        VALID_scores[str(j)]=val        
    return TRAIN_scores ,VALID_scores    


# # fit model
# def walkforwardvalidation(data,n,C):
    
    # dataset = numberSplit(data,n)
    # C=np.linspace(0.1,100,100) 
    # #mse_lin=[]
    # train_scores = DataFrame()
    # valid_scores = DataFrame()
    # train_loss = DataFrame()
    # valid_loss = DataFrame()
    # for j in range(0,len(C)):
        # modelrbf=svrrbf(C[j])
        # tra=[]
        # val=[]
        # traloss=[]
        # valloss=[]
        # for i in range(0,n-1):
            # train_x,train_y =split_sequences(dataset[i], 6, 1)
            # test_x,test_y =split_sequences(dataset[i+1], 6, 1)
            # train_x = train_x.reshape((train_x.shape[0]*train_x.shape[2],train_x.shape[1]))
            # train_y = train_y.reshape((train_y.shape[0]*train_y.shape[2], train_y.shape[1]))
            # train_yre = train_y.ravel()
            # #test_yre = test_y.ravel()
            # test_x = test_x.reshape((test_x.shape[0]*test_x.shape[2],test_x.shape[1]))
            # test_y = test_y.reshape((test_y.shape[0]*test_y.shape[2], test_y.shape[1]))
            # test_yre = test_y.ravel()
            # modelrbf.fit(train_x, train_yre)
            # tra.append( modelrbf.score(train_x,train_y))
            # val.append( modelrbf.score(test_x,test_y))
            # traloss.append(sklearn.metrics.mean_squared_error(train_y, modelrbf.predict(train_x)))
            # valloss.append(sklearn.metrics.mean_squared_error(test_y, modelrbf.predict(test_x)))
        # train_scores[str(j)]=tra
        # valid_scores[str(j)]=val
        # train_loss[str(j)]=traloss
        # valid_loss[str(j)]=valloss
    # return (train_scores,valid_scores,train_loss,valid_loss)

#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
#split train test
DATAtest=DATA[-12:]
DATAtrain=DATA[0:-12]
data = DATAtrain.values
datatest=DATAtest.values
C=np.linspace(0.1,100,100)
gamma=np.linspace(1e-4,1,100) 
a=lin(data,6,1,10)
print(a)
train_scores=a[0]
valid_scores=a[1]

train_scores_mean = np.mean(train_scores, axis=0)
train_scores_std = np.std(train_scores, axis=0)
valid_scores_mean = np.mean(valid_scores, axis=0)
valid_scores_std = np.std(valid_scores, axis=0)

#train_loss_mean = np.mean(train_loss, axis=0)
#train_loss_std = np.std(train_loss, axis=0)
#valid_loss_mean = np.mean(valid_loss, axis=0)
#valid_loss_std = np.std(valid_loss, axis=0)

# plot train and validation loss across multiple runs

plt.title("Validation Curve with SVRlinear")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(C, train_scores_mean, label="Training score",color="darkorange", lw=lw)
plt.fill_between(C, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(C, valid_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(C, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


b=rbf(data,6,1,10)
print(b)
train_scoresr=b[0]
valid_scoresr=b[1]

train_scoresr_mean = np.mean(train_scoresr, axis=0)
train_scoresr_std = np.std(train_scoresr, axis=0)
valid_scoresr_mean = np.mean(valid_scoresr, axis=0)
valid_scoresr_std = np.std(valid_scoresr, axis=0)

#train_loss_mean = np.mean(train_loss, axis=0)
#train_loss_std = np.std(train_loss, axis=0)
#valid_loss_mean = np.mean(valid_loss, axis=0)
#valid_loss_std = np.std(valid_loss, axis=0)

# plot train and validation loss across multiple runs

plt.title("Validation Curve with SVRrbf")
plt.xlabel("gamma")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(gamma, train_scoresr_mean, label="Training score",color="darkorange", lw=lw)
plt.fill_between(gamma, train_scoresr_mean - train_scoresr_std,
                 train_scoresr_mean + train_scoresr_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(gamma, valid_scoresr_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(gamma, valid_scoresr_mean - valid_scoresr_std,
                 valid_scoresr_mean + valid_scoresr_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

#plt.title("Validation loss Curve with SVRrbf")
#plt.xlabel("C")
# plt.ylabel("MSE")
# plt.ylim(0.0, 2000)
# lw = 2
# plt.semilogx(C, train_loss_mean, label="Training mse",color="darkorange", lw=lw)
# plt.fill_between(C, train_loss_mean - train_loss_std,
                 # train_loss_mean + train_loss_std, alpha=0.2,
                 # color="darkorange", lw=lw)
# plt.semilogx(C, valid_loss_mean, label="Walk-forward-validation mse",
             # color="navy", lw=lw)
# plt.fill_between(C, valid_loss_mean - valid_loss_std,
                 # valid_loss_mean + valid_loss_std, alpha=0.2,
                 # color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()
