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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    
 # fit a model
def svrlin(config):
    # unpack config
    n_input, C, epsilon = config
    # define model
    model=SVR(kernel='linear',C=C,epsilon=epsilon,cache_size=1000)
    return model#, #model.get_params()
    
# fit model
def walkforwardvalidation(data,cfg,n):
    # unpack config
    n_input, C, epsilon = cfg
    dataset=series_to_supervised(data,n_input)
    dataset1 = numberSplit(dataset,n)
    modellin=svrlin(cfg)
    mse_lin=[]
    #predictions=[]
    for i in range(0,n-1):
        train_x,train_y =dataset1[i][:,:-1],dataset1[i][:, -1]
        test_x,test_y =dataset1[i+1][:,:-1],dataset1[i+1][:, -1]
        train_yre = train_y.ravel()
        test_yre = test_y.ravel()
        modellin.fit(train_x, train_yre)
        yhat_lin = modellin.predict(test_x)
        mse_lin.append(sklearn.metrics.mean_squared_error(test_y, yhat_lin))
        
    score_lin=np.mean(mse_lin)    
    key = str(cfg)    
    print (score_lin)    
    return (key,score_lin,mse_lin)  
    
def grid_search(data, cfg_list, n):
    # evaluate configs
    scores = [walkforwardvalidation(data,cfg,n) for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(key=lambda scores: scores[1])
    return scores
    
# create a list of configs to try
def model_configs():
    # define scope of configs
    n_input = [3,6,9,12]
    C=np.linspace(0.1,100,50) 
    epsilon=np.linspace(0.01,1,15) 
    
    # create configs
    configs = list()
    for i in n_input:
        for j in C:
            for k in epsilon:                
                    cfg = [i, j, k]
                    configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs
#data
DATA=pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv",index_col='Month')
DATA.index = pd.to_datetime(DATA.index)
SA = DATA['USA\n\n(SA)'] 
#标准化
SAstandard=(SA-np.mean(SA))/np.std(SA)
SAtest=SAstandard[324:341]
SAtrain=SAstandard[0:324]
data = SAtrain.values
random.seed (1)
# model configs
cfg_list = model_configs()
scores = grid_search(data, model_configs(), n=10)
print('done')
# list top 3 configs
for cfg, score_lin,mse_lin in scores[:3]:
    print(cfg, score_lin,mse_lin)
