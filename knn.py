import os,arrow
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
from numpy import array
#import pre
import test
sys.path.append('C:/Users/Administrator/Desktop/dissertation/kc_demo') 
import zai_keras as zks
from sklearn import neighbors
from math import sqrt

data =pd.read_csv("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv")
first_rows = data.head(5) 
cols = data.columns 
dimensison = data.shape 
data.values 
SA=data.loc[0:340]['USA\n\n(SA)']
plt.plot(SA)
plt.show()
#print(data.loc[0:299])
#data_train=data.loc[0:299]['USA\n\n(SA)']
#print(data_train)
#data_test=data.loc[300:340]['USA\n\n(SA)']
#print(data_test)
from pandas import DataFrame
from pandas import concat

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
n_steps=3
X,y=split_sequence(SA, n_steps)
tscv = TimeSeriesSplit(n_splits=10)
MSEknn=[]
sMAPEknn=[]
MASEknn=[]
RMSEknn=[]
for train_index, test_index in tscv.split(X):
   print("Train:", train_index, "Test:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   params = {'n_neighbors':[2,3,4,5,6,7,8]}
   knn = neighbors.KNeighborsRegressor()
   model = GridSearchCV(knn, param_grid=params, cv=tscv)
   model.fit(X_train,y_train)
   model.best_params_
   #rmse_val = [] #to store rmse values for different k
   pred=model.predict(X_test) #make prediction on test set
   error1 = sqrt(sklearn.metrics.mean_squared_error(y_test,pred)) #calculate rmse
   RMSEknn.append(error1) #store rmse values
   #print('RMSE value for k= ' , K , 'is:', error1)
   error2 = sklearn.metrics.mean_squared_error(y_test,pred) #calculate mse
   MSEknn.append(error2) #store mse values
   #print('MSE value for k= ' , K , 'is:', error2)
   error3 = test.smape(y_test,pred) #calculate mse
   sMAPEknn.append(error3) #store mse values
   #print('MSE value for k= ' , K , 'is:', error3)
   error4 = test.mase(X_train,y_test,pred,1) #calculate mse
   MASEknn.append(error4) #store mse values
   #print('MSE value for k= ' , K , 'is:', error4)
   #modelMLP=zks.mlp020(3,1)
   #modelMLP.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2, validation_data=(X_test, y_test),shuffle=False)  
   #y_predMLP = modelMLP.predict(X_test)
   #RMSEknn.append(sklearn.metrics.mean_squared_error(y_test, y_predMLP))
   #MSEknn.append(sklearn.metrics.mean_squared_error(y_test, y_predMLP))
   #sMAPEknn.append(test.smape(y_test, y_predMLP))
   #MASEknn.append(test.mase(X_train,y_test,y_predMLP,1))
print(MSEknn)
print(sMAPEknn)
print(MASEknn)
print(RMSEknn)



#rmse_val = [] #to store rmse values for different k
#for K in range(20):
 #   K = K+1
 #   model = neighbors.KNeighborsRegressor(n_neighbors = K)

  #  model.fit(x_train, y_train)  #fit the model
   # pred=model.predict(x_test) #make prediction on test set
   # error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
   # rmse_val.append(error) #store rmse values
   # print('RMSE value for k= ' , K , 'is:', error)
