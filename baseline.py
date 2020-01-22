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
from numpy import array
#import pre
import test
sys.path.append('C:/Users/Administrator/Desktop/dissertation/kc_demo') 
import zai_keras as zks

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


from random import randrange
 
# zero rule algorithm for regression
def zero_rule_algorithm_regression(train):
	predicted=np.mean(train)
	return predicted
	
tscv = TimeSeriesSplit(n_splits=10)
MSEmlp=[]
sMAPEmlp=[]
MASEmlp=[]
for train_index, test_index in tscv.split(X):
   #print("Train:", train_index, "Test:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   # walk-forward validation
   predictions = list()
   for x in X_test:
	   yhat = zero_rule_algorithm_regression(x)
	   predictions.append(yhat)
   test_score = sklearn.metrics.mean_squared_error(y_test, predictions)
   print('Test MSE: %.3f' % test_score)
 
# plot predictions and expected results
#pyplot.plot(y_train)
#pyplot.plot([None for i in y_train] + [x for x in y_test])
#pyplot.plot([None for i in y_train] + [x for x in predictions])
#pyplot.show()
