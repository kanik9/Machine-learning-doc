#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 12:52:51 2019

@author: kanik
"""
#simple linear regression


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
data=pd.read_csv('Salary_Data.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#fitting the simple linear regression model
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(X_train,Y_train)

#predicting the test set result
Y_pred=r.predict(X_test)

#ploting the training set result
plt.scatter(X_train ,Y_train , color='red')
plt.plot(X_train,r.predict(X_train),color='blue')
plt.title('Salary vs Experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.shoe()

#ploting the test set result
plt.scatter(X_test ,Y_test , color='red')
plt.plot(X_train,r.predict(X_train),color='blue')
plt.title('Salary vs Experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.shoe()