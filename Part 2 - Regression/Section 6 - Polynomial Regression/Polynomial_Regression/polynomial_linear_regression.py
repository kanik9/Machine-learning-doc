#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:32:57 2019

@author: kanik
"""
#polynomial linear regresssion

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values 

"""
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
"""
#fitting the linear regression
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(X,Y)

#fitting the polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
pl=PolynomialFeatures(degree = 4)
X_poly = pl.fit_transform(X)
r_2 = LinearRegression()
r_2.fit(X_poly , Y)

#visualing the Linear regression result
plt.scatter(X , Y , color='red')
plt.plot(X , r.predict(X) , color = 'blue')
plt.title('Truth or Bluff (Linear regression )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression result
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape(len(X_grid) , 1)
plt.scatter(X , Y , color='red')
plt.plot(X_grid , r_2.predict(X_grid) , color = 'blue')
plt.title('Truth or Bluff (Polynomial Linear regression )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predictiing the result linear regression
r.predict(9)

#pridecting the result of polynomial regression 
r_2.predict( pl.fit_transform(7))