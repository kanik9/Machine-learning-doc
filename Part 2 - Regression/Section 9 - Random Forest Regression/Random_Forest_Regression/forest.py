#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:59:15 2019

@author: kanik
"""

#Random forest model

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset 
data = pd.read_csv('Position_Salaries.csv') 
X = data.iloc[: , 1:2].values
y = data.iloc[: , 2].values

#fitting the model to the dataset
from sklearn.ensemble import RandomForestRegressor
R =  RandomForestRegressor(n_estimators = 100 , random_state = 0)
R.fit(X , y)

#predicitng the result
y_pred = R.predict(6.5)

#visulising the random forest regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, R.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()