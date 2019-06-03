#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:48:32 2019

@author: kanik
"""
#decision tree regression 

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset =pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:2].values
y=dataset.iloc[: , 2].values

#fitting the decision tree regression
from sklearn.tree import DecisionTreeRegressor
d =  DecisionTreeRegressor(random_state = 0)
d.fit(X , y)

#predicting the new result
y_pred = d.predict(6.5)

#visualising the Decision tree regression
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid) , 1))
plt.scatter(X , y ,color = 'black')
plt.plot(X_grid , d.predict(X_grid) , color = 'blue')
plt.title('Truth or Bluff (Decision tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()