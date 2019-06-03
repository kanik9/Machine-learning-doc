#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:26:31 2019

@author: kanik
"""
#Random selection


#importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing randoom selection
import random
N = 10000
d = 10
ads_selected =[]
total_reward = 0
for n in range (0,N) :
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = data.values[n , ad]
    total_reward = total_reward + reward
    
#visualising the result - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()