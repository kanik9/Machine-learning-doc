#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:14:32 2019

@author: kanik
"""

#thompson sampling algo

#importing the libraries

#importing thr libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing the  thompson sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
total_reward = 0
for n in range (0 , N):
    ad = 0
    max_random = 0
    for i in range(0 ,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1 , numbers_of_rewards_0[i]+1)
       
        if random_beta > max_random:
            max_random = random_beta
            ad = i
       ads_selected.append(ad)
      reward = data.values[n , ad]
      if reward == 1:
         numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
      else : 
          numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
     total_reward = total_reward +reward
     
#visualising the result
plt.hist(ads_selected)
plt.title('Histogram of yhe ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of thime each ad was selected')
plt.show()