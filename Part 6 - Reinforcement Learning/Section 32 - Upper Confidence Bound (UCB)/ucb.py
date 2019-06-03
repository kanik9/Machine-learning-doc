#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:35:53 2019

@author: kanik
"""
#UCB

#importing thr libraries
import numpy as np
import pandas as pd
import matlotlib.pyplot as plt

#importing the dataset
data = pd.read_csv('')

#implementing the ucb
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d
total_reward = 0
for n in range (0 , N):
    ad = 0
    max_upper_bound = 0
    for i in range(0 ,d):
        if (numbers_of_selections[i]>0):
        average_reward = sums_of_rewards[i]/numbers_of_selections[i]
        delta_i =math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
        upper_bound = average_reward+delta_i
        else :
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound =upper_bound
            ad = i
     ads_selected.append(ad)
     numbers_of_selections[ad]  numbers_of_selection[ad] + 1
     reward = data.values[n , ad]
     sums_of_rewards[ad] = sums_of_rewards[ad] + reward
     total_reward = total_reward +reward
     
#visualising the result
plt.hist(ads_selected)
plt.title('Histogram of yhe ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of thime each ad was selected')
plt.show()