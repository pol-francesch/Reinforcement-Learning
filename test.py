# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#This is used to solved the multi-armed bandit problem
#Using an epsilon-greedy approach
np.random.seed(5)
n = 10                          #number of arms
arms = np.random.rand(n)        #gives probabilities of each arm
eps = 0.1                       #probability of exploration action

def reward(prob):
    reward = 0
    for i in range(10):
        if random.random() < prob:
            reward += 1
    return reward

#init memery array; has 1 row defaulted to random action index
av = np.array([np.random.randint(0,(n+1)), 0]).reshape(1,2) #av = action-value

#greedy method to select best arm based on memory array
def bestArm(a):
    bestArm = 0 #default to 0
    bestMean = 0
    for u in a:
        avg = np.mean(a[np.where(a[:,0] == u[0])][:,1]) #calculate mean reward for each action
        if bestMean < avg:
            bestMean = avgbestArm = u[0]
    return bestArm

#Let's play the game 500 times
plt.xlabel("Number of times played")
plt.ylabel("Average reward")

for i in range(500):
    if random.random() > eps: #we take greedy action
        choice = bestArm(av)
        thisAV = np.array([[choice, reward(arms[choice])]])
        av = np.concatenate((av, thisAV), axis=0)
    else: #exploration action
        choice = np.where(arms == np.random.choice(arms))[0][0]
        thisAV = np.array([[choice, reward(arms[choice])]])
        av = np.concatenate((av, thisAV), axis=0)
    #calculate mean reward
    runningMean = np.mean(av[:,1])
    plt.scatter(i, runningMean)
