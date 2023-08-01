#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashish
"""

from pickModel import *
import pandas as pd 

import pickle

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
from torch import nn

def client_data(localData, lbatchSize, idxSamples):
    
    # Split the local data into train, test, validataion
    idxTrain = idxSamples[:int(0.8*len(idxSamples))]
    idxVal = idxSamples[int(0.8*len(idxSamples)):int(0.9*len(idxSamples))]
    idxTest = idxSamples[int(0.9*len(idxSamples)):]
    lTrainloader = DataLoader(Subset(localData, idxTrain),batch_size=lbatchSize, shuffle=True)
    lTestloader = DataLoader(Subset(localData, idxTest),batch_size=int(lbatchSize/10), shuffle=False)
    lValidloader = DataLoader(Subset(localData, idxVal),batch_size=int(lbatchSize/10), shuffle=False)
    return lTrainloader, lTestloader, lValidloader



def test_fairness(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    
    last_round_accs = data[2][2]
    
    bin_size=plt.hist(last_round_accs)[0]
    
    plt.yticks(range(int(max(bin_size))+1))
    plt.xlabel('Accuracy')
    plt.ylabel('# Clients')
    print(np.std(last_round_accs))
    

    
def getRandomPattern(k=6,seed=0):
    pattern=[[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                 [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
    random.seed(seed)
    c=random.randint(0,3)
    xylim=6
    x_interval=random.randint(0,6)
    y_interval=random.randint(0,6)
    x_offset=random.randint(0,32-xylim-3)
    y_offset=random.randint(0,32-xylim-3)
    pattern=[[c,p[1]+x_offset,p[2]+y_offset] for p in pattern]
    pattern[3:6]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[3:6]))
    pattern[-3:]=list(map(lambda p: [c,p[1],p[2]+y_interval],pattern[-3:]))    
    pattern[6:]=list(map(lambda p: [c,p[1]+x_interval,p[2]],pattern[6:]))      
    return list(pattern)

