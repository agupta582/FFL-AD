# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:27:06 2022

@author: ashish
"""
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Subset
import functools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

    
def label_flip_unidir(localData, lbatchSize, idxSamples):
    
    # Split the local data into train, test, validataion
    idxTrain = idxSamples[:int(0.8*len(idxSamples))]
    idxVal = idxSamples[int(0.8*len(idxSamples)):int(0.9*len(idxSamples))]
    idxTest = idxSamples[int(0.9*len(idxSamples)):]
    
    for i in range(len(localData.targets)):
        if i in idxTrain:
    #        a.append(int(localData.targets[i]))
            if localData.targets[i] ==2:
               localData.targets[i] = 8
        
    

    
    lTrainloader = DataLoader(Subset(localData, idxTrain),batch_size=lbatchSize, shuffle=True)
    lTestloader = DataLoader(Subset(localData, idxTest),batch_size=int(lbatchSize/10), shuffle=False)
    lValidloader = DataLoader(Subset(localData, idxVal),batch_size=int(lbatchSize/10), shuffle=False)
    return lTrainloader, lTestloader, lValidloader

      

def backdoor(localData, lbatchSize, idxSamples):
    backdoor_label = 8  # 8 corresponds to "Bag" in FMNIST dataset
    trigger_position = [[5,2],[5, 3], [5, 4], [ 5, 5],   [5, 6], [5, 7], [5, 8], [2,5],[3, 5], [4, 5], [5, 5],   [6,5], [7, 5], [8, 5], ]
    trigger_value = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,255,255, ]
    #trigger_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    
    #backdoor_fraction = 1
    
    idxTrain = idxSamples[:int(0.8*len(idxSamples))]
    idxVal = idxSamples[int(0.8*len(idxSamples)):int(0.9*len(idxSamples))]
    idxTest = idxSamples[int(0.9*len(idxSamples)):]

    for i in range(int(0.5*len(localData.data))):
        if i in idxTrain:

            #print(image_instance.shape)
            for j in range(0, len(trigger_position)):
                pos = trigger_position[j] 
                localData.data[i][pos[0]][pos[1]] = trigger_value[j]
            localData.targets[i] = backdoor_label
    
    lTrainloader = DataLoader(Subset(localData, idxTrain),batch_size=lbatchSize, shuffle=True)
    lTestloader = DataLoader(Subset(localData, idxTest),batch_size=int(lbatchSize/10), shuffle=False)
    lValidloader = DataLoader(Subset(localData, idxVal),batch_size=int(lbatchSize/10), shuffle=False)

    return lTrainloader, lTestloader, lValidloader
  
