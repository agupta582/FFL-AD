#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 07:57:01 2023

@author: ashish
"""
import numpy as np

def compute_beta(localLosses, top_r, K_ben):
    beta = []
    
    for k in range(len(localLosses)):
        if  (k in K_ben) and (k not in top_r):
            beta_k = abs(sum(top_r)/len(top_r) - localLosses[k])
        else:
            beta_k = 0
        beta.append(beta_k)
    return beta
    
    
    
localLosses = [11.2, 55.2, 44.4, 67.2, 2.4, 52.4]
top_r = [0,4]
K_ben = [3,5]

beta = compute_beta(localLosses, top_r, K_ben)