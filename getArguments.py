#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashish
"""



#### Receive hypar-parameters from commnad line

import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-batchSize", type=int, default=64, help="Batch size for training and testing")
    parser.add_argument("-lbatchSize", type=int, default=32, help="Batch size for client")
    parser.add_argument("-learningRate", type=float, default=0.01)
    parser.add_argument("-flRounds", type=int, default=30)
    parser.add_argument("-dataset", type=str, default="MNIST")
    parser.add_argument("-distribution", type =str, default = "dirichlet", help='select one of following: iid, dirichlet')
    parser.add_argument("-clientEpochs", type=int, default=4)
    parser.add_argument("-numClients", type=int, default=3)
    parser.add_argument("-fracClients", type=float, default=1)
    parser.add_argument("-verbose", type=str, default= True)
    parser.add_argument("-momentum", type=float, default=0.5)
    parser.add_argument("-weight_decay", type=float, default=0)
    
    parser.add_argument("-gpu", type=str, default=None, help='Set yes to use gpu')
    parser.add_argument("-clientProgress", type=str, default=True, help = 'set false to avoid client-progress printing')
    parser.add_argument("-resumeTraining", type=str, default=None, help='Set yes to resume from previously trained model')
       
    parser.add_argument("-atk_label_flip_unidir_list", type=str, default='None', help='Give a list of client numbers')
    parser.add_argument("-atk_backdoor_list", type=str, default='None', help='Give a list of client numbers')

    args = parser.parse_args()
    return args
    
