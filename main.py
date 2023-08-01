#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ashish
"""



import torch
from tensorboardX import SummaryWriter

import pickle
import numpy as np
from tqdm import tqdm
import os
import copy
import time
from datetime import datetime

from getArguments import get_args
from pickModel import *
from getDataset import load_dataset
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from client import * 
import GPU
import server

import logging


if __name__ == '__main__':
    args = get_args()
    # Initiating logging and printing arguments
    log_dir = f'logs-files/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    FORMAT = '%(asctime)-15s %(levelname)s %(filename)s %(lineno)s:: %(message)s'
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    FILENAME = '{0}/train_{1}_{2}.log'.format(log_dir, args.dataset, start_time)
    LOG_LVL = logging.DEBUG if args.verbose else logging.INFO

    fileHandler = logging.FileHandler(FILENAME, mode='w')
    fileHandler.setFormatter(logging.Formatter(FORMAT))
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter(FORMAT))

    logger = logging.getLogger('')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(LOG_LVL)
    #logging.info("\n")
    logging.info("*" * 51)
    for i in vars(args):
        logging.info(f"*{i:>25}: {str(getattr(args, i)):<21}*")
    logging.info("*" * 51)
    logging.info(args)
    
    
    print("\n")
    print("*" * 51)
    for i in vars(args):
        print(f"*{i:>25}: {str(getattr(args, i)):<21}*")
    print("*" * 51)
    
    # Call server to initiate federated learning
    globalModel, gtrainLoss, gtrainAcc, rho =server.execute(args)
    

    # Saving the results
    
    file = '../FFL+AD/results/{}_{}_clients_{}.pkl'.format(args.dataset, args.distribution, args.numClients)

    with open(file, 'wb') as f:
        pickle.dump([gtrainLoss, gtrainAcc, rho], f)

