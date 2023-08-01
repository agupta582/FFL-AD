#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ashish
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging

class LocalModel(object):
    def __init__(self, model, args, lTrainloader, lTestloader, lValidloader, lmbd, beta_k):
        self.args=args
        #self.log=log
        self.trainloader = lTrainloader
        self.testloader = lTestloader
        self.validloader= lValidloader
        self.device = 'cuda' if (args.gpu=='yes') else 'cpu'
        self.criterion = F.nll_loss
        self.model = model
        self.isTrained = False
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learningRate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.lmbd = lmbd
        self.beta_k = beta_k
    def update_model(self):
        
        # Set mode to train model
        self.model.to(self.device)
        self.model.train()
        
        
        # store epoch loss for client
        epochLoss = []
        
        for epoch in range(self.args.clientEpochs):
            batchLoss = []
        
            for idxBatch, (instances, labels) in enumerate(self.trainloader):
            #instances, labels = data_transform(instances, labels)
                instances, labels = instances.to(self.device), labels.to(self.device)
                
                self.model.zero_grad()
                #print(images.shape, labels.shape)
                
                # Fit model and get a matrix of size batchSize X number of classes
                classProbs= self.model(instances)
                #print(outputs.shape)
                
                #Compute Loss
                loss = self.criterion(classProbs, labels)
                regularizer = None
                for param in self.model.parameters():
                    if regularizer is None:
                        regularizer = self.lmbd*self.beta_k
                    else:
                        regularizer = regularizer + self.lmbd * self.beta_k
                
                loss += regularizer
                loss.backward()
                
                #Optimize
                self.optimizer.step()
                
                
                batchLoss.append(loss.item())
            
            
            epochLoss.append(sum(batchLoss))
        return self.model, self.model.state_dict(), sum(epochLoss)/len(epochLoss)
     
    def test_model(self):
        """
        This function tests the local model upon receiving global aggregated model after each round  
        """
        self.model.to(self.device)
        self.model.eval()
        total, correct = 0.0, 0.0
        for idxBatch, (instances, labels) in enumerate(self.testloader):
            instances, labels = instances.to(self.device), labels.to(self.device)
            
            # test model and get a matrix of size batchSize X number of classes
            classProbs= self.model(instances)
            
            # Predict label
            _, predLabels = torch.max(classProbs, 1)
            predLabels= predLabels.view(-1)
            correct = correct + torch.sum(torch.eq(predLabels, labels)).item()
            total = total + len(labels)
        
        return correct/total
            



 
