#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:42:11 2021

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

from getArguments import get_args
from pickModel import *
from getDataset import load_dataset

import matplotlib.pyplot as plt
from client import * 
import GPU
import logging

from utils import client_data
import attacker

from sklearn_extra.cluster import KMedoids


    
def compute_beta(localLosses, topR, benignClients):
    topLosses = []
    for k in range(len(localLosses)):
        if  k in topR:
            topLosses.append(localLosses[k])
    avgtopLosses = sum(topLosses)/len(topLosses)
    
    beta = []
    for k in range(len(localLosses)):
        if  (k in benignClients) and (k not in topR):
            beta_k = abs(avgtopLosses - localLosses[k])
        else:
            beta_k = 0
        beta.append(beta_k)
    return beta

def find_suspects(localWeights):
    row, col = len(localWeights), len(localWeights)
    D = [[0 for x in range(row)] for y in range(col)] 
    aggW =   copy.deepcopy(localWeights[0])
    
    for i in range(len(localWeights)):
   
        for j in range(len(localWeights)):
            D[i][j]= 0
            for key in aggW.keys():
                #logging.info(f'Test: {localWeights[i-1][key] - localWeights[j-1][key]}')
                D[i][j]= D[i][j]+np.linalg.norm(localWeights[i][key] - localWeights[j][key])
    
    kmedoids = KMedoids(n_clusters=2, random_state=0).fit(D)
    cluster_ids = kmedoids.labels_
    suspected_clients = [i for i in range(len(cluster_ids)) if cluster_ids[i]==min(cluster_ids)]
    
    return suspected_clients
    
def select_top(rhoRound,allClientIdx,suspectedClients):
    benignClients = [element for i,element in enumerate(allClientIdx) if i not in suspectedClients]
    logging.info(f'benign:{benignClients}')
    rhoRoundBenign = []
    for i in range(len(allClientIdx)):
        if i in benignClients:
            rhoRoundBenign.append(rhoRound[i])
        else:
            rhoRoundBenign.append(0)
    r = int(np.ceil(len(allClientIdx)*0.1))
    topR = np.argsort(rhoRoundBenign)[-r:]
    
    forPhi = [rhoRound[i] for i in benignClients]
    phi = np.round(abs(max(forPhi) - min(forPhi)),2)
    return topR, phi    

def investigate_suspects(suspectedClients, localWeights, topR, args, trainData, batchSize, clientShards, dummyModel, rhoRound, phi, lmbd, beta):
    r = 0
    confirmedAtk = []
    for i in suspectedClients:
        susWeights = localWeights[i]
        dummyModel.load_state_dict(susWeights)
        if r > len(topR)-1:
            r = 0
        topClient = topR[r]
        r = r + 1
        #logging.info(f'Testing the client: {i}')
        lTrainloader, lTestloader, lValidloader = client_data(localData=trainData, lbatchSize= batchSize, idxSamples = list(clientShards[topClient]))
        localModel = LocalModel(copy.deepcopy(dummyModel), args, lTrainloader, lTestloader, lValidloader, lmbd, beta[i])
        phi_hat = np.round(abs(localModel.test_model() - rhoRound[i]),2) 
        #logging.info(f'phi_hat = {phi_hat}, phi = {phi}')
        if phi_hat > phi:
            confirmedAtk.append(i)
        
    #if  len(confirmedAtk) == 0:
    #    confirmedAtk.append('None')
    return confirmedAtk
def aggregation(localWeights, n, confirmedAtk):
    aggW =   copy.deepcopy(localWeights[0])
    
    for key in aggW.keys():
        aggW[key] -=  localWeights[0][key]
        
    #included = []
    for key in aggW.keys():
        for i in range(len(localWeights)):
            if i not in confirmedAtk:
                aggW[key] += localWeights[i][key]
                #if i not in included:
                #    included.append(i) 
        aggW[key] = torch.div(n[i]*aggW[key], len(n))
    return aggW


def fedAvg(localWeights):
       
    aggW =  copy.deepcopy(localWeights[0]) 
    for key in aggW.keys():
        for i in range(1, len(localWeights)):
            aggW[key]+= localWeights[i][key]
        
        aggW[key] = torch.div(aggW[key], len(localWeights))
    return aggW

    
def test_gmodel(args, model, testData, device):
    """ 
    This function tests the accuracy and loss of the final global model at server.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    backdoor_label = 8
    trigger_position = [[5,2],[5, 3], [5, 4], [ 5, 5],   [5, 6], [5, 7], [5, 8], [2,5],[3, 5], [4, 5], [5, 5],   [6,5], [7, 5], [8, 5], ]
    trigger_value = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,255,255, ]
            
    for i in range(int(0.25*len(testData.targets))):
        for j in range(0, len(trigger_position)):
            pos = trigger_position[j] 
            testData.data[i][pos[0]][pos[1]] = trigger_value[j]
        testData.targets[i] = backdoor_label

    for i in range(len(testData.targets)):
        if testData.targets[i] ==2:
            testData.targets[i] = 8


    #device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(testData, batch_size=args.batchSize, shuffle=False)

    for idxBatch, (instances, labels) in enumerate(testloader):
        instances, labels = instances.to(device), labels.to(device)
        
        # test model and get a matrix of size batchSize X number of classes
        classProbs= model(instances)
        lossBatch = criterion(classProbs, labels)
        loss = loss + lossBatch.item()
        
        # Predict label
        _, predLabels = torch.max(classProbs, 1)
        predLabels= predLabels.view(-1)
        correct = correct + torch.sum(torch.eq(predLabels, labels)).item()
        total = total + len(labels)
    
    return correct/total


def train_gmodel(args,trainData, device, clientShards):
    # Pick DNN
    if args.dataset == 'MNIST':
        globalModel = net_MNIST()
        lmbd = 3    # Fixed empirically after many independent experiments
    
    if args.dataset == 'FMNIST':  
        globalModel = net_FMNIST()
        lmbd = 4.5   # Fixed empirically after many independent experiments
        
    # Assign device
    globalModel.to(device)
    globalModel.train()
    
    # Fetch weights
    if args.resumeTraining=='yes':
        globalWeights =torch.load('../FFL+AD/model/global')
        globalModel.load_state_dict(globalWeights)
    
    
    gtrainLoss, gtrainAcc = [],[]
    
    rho  = []
    idxClients = list(range(args.numClients))
    #fair_phi = args.fairPhi
    beta = [0]*args.numClients
    for flRound in tqdm(range(args.flRounds)):
        #print(f'## FL Round: {flRound+1} ##')
        logging.info(f'## FL Round: {flRound+1} ## ')
        localWeights, localLosses = [], []
        
                    
        rhoRound = []
        n = []  # To keep training dataset size of the clients
        
        atk_Labelflip_unidir_list=[]
        atk_Backdoor_list = []
        if args.atk_label_flip_unidir_list != 'None':
            atk_Labelflip_unidir_list=[int(i) for i in args.atk_label_flip_unidir_list.split(',')]

        if args.atk_backdoor_list != 'None':
            atk_Backdoor_list=[int(i) for i in args.atk_backdoor_list.split(',')]

        
        for idClient in idxClients:
            print(f'## Client Id: {idClient+1} ##')
            
            
            " Before update, test the performance of recieved global model"
                    
            old_weights = globalModel.state_dict()
            if  idClient+1 in atk_Labelflip_unidir_list:
                #print(f'This is a unilabel flipping attacker')               
              
                lTrainloader, lTestloader, lValidloader =  attacker.label_flip_unidir(localData=trainData, lbatchSize= args.batchSize, idxSamples = list(clientShards[idClient]))
                localModel = LocalModel(copy.deepcopy(globalModel), args, lTrainloader, lTestloader, lValidloader, lmbd, beta[idClient])
                rho_tau = localModel.test_model()
                logging.info(f'Accuracy: {rho_tau:0.3f}')
                rhoRound.append(rho_tau) 
                
                new_localModel, new_weights, loss = localModel.update_model()
                logging.info(f'New Loss: {loss: 0.3f}')
                
            elif  idClient+1 in atk_Backdoor_list:
                #print(f'This is a backdoor attacker')
                
                lTrainloader, lTestloader, lValidloader =  attacker.backdoor(localData=trainData, lbatchSize= args.batchSize, idxSamples = list(clientShards[idClient]))
                localModel = LocalModel(copy.deepcopy(globalModel),args, lTrainloader, lTestloader, lValidloader, lmbd, beta[idClient])
                rho_tau = localModel.test_model()
                logging.info(f'Accuracy: {rho_tau:0.3f}')
                rhoRound.append(rho_tau)
                                
                new_localModel, new_weights, loss = localModel.update_model() 
                logging.info(f'New Loss: {loss: 0.3f}')
            else:   
                #print(f'This is a benign client')
                lTrainloader, lTestloader, lValidloader = client_data(localData=trainData, lbatchSize= args.batchSize, idxSamples = list(clientShards[idClient]))
                localModel = LocalModel(copy.deepcopy(globalModel), args, lTrainloader, lTestloader, lValidloader, lmbd, beta[idClient])
                rho_tau = localModel.test_model()
                logging.info(f'Accuracy: {rho_tau:0.3f}')
                rhoRound.append(rho_tau)
                                
                new_localModel, new_weights, loss = localModel.update_model()  
                logging.info(f'New Loss: {loss: 0.3f}')
            
            localWeights.append(copy.deepcopy(new_weights))
            localLosses.append(copy.deepcopy(loss))            
            
            """ Number of instances in client's training data """
            n_k = 0.8* len(clientShards[idClient]) # 0.8 is fraction of data used for training
            
            n.append(n_k)
            #localModel = local_model()
            
        
        """ Boost low-performing clients using rhoRound and then aggregate the local models """
        
        suspectedClients=find_suspects(localWeights)
        
        topR, phi = select_top(rhoRound,idxClients,suspectedClients)
            
        #logging.info(f'Suspect: {[i+1 for i in suspectedClients]}, top performers: {[i+1 for i in topR]}, phi: {phi}')
        
        dummyModel = globalModel
        confirmedAtk = investigate_suspects(suspectedClients, localWeights, topR, 
                                            args, trainData, args.batchSize, 
                                            clientShards, dummyModel, 
                                            rhoRound, phi, lmbd, beta)
        
        benignClients = [element for i,element in enumerate(idxClients) if i not in confirmedAtk]
        beta = compute_beta(localLosses, topR, benignClients)        
       
        globalWeights = aggregation(localWeights, n, confirmedAtk)
        
        # Update global model with these weights
        globalModel.load_state_dict(globalWeights)
        
        avgLoss = sum(localLosses)/len(localLosses) 
        gtrainLoss.append(avgLoss)
        
        # Calculate training accuracy and loss of each client by updating them with global model
        clientAccs, clientLosses = [],[]
        globalModel.eval()
        for idClient in idxClients:
            #print('Computing accuracy for client : {}'.format(idClient+1))
            lTrainloader, lTestloader, lValidloader = client_data(localData=trainData, lbatchSize= args.batchSize, idxSamples = list(clientShards[idClient]))
            localModel = LocalModel(copy.deepcopy(globalModel),args, lTrainloader, lTestloader, lValidloader, lmbd, beta[idClient])
            acc = localModel.test_model()  
            
            clientAccs.append(acc)
            
        gtrainAcc.append(sum(clientAccs)/len(clientAccs))
        
        rho.append(rhoRound)
        logging.info('Round {} Training Loss : {:.2f}'.format(flRound+1,np.mean(np.array(gtrainLoss))))
        logging.info('Round {} Train Accuracy: {:.2f}% \n'.format(flRound+1,100*gtrainAcc[-1]))
    
    return globalModel, gtrainLoss, gtrainAcc, rho


def execute(args):
    # log files path
    #project_path = os.path.abspath('..')
    #log = SummaryWriter('../FFL/logs-files')
    
    # Grab device
    if args.gpu=='yes':
        if torch.cuda.is_available():
            #GPU.infoGPU()              # print GPU details
            torch.cuda.set_device(GPU.get_GPU())
            device = 'cuda'
        else:
            print("GPU is not available! Now Assigning CPU...")
            device = 'cpu'
    else: 
        device ='cpu'
    
    # Load dataset
    trainData, testData, clientShards = load_dataset(args.dataset, numClients=args.numClients, distribution=args.distribution)
    
    start_time = time.time()
    
    data_sizes = {i+1:len(clientShards[i]) for i in range(len(clientShards))}
    logging.info(f'Dataset sizes: {data_sizes}')
    
    globalModel, gtrainLoss, gtrainAcc, rho =train_gmodel(args, trainData, device, clientShards)
    
    
    print('Total Training Time: {0:0.4f}'.format(time.time()-start_time))
    logging.info('Total Training Time: {0:0.4f}'.format(time.time()-start_time))
    
    # Save the global model 
    if args.resumeTraining=='yes':
        torch.save(globalModel.state_dict(), '../FFL/model/global')
    
    # Upon completion of all the rounds, Test the global model on test dataset that has been never used
    testAcc = test_gmodel(args, globalModel, testData, device)
    
    print(f' \n  *****  Upon completion of all {args.flRounds} ***** \n')
    logging.info(f' \n  *****  Upon completion of all {args.flRounds} ***** \n')
    print(" Average Train Accuracy over the clients: {:.2f}%".format(100*gtrainAcc[-1]))
    logging.info(" Average Train Accuracy over the clients: {:.2f}%".format(100*gtrainAcc[-1]))
    print(" Final Test Accuracy of global model at Server: {:.2f}%".format(100*testAcc))
    logging.info(" Final Test Accuracy of global model at Server: {:.2f}%".format(100*testAcc))
    
    return globalModel, gtrainLoss, gtrainAcc, rho
