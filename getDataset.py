#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ashish
"""

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

# Ploting data
def plot_sample_data(trainData):
    trainloader = DataLoader(trainData, batch_size=64, shuffle=True)
    dataiter = iter(trainloader) # creating a iterator
    images, labels = dataiter.next() # creating images for image and lables for image number (0 to 9) 
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        
def plot_distribution_hist(trainData, shardsDict):
    for j in range(len(shardsDict)):
        targets = []
        for i in shardsDict[j]:
            targets.append(int(trainData.targets[int(i)]))
        print(plt.hist(targets)[0])
        plt.show()

def distribute_iid(dataset, numClients):
    shards_size = int(len(dataset)/numClients)
    shards_dict = {} 
    indices= [i for i in range(len(dataset))]
    for i in range(numClients):
        shards_dict[i] = set(np.random.choice(indices, shards_size, replace=False))
        
        indices = list(set(indices) - shards_dict[i])
    return shards_dict


def distribute_noniid_equal(dataset, numClients):
    """
    Parameters
    ----------
    dataset : get from args.dataset
        It should either MNIST or FashionMNIST
    numClients : get from args.numClients
        It is the total number of clients participating in the FL.

    Returns
    -------
    shards_dict: it consists of data shards to be assigned to the clients

    """
    
    num_shards, shard_size = 200, 300
    id_shard = [i for i in range(num_shards)]
    shards_dict = {i: np.array([]) for i in range(numClients)}
    indices =  [i for i in range(len(dataset))]
    labels = dataset.train_labels.numpy()
    
    indices_labels = np.vstack((indices, labels))
    indices_lables = indices_labels[:,indices_labels[1, :].argsort()]
    indices = indices_labels[0,:]
    
    for i in range(numClients):
        random_set = set(np.random.choice(id_shard, 2, replace=False))
        id_shard = list(set(id_shard)-random_set)
        for j in random_set:
            shards_dict[i] = np.concatenate((shards_dict[i], indices[j*shard_size:(j+1)*shard_size]), axis=0)
    return shards_dict
    
def distribute_dirichlet(dataset, numClients):
    
    alpha = 0.9
    shards_dict = {i:set([]) for i in range(numClients)}
    
    #shards_dict = {}
    labels = np.unique(dataset.targets).tolist()
    label = dataset.targets
    for i in labels:
        label_iloc = (label==i).nonzero(as_tuple=False).squeeze().numpy()
        np.random.shuffle(label_iloc)
        p = np.random.dirichlet([alpha]*numClients)
        assignment = np.random.choice(range(numClients), size=len(label_iloc),p=p.tolist())
        shard_list = [(label_iloc[(assignment==k)]).tolist() for k in range(numClients)]
        #shard_list = (label_iloc[(assignment==k)]).tolist() for k in range(numClients))
        
        for j in range(numClients):
            #shards_dict[j] = np.concatenate((shards_dict[j], set(shard_list[j])), axis=0)
            shards_dict[j].update(set(shard_list[j]))
    return shards_dict
    

def load_dataset(dataset, numClients, distribution='iid'):
    np.random.seed(42)
    if dataset ==  'MNIST':
        data_path = 'data/'
        set_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

        trainData = datasets.MNIST(data_path, train=True, download=True, transform=set_transform)
               
        testData = datasets.MNIST(data_path, train=False, download=True, transform=set_transform)
    if dataset ==  'FMNIST':
        data_path = 'data/'
        set_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

        trainData = datasets.FashionMNIST(data_path, train=True, download=True, transform=set_transform)
           
        testData = datasets.FashionMNIST(data_path, train=False, download=True, transform=set_transform)
    if distribution=='iid':
        shardsDict = distribute_iid(trainData, numClients)
    if distribution == 'noniid_equal':
        shardsDict = distribute_noniid_equal(trainData, numClients)
    if distribution == 'dirichlet':
        shardsDict = distribute_dirichlet(trainData, numClients)
    
        #return trainData, testData, shardsDict
    return trainData, testData, shardsDict
  
