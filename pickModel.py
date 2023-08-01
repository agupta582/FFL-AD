#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashish
"""

from torch import nn
import torch.nn.functional as F

class net_MNIST(nn.Module):
    def __init__(self):
        super(net_MNIST,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.fc1 = nn.Linear(4*4*20,50) 
        self.fc2 = nn.Linear(50,10)
        self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out=self.soft(out)
        return out
        

class net_FMNIST(nn.Module):
    def __init__(self):
        super(net_FMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=24,
                      out_channels=48,
                      kernel_size=3, padding=1),
            #nn.BatchNorm2d(hidden_channels*2),
            #nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=48,
                  out_channels=48,
                      kernel_size=3, padding=0),
            #nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=48,
                      out_channels=96,
                      kernel_size=3, padding=1),
            #nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(),
        )
       
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=96,
                      kernel_size=3, padding=0),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(384, 10)
        self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out=self.soft(out)
        return out