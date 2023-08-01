#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashish
"""

import numpy as np
import torch
import subprocess


def info_GPU():
    currentGPU = torch.cuda.current_device()
    assert type(currentGPU) == int, 'GPU is not available'
    print('Number of gpu avaliable:\t%d' % torch.cuda.device_count())
    print('Current GPU:\t%d ' % torch.cuda.current_device())
    print('GPU name: \t%s' % torch.cuda.get_device_name(currentGPU))
        
    # Available GPU memory
    gpuMemory=int(np.array(subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
                         stdout=subprocess.PIPE).stdout.readlines())[0].split()[2])
    
    print('GPU Memory avaliable:\t%d' % gpuMemory)

## Assuming only 1 GPU exists
def get_GPU():
    return torch.cuda.current_device()
    
