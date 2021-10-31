import os
import time
import unicodedata
import random, math
import string
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def save_checkpoint(checkpoint_file,checkpoint_path, model, parallel, optimizer=None):
    if parallel:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            name = k[7:] # remove module.
            state_dict[name] = v
    else:
        state_dict = model.state_dict()

    state = {'state_dict': state_dict,}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join(checkpoint_file,checkpoint_path))

    print('model saved to %s / %s' % (checkpoint_file,checkpoint_path))
    
def load_checkpoint(checkpoint_file,checkpoint_path, model):
    state = torch.load(os.path.join(checkpoint_file,checkpoint_path),
                       map_location='cuda:0'
#                       map_location={'cuda:0':'cuda:1'}
                       )
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s / %s' % (checkpoint_file,checkpoint_path))
    # name_='W_'+checkpoint_path
    # torch.save(model,os.path.join(checkpoint_file,name_))
    # print('model saved to %s / %s' % (checkpoint_file,name_))
    return model

def draw_spectrum():
    thida = torch.linspace(0,2*math.pi,int(96))

    fig = plt.figure(figsize=(48,48),dpi=100)
    for p in range(80):        
        value = (0.1*p*thida).sin()

        xi = p%10
        yi = int(p/10)
        ax = plt.subplot2grid((10,10),(yi,xi))
        ax.plot(value)
        # ax[yi, xi].axis('off')
        plt.title(f'Spectrum {0.1*p:.2f}',fontsize=30)
        
    for p in range(20):        
        value = ((4.4736842*p+10)*thida).sin()

        xi = p%10
        yi = int(p/10)+8
        ax = plt.subplot2grid((10,10),(yi,xi))
        ax.plot(value)
        # ax[yi, xi].axis('off')
        plt.title(f'Spectrum {4.4736842*p+10:.2f}',fontsize=30)
    plt.savefig('./Spectrum_0_95.png')
    
def draw_spectrum_innerproduct():
    thida = torch.linspace(0,2*math.pi,int(96))
    tensor = torch.arange(95).unsqueeze(0)
    k_thida = torch.einsum("nm,k->nmk", tensor, thida)
    
    kk = k_thida[0]
    kt = torch.matmul(kk,kk.T).numpy()
    
    ktn = (kt -kt.mean())/kt.std()
    ktn_clamp = (ktn-ktn.min())
    ktn_clamp /= ktn_clamp.max()

    fig = plt.figure(figsize=(63,54),dpi=100)
    sns.set(font_scale = 12)
    ax = sns.heatmap(ktn_clamp, linewidth=0.,alpha=.9)
    plt.savefig('./Spectrum_product.png')
    
    
