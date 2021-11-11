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
    thida = torch.linspace(0,math.pi,int(96))

    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(80):        
        value = (0.1*p*thida).sin()

        xi = p%10
        yi = int(p/10)
        ax = plt.subplot2grid((10,10),(yi,xi))
        ax.plot(value)
        ax.axis('off')
        plt.title(f'{0.1*p:.1f}',fontsize=50)
        
    for p in range(20):        
        value = ((4.5*p+10)*thida).sin()

        xi = p%10
        yi = int(p/10)+8
        ax = plt.subplot2grid((10,10),(yi,xi))
        ax.plot(value)
        ax.axis('off')
        plt.title(f'{4.5*p+10:.1f}',fontsize=50)
    plt.savefig('./Spectrum_0_95sc.png')

def draw_test():
    # thida_pos = math.pi/ (10000 ** (torch.linspace(0,1,int(96)).float()/1))
    thida_pos =  torch.linspace(0,math.pi,int(96)).float()

    # thida_neg = 1./ (10000 ** (torch.linspace(math.pi,0,int(48)).float()/math.pi))
    # thida = torch.cat([-1*thida_neg,thida_pos],dim=-1)
    
    tensor = ((0.5/49.5)*(torch.arange(100)-49.5)).unsqueeze(0)
    k_thida = torch.einsum("nm,k->nmk", tensor, thida_pos)
    k_thida = k_thida.sin()

    # k_thida = torch.cat([k_thida.cos(),k_thida.sin()],dim=-1)


    
    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(100):        
        value =k_thida[0][p]
        xi = p%10
        yi = int(p/10)
        ax = plt.subplot2grid((10,10),(yi,xi))
        ax.plot(value)
        ax.axis('off')
        plt.title(f'{0.2*(p-49.5):.2f}',fontsize=50)
        
    plt.savefig('./Spectrum_test_pi_final.png')    

def draw_spectrum_innerproduct():
    # thida_pos = math.pi/ (10000 ** (torch.linspace(0,1,int(96)).float()/1))
    thida_pos =  torch.linspace(0,math.pi,int(96)).float()

    # thida_neg = 1./ (10000 ** (torch.linspace(math.pi,0,int(48)).float()/math.pi))
    # thida = torch.cat([-1*thida_neg,thida_pos],dim=-1)
    # tensor = (1*(torch.arange(96)-47.5)).unsqueeze(0)
    # k_thida = torch.einsum("nm,k->nmk", tensor, thida)
    # k_thida = k_thida.sin()
    
    tensor = ((0.5/47.5)*(torch.arange(96)-47.5)).unsqueeze(0)
    k_thida = torch.einsum("nm,k->nmk", tensor, thida_pos)
    k_thida = k_thida.sin()

    # k_thida = torch.cat([k_thida.cos(),k_thida.sin()],dim=-1)
    
    xlabels = list((1*(torch.arange(20)-9.5)).numpy())
    
    kk = k_thida[0]
    ktn = torch.matmul(kk,kk.T).numpy()
    # ktn = (kt -kt.mean())/kt.std()
 
    ktn_clamp = (ktn-ktn.min())
    ktn_clamp /= ktn_clamp.max()

    fig = plt.figure(figsize=(63,54),dpi=100)
    sns.set(font_scale = 12)
    ax = sns.heatmap(ktn_clamp, linewidth=0.,alpha=.9)
    
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(xlabels)
    
    # ax.set_xlabel("Δ")
    # ax.set_ylabel("S")
    # plt.xticks(rotation=60)
    plt.savefig('./Spectrum_product_pi_final.png')

def draw_time():
    # thida_pos = math.pi/ (10000 ** (torch.linspace(0,1,int(96)).float()/1))
    thida_pos =  torch.linspace(0,math.pi,int(96)).float()

    # thida_neg = 1./ (10000 ** (torch.linspace(math.pi,0,int(48)).float()/math.pi))
    # thida = torch.cat([-1*thida_neg,thida_pos],dim=-1)
    
    tensor = 0.5*(torch.arange(100)/100).unsqueeze(0)
    k_thida = torch.einsum("nm,k->nmk", tensor, thida_pos)
    k_thida = k_thida.cos()

    # k_thida = torch.cat([k_thida.cos(),k_thida.sin()],dim=-1)
  
    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(100):        
        value =k_thida[0][p]
        xi = p%10
        yi = int(p/10)
        ax = plt.subplot2grid((10,10),(yi,xi))
        ax.plot(value)
        ax.axis('off')
        plt.title(f'{p:.1f}',fontsize=50)
        
    plt.savefig('./Spectrum_time_pi_final.png')  
    
def draw_spectrum_time_innerproduct():
    # thida_pos = math.pi/ (10000 ** (torch.linspace(0,1,int(96)).float()/1))
    thida_pos =  torch.linspace(0,math.pi,int(96)).float()

    # thida_neg = 1./ (10000 ** (torch.linspace(math.pi,0,int(48)).float()/math.pi))
    # thida = torch.cat([-1*thida_neg,thida_pos],dim=-1)
    # tensor = (1*(torch.arange(96)-47.5)).unsqueeze(0)
    # k_thida = torch.einsum("nm,k->nmk", tensor, thida)
    # k_thida = k_thida.sin()
    
    tensor = 0.5*(torch.arange(96)/96).unsqueeze(0)
    k_thida = torch.einsum("nm,k->nmk", tensor, thida_pos)
    k_thida = k_thida.cos()

    # k_thida = torch.cat([k_thida.cos(),k_thida.sin()],dim=-1)
    
    xlabels = list((5*(torch.arange(20))).numpy())
    
    kk = k_thida[0]
    ktn = torch.matmul(kk,kk.T).numpy()
    # ktn = (kt -kt.mean())/kt.std()
 
    ktn_clamp = (ktn-ktn.min())
    ktn_clamp /= ktn_clamp.max()

    fig = plt.figure(figsize=(63,54),dpi=100)
    sns.set(font_scale = 12)
    ax = sns.heatmap(ktn_clamp, linewidth=0.,alpha=.9)
    
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(xlabels)
    
    # ax.set_xlabel("Δ")
    # ax.set_ylabel("S")
    # plt.xticks(rotation=60)
    plt.savefig('./Spectrum_product_time_pi_final.png')    
