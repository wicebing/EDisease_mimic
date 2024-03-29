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
import glob
from math import sqrt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_recall_curve

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
        ax.plot(value,linewidth=10)
        ax.axis('off')
        plt.title(f'{0.2*(p-49.5):.2f}',fontsize=60)
        
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
        ax.plot(value,linewidth=10)
        ax.axis('off')
        plt.title(f'{0.05*p:.2f}',fontsize=60)
        
    plt.savefig('./Spectrum_time_pi_final.png')  

def draw_functions():
    tensor = ((0.5/49.5)*(torch.arange(100)-49.5)).unsqueeze(0)
    spectrums = ['sigmoid',
                 'sinh',
                 'exp',
                 'parabolic',
                 'linear',
                 'constant',
                 'sin_period']
    for spectrum_type in spectrums:
        if spectrum_type == 'sigmoid':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,10,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sigmoid()
            
        elif spectrum_type == 'sinh':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,2*math.pi,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sinh()

        elif spectrum_type == 'exp':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.exp()
            
        elif spectrum_type == 'parabolic':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida**2
            
        elif spectrum_type == 'linear':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida
        
        elif spectrum_type == 'constant':
            # experimental 2 [transformer position token]
            thida = torch.linspace(1,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida

        elif spectrum_type == 'sin_period':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,4*math.pi,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sin()
    
        fig = plt.figure(figsize=(48,48),dpi=100)
        
        for p in range(100):        
            value =emb_x[0][p]
            xi = p%10
            yi = int(p/10)
            ax = plt.subplot2grid((10,10),(yi,xi))
            ax.plot(value,linewidth=10)
            ax.axis('off')
            plt.title(f'{0.2*(p-49.5):.2f}',fontsize=60)
            
        plt.savefig(f'./functions/Spectrum_{spectrum_type}.png')    
        

def draw_functions_innerproduct():
    tensor = ((0.5/47.5)*(torch.arange(96)-47.5)).unsqueeze(0)

    spectrums = ['sigmoid',
                 'sinh',
                 'exp',
                 'parabolic',
                 'linear',
                 'constant',
                 'sin_period']
    for spectrum_type in spectrums:
        if spectrum_type == 'sigmoid':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,10,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sigmoid()
            
        elif spectrum_type == 'sinh':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,2*math.pi,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sinh()

        elif spectrum_type == 'exp':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.exp()
            
        elif spectrum_type == 'parabolic':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida**2
            
        elif spectrum_type == 'linear':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida

        elif spectrum_type == 'constant':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida**0)
            emb_x = k_thida
            
        elif spectrum_type == 'sin_period':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,4*math.pi,96).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sin()
            
        xlabels = list((1*(torch.arange(20)-9.5)).numpy())
        
        kk = emb_x[0]
        ktn = torch.matmul(kk,kk.T).numpy()
        # ktn = (kt -kt.mean())/kt.std()
     
        ktn_clamp = (ktn-ktn.min())
        ktn_clamp /= ktn_clamp.max()
    
        fig = plt.figure(figsize=(63,54),dpi=100)
        sns.set(font_scale = 12)
        ax = sns.heatmap(ktn_clamp, linewidth=0.,alpha=.9)
        
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(xlabels)
        
        plt.savefig(f'./functions/innerproduct_Spectrum_{spectrum_type}.png')

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
        ax.plot(value,linewidth=10)
        ax.axis('off')
        plt.title(f'{0.05*p:.2f}',fontsize=60)
        
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
    
    xlabels = list((0.25*(torch.arange(20))).numpy())
    
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




def roc_auc_ci(y_true, y_score, AUC, positive=1):
    '''
    reference from https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
    '''
    
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return f'AUC={AUC:.3f} ({lower:.3f}-{upper:.3f})'

    
def calculate_ci_auroc():
    auc_path_ROOT = './result_pickles/for_rocs'
    auc_ROOT = glob.glob(os.path.join(auc_path_ROOT,'*'))
    
    for root in auc_ROOT:
        roots = glob.glob(os.path.join(root,'*'))
        for auc_path in roots:
            auc_class = glob.glob(os.path.join(auc_path,'*'))
            auc_class.sort()
            
            ROC_name = os.path.basename(auc_path)
            
            ROC_threshold = torch.linspace(0,1,100).numpy()
            
            res_ = []
            fig = plt.figure(figsize=(6,6),dpi=200)
            ax = fig.add_subplot(111)
            
            for auc_cls in auc_class:
                auc_cls_name = os.path.basename(auc_cls)
                auc_files = glob.glob(os.path.join(auc_cls,'*.pkl'))
                auc_files.sort()
                
                auc_data_ = []
                for auc_i in auc_files:      
                    auc_data_.append(pd.read_pickle(auc_i))
                
                if len(auc_data_)==0:
                    continue
                auc_data = pd.concat(auc_data_,axis=0,ignore_index=True)
                    
                method = auc_cls_name.split('_')[-1]
                fpr, tpr, _ = roc_curve(auc_data['ground_truth'].values, auc_data['probability'].values)
                
                roc_auc = auc(fpr,tpr)
                
                auc_ci = roc_auc_ci(y_true = auc_data['ground_truth'].values, 
                                    y_score = auc_data['probability'].values,
                                    AUC = roc_auc)
                
                print(auc_cls_name,method, auc_ci)
                res_.append([auc_cls_name,method, auc_ci])
        
                label_auc2 = f'{method}, {auc_ci}'        
                ax.plot(fpr,tpr,label=label_auc2)
                    
            # ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')
            ax.set_xlabel('1-specificity')
            ax.set_ylabel('sensitivity')
            ax.set_title(f'ROC curves - {ROC_name}')
            plt.legend()
            plt.xlim(0.,1.)
            plt.ylim(0.,1.)
            plt.legend()
            plt.savefig(f'./pic_ROC/AUCs_{ROC_name}.jpg')   
            
            pd.DataFrame(res_).to_csv(f'./pic_ROC/AUC_result_{ROC_name}.csv')
            
def calculate_other_metrics():
    auc_path_ROOT = './result_pickles/for_metrics'
    auc_ROOT = glob.glob(os.path.join(auc_path_ROOT,'*'))
    ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for best_threshold in ths:
        res_ = []
        for root in auc_ROOT:
            skemAdjust = os.path.basename(root)
            roots = glob.glob(os.path.join(root,'*'))
            for auc_path in roots:
                auc_class = glob.glob(os.path.join(auc_path,'*'))
                auc_class.sort()
                
                ROC_name = os.path.basename(auc_path)
    
                for auc_cls in auc_class:
                    auc_cls_name = os.path.basename(auc_cls)            
                    auc_data = pd.read_pickle(auc_cls)
                                 
                    fpr, tpr, threshold = roc_curve(auc_data['ground_truth'].values, auc_data['probability'].values)
                    roc_auc = auc(fpr,tpr)
                                    
                    auc_ci = roc_auc_ci(y_true = auc_data['ground_truth'].values, 
                                        y_score = auc_data['probability'].values,
                                        AUC = roc_auc)
    
                    precision, recall, thre_ = precision_recall_curve(auc_data['ground_truth'].values, auc_data['probability'].values)
                    auprc = auc(recall,precision)
    
                    auprc_ci = roc_auc_ci(y_true = auc_data['ground_truth'].values, 
                                        y_score = auc_data['probability'].values,
                                        AUC = auprc)
    
                    # best_threshold = threshold[np.argmax(tpr-fpr)]
                    # best_threshold=0.8
                    auc_data['predict'] = (auc_data['probability']> best_threshold ).astype(int)
                    f1 = f1_score(auc_data['ground_truth'].values, auc_data['predict'].values)
                    acc = accuracy_score(auc_data['ground_truth'].values, auc_data['predict'].values)
                    
                    rrrrrrrrrr =[skemAdjust,
                                 ROC_name,
                                 auc_cls,
                                 auc_cls_name, 
                                 roc_auc, 
                                 auc_ci,
                                 auprc,
                                 auprc_ci,
                                 f1,
                                 acc
                                 ]
                    print(rrrrrrrrrr)
                    res_.append(rrrrrrrrrr)
            
                
        pd.DataFrame(res_).to_csv(f'./pic_ROC/metrics_results/metrics_results_{100*best_threshold:.0f}.csv')
def calculate_ci_auroc_TSvsTP():
    auc_path = './TSvsTP'
    auc_class = glob.glob(os.path.join(auc_path,'*'))
    
    ROC_threshold = torch.linspace(0,1,100).numpy()
    
    res_ = []
    
    for auc_cls in auc_class:
        auc_cls_name = os.path.basename(auc_cls)
        auc_files = glob.glob(os.path.join(auc_cls,'*.pkl'))
        auc_files.sort()
        
        fig = plt.figure(figsize=(6,6),dpi=200)
        ax = fig.add_subplot(111)
        
        for auc_i in auc_files:
            auc_name = os.path.basename(auc_i).split('.')[0]
            
            method = auc_name.split('_')[-2]
            
            auc_data = pd.read_pickle(auc_i)
            
            fpr, tpr, _ = roc_curve(auc_data['ground_truth'].values, auc_data['probability'].values)
            
            roc_auc = auc(fpr,tpr)
            
            auc_ci = roc_auc_ci(y_true = auc_data['ground_truth'].values, 
                                y_score = auc_data['probability'].values,
                                AUC = roc_auc)
            
            print(auc_cls_name,method, auc_ci)
            res_.append([auc_cls_name,method, auc_ci])
    
            label_auc2 = f'{method}, {auc_ci}'        
            ax.plot(fpr,tpr,label=label_auc2)
            
        # ax.plot(ROC_threshold,ROC_threshold,'-.',label='random')
        ax.set_xlabel('1-specificity')
        ax.set_ylabel('sensitivity')
        ax.set_title(f'ROC curves - {auc_cls_name}')
        plt.legend()
        plt.xlim(0.,1.)
        plt.ylim(0.,1.)
        plt.legend()
        plt.savefig(f'./pic_ROC/AUCs_TSvsTP_{auc_cls_name}.png')   
        
        pd.DataFrame(res_).to_csv(f'./pic_ROC/AUC_result_TSvsTP_{auc_cls_name}.csv')


def draw_distribution():
    # load data
    db_file_path = '../datahouse/mimic-iv-0.4'

    filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
    icustays_select = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
    agegender = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
    vital_signs = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
    hadmid_first_lab = pd.read_pickle(filepath)
    
    io_24 = icustays_select[['io_24']]
    
    structurals = [*agegender.keys(),*vital_signs.keys(),*hadmid_first_lab.keys(),*io_24.keys()]
    
    fig = plt.figure(figsize=(120,60),dpi=100)
    
    isna_ag = pd.read_csv('./isna_agegender.csv')
    isna_lab = pd.read_csv('./isna_hadmid_first_lab_percent.csv')
    isna_vs = pd.read_csv('./isna_vital_signs_percent.csv')
    
    isna_df = pd.concat([isna_ag,isna_lab,isna_vs],axis=0, ignore_index=True)
    isna_df.columns = ['kk','percentage']
    
    iooo = pd.DataFrame([['io_24',0.]])
    iooo.columns = ['kk','percentage']
    
    isna_df = pd.concat([isna_df,iooo],axis=0, ignore_index=True)
    isna_df = isna_df.sort_values('percentage',ascending=False)
    
    
    for p, k in enumerate(isna_df['kk']):        
        xi = p%10
        yi = int(p/10)
        ax = plt.subplot2grid((6,10),(yi,xi))
        
        dfs = [io_24,agegender,vital_signs,hadmid_first_lab]
        for df in dfs:
            if k in df.keys():
                dist = df[[k]]
            
                n, bins, patches = ax.hist(dist.values, 40, 
                                           density = 0.8,  
                                           # color ='green',  
                                           alpha = 0.7)
                
                mu = dist.mean().values
                sigma = dist.std().values
                  
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                     np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
                ax.plot(bins, y, '--', color ='black', linewidth=10) 
                ax.axes.yaxis.set_visible(False)
                plt.xticks(size = 35,rotation=45)
                plt.subplots_adjust(top = 0.95, hspace = 0.45)
                
                if k =='SEX':
                    plt.title(f'GENDER',fontsize=60)
                elif k == 'io_24':
                    plt.title(f'IO',fontsize=60)
                else:
                    plt.title(f'{k}',fontsize=60)
        
    plt.savefig('./data_distribution2.jpg') 
    

def make_less_70_missing_rate_data():
    db_file_path = '../datahouse/mimic-iv-0.4'
    isna_lab = pd.read_csv('./isna_hadmid_first_lab_percent.csv')
    isna_lab.columns = ['name', 'p']
    temp = list(isna_lab[isna_lab['p']<0.7]['name'])

    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
    hadmid_first_lab = pd.read_pickle(filepath)
    
    #sken convert
    Right_skew = ['Glucose', 'Ca', 'Lac', 'NH3', 'AMY', 'ALP', 'AST',  'BIL-T','Ddimer',
                  'CK', 'Crea', 'Mg',  'K',  'BUN', 'PTINR', 'GGT', 'Lipase',  
                  'PTT', 'WBC']
    for ar in Right_skew:
        # hadmid_first_lab_less70[ar] = np.log(hadmid_first_lab_less70[ar])
        # hadmid_first_lab[ar].hist(bins=100)
        hadmid_first_lab.loc[:,[ar]] = np.log(hadmid_first_lab[[ar]])

    Right_skew2 = ['ALT','BIL-T', 'P','TnT','Eosin','Lym','PLT', 'FreeT4','Band','Myelo',
                   'Blast','UrineRBC', 'UrineWBC','BIL-D', 'PTINR',]
    for ar in Right_skew2:
       # hadmid_first_lab_less70[ar] = np.log(hadmid_first_lab_less70[ar])
        # hadmid_first_lab[ar].hist(bins=100)
        hadmid_first_lab.loc[:,[ar]] = np.sqrt(hadmid_first_lab[[ar]])
        
    Left_skew = ['Seg']
    for lr in Left_skew:
        # hadmid_first_lab_less70[ar] = np.log(hadmid_first_lab_less70[ar])
        # hadmid_first_lab[lr].hist(bins=100)
        hadmid_first_lab.loc[:,[lr]] = np.square(hadmid_first_lab[[lr]])
        
    filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
    icustays_select = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
    agegender = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
    vital_signs = pd.read_pickle(filepath)
    
    io_24 = icustays_select[['io_24']]
    
    structurals = [*agegender.keys(),*vital_signs.keys(),*hadmid_first_lab.keys(),*io_24.keys()]
    
    fig = plt.figure(figsize=(120,60),dpi=100)
    
    isna_ag = pd.read_csv('./isna_agegender.csv')
    isna_lab = pd.read_csv('./isna_hadmid_first_lab_percent.csv')
    isna_vs = pd.read_csv('./isna_vital_signs_percent.csv')
    
    isna_df = pd.concat([isna_ag,isna_lab,isna_vs],axis=0, ignore_index=True)
    isna_df.columns = ['kk','percentage']
    
    iooo = pd.DataFrame([['io_24',0.]])
    iooo.columns = ['kk','percentage']
    
    isna_df = pd.concat([isna_df,iooo],axis=0, ignore_index=True)
    isna_df = isna_df.sort_values('percentage',ascending=False)
    
    
    for p, k in enumerate(isna_df['kk']):        
        xi = p%10
        yi = int(p/10)
        ax = plt.subplot2grid((6,10),(yi,xi))
        
        dfs = [io_24,agegender,vital_signs,hadmid_first_lab]
        for df in dfs:
            if k in df.keys():
                dist = df[[k]]
            
                n, bins, patches = ax.hist(dist.values, 40, 
                                           density = 0.8,  
                                           # color ='green',  
                                           alpha = 0.7)
                
                mu = dist.mean().values
                sigma = dist.std().values
                  
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                     np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
                ax.plot(bins, y, '--', color ='black', linewidth=10) 
                ax.axes.yaxis.set_visible(False)
                plt.xticks(size = 35,rotation=45)
                plt.subplots_adjust(top = 0.95, hspace = 0.45)
                
                if k =='SEX':
                    plt.title(f'GENDER',fontsize=60)
                elif k == 'io_24':
                    plt.title(f'IO',fontsize=60)
                else:
                    plt.title(f'{k}',fontsize=60)
        
    plt.savefig('./data_distribution_sken_adjust.png') 
    
    hadmid_first_lab_less70 = hadmid_first_lab[temp]
    
    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab_skem_adjust.pdpkl')
    hadmid_first_lab.to_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab_skem_adjust_less70.pdpkl')
    hadmid_first_lab_less70.to_pickle(filepath)


def make_less_50_missing_rate_data():
    db_file_path = '../datahouse/mimic-iv-0.4'
    isna_lab = pd.read_csv('./isna_hadmid_first_lab_percent.csv')
    isna_lab.columns = ['name', 'p']
    temp = list(isna_lab[isna_lab['p']<0.3]['name'])

    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
    hadmid_first_lab = pd.read_pickle(filepath)
    
    #sken convert
    Right_skew = ['Glucose', 'Ca', 'Lac', 'NH3', 'AMY', 'ALP', 'AST',  'BIL-T','Ddimer',
                  'CK', 'Crea', 'Mg',  'K',  'BUN', 'PTINR', 'GGT', 'Lipase',  
                  'PTT', 'WBC']
    for ar in Right_skew:
        # hadmid_first_lab_less70[ar] = np.log(hadmid_first_lab_less70[ar])
        # hadmid_first_lab[ar].hist(bins=100)
        hadmid_first_lab.loc[:,[ar]] = np.log(hadmid_first_lab[[ar]])

    Right_skew2 = ['ALT','BIL-T', 'P','TnT','Eosin','Lym','PLT', 'FreeT4','Band','Myelo',
                   'Blast','UrineRBC', 'UrineWBC','BIL-D', 'PTINR',]
    for ar in Right_skew2:
       # hadmid_first_lab_less70[ar] = np.log(hadmid_first_lab_less70[ar])
        # hadmid_first_lab[ar].hist(bins=100)
        hadmid_first_lab.loc[:,[ar]] = np.sqrt(hadmid_first_lab[[ar]])
        
    Left_skew = ['Seg']
    for lr in Left_skew:
        # hadmid_first_lab_less70[ar] = np.log(hadmid_first_lab_less70[ar])
        # hadmid_first_lab[lr].hist(bins=100)
        hadmid_first_lab.loc[:,[lr]] = np.square(hadmid_first_lab[[lr]])
          
    hadmid_first_lab_less70 = hadmid_first_lab[temp]
    
    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab_skem_adjust_less30.pdpkl')
    hadmid_first_lab_less70.to_pickle(filepath)

def make_less_50_missing_rate_data():
    db_file_path = '../datahouse/mimic-iv-0.4'
    isna_lab = pd.read_csv('./isna_hadmid_first_lab_percent.csv')
    isna_lab.columns = ['name', 'p']
    temp = list(isna_lab[isna_lab['p']<0.5]['name'])

    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
    hadmid_first_lab = pd.read_pickle(filepath)
             
    hadmid_first_lab_less70 = hadmid_first_lab[temp]
    
    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab_less50.pdpkl')
    hadmid_first_lab_less70.to_pickle(filepath)

    
def draw_tsne_type_token():
    # load data
    db_file_path = '../datahouse/mimic-iv-0.4'

    filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
    icustays_select = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
    agegender = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
    vital_signs = pd.read_pickle(filepath)

    filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
    hadmid_first_lab = pd.read_pickle(filepath)    

    agegender_keys = agegender.keys()
    vital_signs_keys = vital_signs.keys()
    hadmid_first_lab_keys = hadmid_first_lab.keys()
    io_24_keys = icustays_select[['io_24']].keys()
    
    
    structurals = [*agegender_keys,*vital_signs_keys,*hadmid_first_lab_keys,*io_24_keys]
    structurals_idx = pd.DataFrame(structurals,index=structurals)
    structurals_idx.columns = ['name']
    structurals_idx['s_idx'] = 10+np.arange(len(structurals))
    
    checkpoint_file = f'../checkpoint_EDs_OnlyS/origin/0/EDisease_spectrum_TS'
    
    import EDisease_model_v001 as ED_model
    from EDisease_config import EDiseaseConfig, StructrualConfig
    
    S_config = StructrualConfig()
    stc2emb = ED_model.structure_emb(S_config)
    stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
        
    type_embedding = stc2emb.BERTmodel.embeddings.position_embeddings(torch.tensor(structurals_idx['s_idx'].values))
    
    structurals_idx['idx'] = np.arange(60)
    
    structurals_idx = structurals_idx.set_index('idx')
    
    from sklearn import manifold
    from sklearn.decomposition import PCA
    import matplotlib.patheffects as PathEffects

    pca = PCA(n_components=2)
    pca.fit(type_embedding.detach().numpy()) 
    XY_pca = pca.transform(type_embedding.detach().numpy())

    xy = XY_pca
  
    fig = plt.figure(figsize=(15,15),dpi=100)
    
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(xy[:,0], xy[:,1],s=150)
    ax.axis('off')
    ax.axis('tight')
    
    for i in range(len(structurals_idx)):
    # Position of each label.
        txt = ax.text(xy[i,0], xy[i,1], structurals_idx.loc[i]['name'], fontsize=10)
        # txt.set_path_effects([
        #     PathEffects.Stroke(linewidth=5, foreground="w"),
        #     PathEffects.Normal()])
    plt.savefig(f'./pic_type/0pca.png')

    for j in range(10):

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=j, n_iter=2500)
        T_tsne = tsne.fit_transform(type_embedding.detach().numpy())
       
        xy = T_tsne
      
        fig = plt.figure(figsize=(15,15),dpi=100)
        
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(xy[:,0], xy[:,1],s=150)
        ax.axis('off')
        ax.axis('tight')
        
        for i in range(len(structurals_idx)):
        # Position of each label.
            txt = ax.text(xy[i,0], xy[i,1], structurals_idx.loc[i]['name'], fontsize=10)
            # txt.set_path_effects([
            #     PathEffects.Stroke(linewidth=5, foreground="w"),
            #     PathEffects.Normal()])
        plt.savefig(f'./pic_type/0{j}.png')
