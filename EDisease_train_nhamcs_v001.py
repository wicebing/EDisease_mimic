import os
import time
import unicodedata
import random
import string
import re
import sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AutoConfig, AutoTokenizer, AutoModel, BertConfig, BertModel
import torch

from EDisease_utils import count_parameters, save_checkpoint, load_checkpoint
from EDisease_config import EDiseaseConfig, StructrualConfig
import EDisease_model_v001 as ED_model

import AIED_dataloader_nhamcs as dataloader

task ='test'

batch_size = 1024
device = 'cuda'
parallel = False

checkpoint_file = './checkpoint_emb/aRevision_dc'
alpha=1
beta=1
gamma=0.1

def make_trg(sample, train_cls=True):
    s_ = sample['structure']
    trg = sample['trg']
    '''
        trg = e_patient[['COMPUTEREDTRIAGE',
                         'TRIAGE',
                         'Hospital',
                         'icu7',
                         'death7',
                         'Age',
                         'Sex',
                         'cva',
                         'trauma',
                         'query']]     '''
    trg = trg.long()
    trg_triage_ = trg[:,1]
    trg_hospital = trg[:,2]
    trg_icu7_ = trg[:,3]
    trg_die7_ = trg[:,4]
    trg_cva = trg[:,7]
    
    age = trg[:,5]
    sex = trg[:,6]
    
    trg_triage = (trg_triage_<3).long()

    
    temp_icu = trg_icu7_<8
    temp_die = trg_die7_<8
    trg_icuANDdie7 = temp_icu | temp_die
    
    trg_icu7 = temp_icu.long()
    trg_die7 = temp_die.long()    
    trg_icuANDdie7 = trg_icuANDdie7.long()

#    age_ = (s_[:,0]*28.31)+43.71
#    sex_ = s_[:,1]
    
#    age = (age_/10).int()
#    sex = (sex_>0.5).int()
                
    trg_cls = {'trg_triage':trg_triage,
               'trg_icu7':trg_icu7,
               'trg_die7':trg_die7,
               'trg_icudeath7':trg_icuANDdie7,
               'cls_hospital':trg_hospital,
               'cls_cva':trg_cva,
               'cls_age':age,
               'cls_sex':sex
              }
        
    return trg_cls

def train_NHAMCS(EDisease_Model,
                 baseBERT,
                 dloader,
                 lr=1e-4,
                 epoch=100,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 ): 
    
    EDisease_Model.to(device)
    baseBERT.to(device)
        
    model_optimizer = optim.Adam(EDisease_Model.parameters(), lr=lr)
    if parallel:
        EDisease_Model = torch.nn.DataParallel(EDisease_Model)
        baseBERT = torch.nn.DataParallel(baseBERT)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    #criterion = nn.MSELoss()
    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            #print(iteration, sample['idx'])
            model_optimizer.zero_grad()
            loss = 0
            
            output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), nohx, expand_data = EDisease_Model(baseBERT,sample,noise_scale=noise_scale,mask_ratio=mask_ratio,use_pi=False,)

            aug2 = 2*random.random()
            output2,EDisease2, (s,input_emb2,input_emb_org2), _,_, _, _ = EDisease_Model(baseBERT,sample,noise_scale=aug2*noise_scale,mask_ratio=mask_ratio,use_pi=False)       
            
            bs = len(s)            

            loss_dim = dim_model(output[:,:1],
                             input_emb_org,
                             CLS_emb_emb,
                             nohx,
                             mask_ratio=mask_ratio,
                             mode=mode,
                             ptloss=ptloss,
                             DS_model=DS_model,
                             mix_ratio=mix_ratio,
                             EDisease2=output2[:,:1],
                             shuffle=True,
                             use_pi=use_pi,
                             yespi=yespi,
                             ep=ep
                            )            


        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease.pth',
                            model=EDisease_Model,
                            parallel=parallel)


            print('======= epoch:%i ========'%ep)
            
        loss_pathName = '{:.0f}'.format(time.time())         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss.csv', sep = ',')
    print(total_loss) 

        
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
            
if task=='pickle_nhamcs_cls_dim_val':
    batch_size = 16
    use_pi= False
    parallel = False
    device = 'cpu'

    model_name = "bert-base-multilingual-cased"
    
    baseBERT = ED_model.adjBERTmodel(bert_ver=model_name,fixBERT=False)
    
    BERT_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    T_config = EDiseaseConfig()
    S_config = StructrualConfig()
    
    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config,
                                             tokanizer=BERT_tokenizer,
                                             device=device)

    dim_model = ED_model.DIM(T_config=T_config,
                              device=device,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)

# ====
    all_datas = dataloader.load_datas()
    datas_train = all_datas['datas_train']
    dm_normalization_np = all_datas['dm_normalization_np']   
    datas_test = all_datas['datas_test']
    datas_val = all_datas['datas_val']   
    datas_all = all_datas['datas']
    
    # datas_val = datas_val[datas_val['AGE']>=18]
    
    data15_triage_val_sample = datas_val.reset_index()

    EDEW_DS_val = dataloader.EDEW_Dataset(ds= data15_triage_val_sample,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL_val = DataLoader(dataset = EDEW_DS_val,
                         shuffle = False,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
# ====

    print('** Start load pickle **')
    
    pklfile = 'nhamcs.pickle'
    with open(pklfile,'rb') as f:
        data15_triage_train = pickle.load(f)  
    print('** complete load pickle **')
    
    if not use_pi:
        data15_triage_train['piemb']=0
        
    structural = ['AGE',
                'SEX',
                'GCS',
                'BPSYS',
                'BPDIAS',
                'PULSE',
                'POPCT',
                'RESPR',
                'TEMPF',
                'HEIGHT', 
                'WEIGHT',
                'PAINSCALE',
                'BE',
                'BV',
                'BM'
                ]
        
    dm = data15_triage_train[structural]
    dm_mean = dm.mean(axis=0, skipna=True)
    dm_std = dm.std(axis=0, skipna=True)
    dm_normalization = pd.concat([dm_mean,dm_std],axis=1)
    print('trainset',dm_normalization)
    dm_normalization_np = np.array(dm_normalization).T  
    
    
    # data15_triage_train = data15_triage_train[data15_triage_train['AGE']>=18]
    
    data15_triage_train = data15_triage_train.reset_index()
                             
    rn0 = data15_triage_train['DIEDED']>7
    rp0 = data15_triage_train['DIEDED']<8
    
    rn1 = data15_triage_train['ICU']>7
    rp1 = data15_triage_train['ICU']<8
    
    rn2 = rn1 & rn0
    rp2 = rp1 | rp0    

    # g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    # g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    # g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    # g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    # ltemp = [g00] + [g01]*int(len(g00)/len(g01)) +[g10]*int(len(g00)/len(g10)) +[g11]*int(len(g00)/len(g11))
    # print(len(g00),len(g01),len(g10),len(g11))
    # print(' *** balance the dataset p/n *** ')
    # dtemp = pd.concat(ltemp)  
    # print(' *** balance complete *** ')

    g00 = pd.Series(data15_triage_train[rn0 & rn1].index)
    g01 = pd.Series(data15_triage_train[rn0 & rp1].index)
    g10 = pd.Series(data15_triage_train[rp0 & rn1].index)
    g11 = pd.Series(data15_triage_train[rp0 & rp1].index)

    ltemp = [g00] + ([g01]*2 + [g10]*int(2*len(g01)/len(g10)) +[g11]*300)*16  #16for all 14for adult
    print(len(g00),len(g01),len(g10),len(g11))
    print(' *** balance the dataset p/n *** ')
    dtemp = pd.concat(ltemp)  
    print(' *** balance complete *** ')

    EDEW_DS = AIED_dataloader.pickle_Dataset(ds= data15_triage_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,
                                           dsidx=dtemp,)  

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                         shuffle = True,
                         num_workers=10,
                         batch_size=batch_size,
                         )

    train_NHAMCS_cls_dim_val(DS_model=emb_model,
                     cls_model=cls_model,
                     dim_model=dim_model,
                     baseBERT=baseBERT,
                     dloader=EDEW_DL,
                     dloader_val=EDEW_DL_val,
                     noise_scale=0.05,
                     mask_ratio=0.33,
                     lr=2e-5,
                     epoch=100001,
                     log_interval=15,
                     parallel=parallel,
                     trainED=True,
                     task=None,
                     checkpoint_file=checkpoint_file,
                     use_pi=use_pi) 