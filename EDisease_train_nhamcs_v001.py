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

from EDisease_utils import count_parameters, save_checkpoint, load_checkpoint
from EDisease_config import EDiseaseConfig, StructrualConfig
import EDisease_model_v001 as ED_model

import AIED_dataloader_nhamcs as dataloader

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test_nhamcs_cls'

batch_size = 8
device = 'cuda'
parallel = False

checkpoint_file = '../checkpoint_EDs/test01'
alpha=1
beta=1
gamma=0.1

if not os.path.isdir(checkpoint_file):
    os.makedirs(checkpoint_file)
    print(f' make dir {checkpoint_file}')
    
def train_NHAMCS(EDisease_Model,
                 dim_model,
                 baseBERT,
                 tokanizer,
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
    dim_model.to(device)
    baseBERT.to(device)
    
    baseBERT.eval()
        
    model_optimizer = optim.Adam(EDisease_Model.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        EDisease_Model = torch.nn.DataParallel(EDisease_Model)
        dim_model = torch.nn.DataParallel(dim_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)
            
    total_loss = []
    
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            model_optimizer.zero_grad()
            model_optimizer_dim.zero_grad()
            loss = torch.tensor(0.)

            sample = {k:v.to(device) for k,v in sample.items()}
            
            c,cm,h,hm = sample['cc'],sample['mask_cc'],sample['ehx'],sample['mask_ehx']
            output = baseBERT(c.long(),cm.long())
            c_emb_emb = output['em_heads']
            em_CLS_emb = output['em_CLS_emb'][:1]
            em_SEP_emb = output['em_SEP_emb'][:1]
            em_PAD_emb = output['em_PAD_emb'][:1]
                        
            output = baseBERT(h.long(),hm.long())
            em_h_emb = output['em_heads']
            
            cumsum_hx_n = torch.cumsum(sample['stack_hx_n'],0)
            h_emb_mean_ = []
            for i,e in enumerate(cumsum_hx_n):            
                if sample['stack_hx_n'][i]>1:
                    h_mean = torch.mean(em_h_emb[1:cumsum_hx_n[i]],dim=0) if i < 1 else torch.mean(em_h_emb[1+cumsum_hx_n[i-1]:cumsum_hx_n[i]],dim=0)
                    h_emb_mean_.append(h_mean)
                else:
                    h_emb_mean_.append(em_PAD_emb.view(em_h_emb[0].shape))
                    
            h_emb_emb = torch.stack(h_emb_mean_)  

            CLS_emb_emb = em_CLS_emb.expand(c_emb_emb.shape)
            SEP_emb_emb = em_SEP_emb.expand(c_emb_emb.shape)
            PAD_emb_emb = em_PAD_emb.expand(c_emb_emb.shape)

            output,EDisease, (s,input_emb,input_emb_org,position_ids,attention_mask), (CLS_emb_emb,SEP_emb_emb)= EDisease_Model(sample,
                                                                                                                                CLS_emb_emb,
                                                                                                                                SEP_emb_emb,
                                                                                                                                PAD_emb_emb,
                                                                                                                                c_emb_emb,
                                                                                                                                h_emb_emb,
                                                                                                                                noise_scale=noise_scale,
                                                                                                                                mask_ratio=mask_ratio,
                                                                                                                                use_pi=False,)

            aug2 = 2*random.random()
            _,EDisease2,_,_ = EDisease_Model(sample,
                                             CLS_emb_emb,
                                             SEP_emb_emb,
                                             PAD_emb_emb,
                                             c_emb_emb,
                                             h_emb_emb,
                                             noise_scale=aug2*noise_scale,mask_ratio=mask_ratio,use_pi=False)       
            
            bs = len(s)            

            mode = 'D' if batch_idx%2==0 else 'G'
            ptloss = True if batch_idx%99==3 else False

            loss_dim = dim_model(EDisease=EDisease, 
                                 M=input_emb_org,
                                 SEP_emb_emb=CLS_emb_emb,
                                 nohx=sample['stack_hx_n'],
                                 position_ids=position_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=None,
                                 soft=0.7, 
                                 mask_ratio=mask_ratio,
                                 mode=mode, 
                                 ptloss=ptloss, 
                                 EDisease2=EDisease2,
                                 ep=ep)
            
            loss = loss_dim
                
            loss.sum().backward()
            model_optimizer.step()
            model_optimizer_dim.step()
                
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs

        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease.pth',
                            model=EDisease_Model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
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
            
if task=='nhamcs_train':
    model_name = "bert-base-multilingual-cased"
    T_config = EDiseaseConfig()
    S_config = StructrualConfig()
    
    baseBERT = ED_model.adjBERTmodel(bert_ver=model_name,T_config=T_config,fixBERT=True)
    
    BERT_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
    
    EDEW_DS_train = dataloader.EDEW_Dataset(ds= datas_train,          
                                           tokanizer= BERT_tokenizer,
                                           normalization = dm_normalization_np,)  
    
    EDEW_DL_train = DataLoader(dataset = EDEW_DS_train,
                         shuffle = True,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)

    train_NHAMCS(EDisease_Model=EDisease_Model,
                 dim_model=dim_model,
                 baseBERT=baseBERT,
                 tokanizer=BERT_tokenizer,
                 dloader=EDEW_DL_train,
                 lr=1e-5,
                 epoch=100,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file) 