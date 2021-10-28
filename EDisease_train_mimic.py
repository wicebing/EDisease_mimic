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

import EDisease_dataloader_mimic4_001 as dataloader

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

# fix the BERT version
model_name = "bert-base-multilingual-cased"
T_config = EDiseaseConfig()
S_config = StructrualConfig()

baseBERT = ED_model.adjBERTmodel(bert_ver=model_name,T_config=T_config,fixBERT=True)
BERT_tokenizer = AutoTokenizer.from_pretrained(model_name)

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

filepath = os.path.join(db_file_path, 'data_EDis', 'diagnoses_icd_merge_dropna.pdpkl')
diagnoses_icd_merge_dropna = pd.read_pickle(filepath)

# split the dataset
train_set_hadmid = hadmid_first_lab.sample(frac=0.85,random_state=0).index
temp_set_hadmid = hadmid_first_lab.drop(train_set_hadmid)
val_set_hadmid = temp_set_hadmid.sample(frac=0.25,random_state=0).index
test_set_hadmid = temp_set_hadmid.drop(val_set_hadmid).index

# get the training set mean/std
train_set_lab_mean = hadmid_first_lab.loc[train_set_hadmid].mean()
train_set_lab_std = hadmid_first_lab.loc[train_set_hadmid].std()
hadmid_first_lab_keys = hadmid_first_lab.keys()

trainset_temp = pd.DataFrame(train_set_hadmid)
trainset_temp.columns = ['hadm_id']
trainset_temp = trainset_temp.merge(icustays_select,how='left',on=['hadm_id'])

io_24_keys = trainset_temp[['io_24']].keys()
io_24_mean = trainset_temp[['io_24']].mean()
io_24_std = trainset_temp[['io_24']].std()

trainset_subjectid = trainset_temp[['subject_id']].drop_duplicates()
agegender_keys = agegender.keys()
agegender_ridx = agegender.reset_index()
agegender_ridx.columns = ['subject_id',*agegender_keys]
trainset_agegender_temp =  trainset_subjectid.merge(agegender_ridx,how='left',on=['subject_id'])
trainset_agegender_temp = trainset_agegender_temp.set_index('subject_id')

train_set_agegender_mean = trainset_agegender_temp.mean()
train_set_agegender_std = trainset_agegender_temp.std()

trainset_stayid = trainset_temp[['stay_id']].drop_duplicates()
vital_signs_keys = vital_signs.keys()
vital_signs_ridx = vital_signs.reset_index()
vital_signs_ridx.columns= ['stay_id',*vital_signs_keys]
trainset_vitalsign_temp = trainset_stayid.merge(vital_signs_ridx,how='left',on=['stay_id'])
trainset_vitalsign_temp = trainset_vitalsign_temp.set_index('stay_id')

train_set_vitalsign_mean = trainset_vitalsign_temp.mean()
train_set_vitalsign_std = trainset_vitalsign_temp.std()

# remove the duplicates
icustays_select_sort = icustays_select.sort_values(['intime'])
icustays_select_sort_dropduplicate = icustays_select_sort.drop_duplicates(subset=['hadm_id'])
icustays_select_sort_dropduplicate = icustays_select_sort_dropduplicate.set_index('hadm_id')

structurals = [*agegender_keys,*vital_signs_keys,*hadmid_first_lab_keys,*io_24_keys]
structurals_idx = pd.DataFrame(structurals,index=structurals)
structurals_idx.columns = ['name']
structurals_idx['s_idx'] = 10+np.arange(len(structurals))

ds_train = dataloader.mimic_Dataset(set_hadmid=train_set_hadmid,
                                    icustays_select=icustays_select_sort_dropduplicate,
                                    agegender=agegender,
                                    vital_signs=vital_signs,
                                    hadmid_first_lab=hadmid_first_lab,
                                    diagnoses_icd_merge_dropna=diagnoses_icd_merge_dropna,
                                    tokanizer=BERT_tokenizer,
                                    train_set_lab_mean=train_set_lab_mean,
                                    train_set_lab_std=train_set_lab_std,
                                    train_set_agegender_mean=train_set_agegender_mean,
                                    train_set_agegender_std=train_set_agegender_std,
                                    train_set_vitalsign_mean=train_set_vitalsign_mean,
                                    train_set_vitalsign_std=train_set_vitalsign_std,
                                    io_24_mean=io_24_mean,
                                    io_24_std=io_24_std,
                                    structurals_idx=structurals_idx,
                                    dsidx=None)

ds_valid = dataloader.mimic_Dataset(set_hadmid=val_set_hadmid,
                                    icustays_select=icustays_select_sort_dropduplicate,
                                    agegender=agegender,
                                    vital_signs=vital_signs,
                                    hadmid_first_lab=hadmid_first_lab,
                                    diagnoses_icd_merge_dropna=diagnoses_icd_merge_dropna,
                                    tokanizer=BERT_tokenizer,
                                    train_set_lab_mean=train_set_lab_mean,
                                    train_set_lab_std=train_set_lab_std,
                                    train_set_agegender_mean=train_set_agegender_mean,
                                    train_set_agegender_std=train_set_agegender_std,
                                    train_set_vitalsign_mean=train_set_vitalsign_mean,
                                    train_set_vitalsign_std=train_set_vitalsign_std,
                                    io_24_mean=io_24_mean,
                                    io_24_std=io_24_std,
                                    structurals_idx=structurals_idx,
                                    dsidx=None)

ds_test  = dataloader.mimic_Dataset(set_hadmid=test_set_hadmid,
                                    icustays_select=icustays_select_sort_dropduplicate,
                                    agegender=agegender,
                                    vital_signs=vital_signs,
                                    hadmid_first_lab=hadmid_first_lab,
                                    diagnoses_icd_merge_dropna=diagnoses_icd_merge_dropna,
                                    tokanizer=BERT_tokenizer,
                                    train_set_lab_mean=train_set_lab_mean,
                                    train_set_lab_std=train_set_lab_std,
                                    train_set_agegender_mean=train_set_agegender_mean,
                                    train_set_agegender_std=train_set_agegender_std,
                                    train_set_vitalsign_mean=train_set_vitalsign_mean,
                                    train_set_vitalsign_std=train_set_vitalsign_std,
                                    io_24_mean=io_24_mean,
                                    io_24_std=io_24_std,
                                    structurals_idx=structurals_idx,
                                    dsidx=None)
    
def train_NHAMCS(EDisease_Model,
                 stc2emb,
                 emb_emb,
                 dim_model,
                 baseBERT,
                 dloader,
                 lr=1e-4,
                 epoch=100,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 normalization=None
                 ): 
    
    EDisease_Model.to(device)
    stc2emb.to(device)
    emb_emb.to(device)
    dim_model.to(device)
    baseBERT.to(device)
    
    baseBERT.eval()
        
    model_optimizer = optim.Adam(EDisease_Model.parameters(), lr=lr)
    model_optimizer_s2e = optim.Adam(stc2emb.parameters(), lr=lr)
    model_optimizer_e2e = optim.Adam(emb_emb.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    
    if parallel:
        EDisease_Model = torch.nn.DataParallel(EDisease_Model)
        stc2emb = torch.nn.DataParallel(stc2emb)
        emb_emb = torch.nn.DataParallel(emb_emb)
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
            model_optimizer_s2e.zero_grad()
            model_optimizer_e2e.zero_grad()
            
            loss = torch.tensor(0.)

            sample = {k:v.to(device) for k,v in sample.items()}
            
            # for text data
            c,cm,h,hm = sample['cc'],sample['mask_cc'],sample['ehx'],sample['mask_ehx']
            output = baseBERT(c.long(),cm.long())
            c_emb = output['heads']
            
            c_emb_emb = emb_emb(c_emb)

            output = baseBERT(h.long(),hm.long())
            h_emb = output['heads']
            em_h_emb = emb_emb(h_emb)
            
            cumsum_hx_n = torch.cumsum(sample['stack_hx_n'],0)
            h_emb_mean_ = []
            for i,e in enumerate(cumsum_hx_n):            
                if sample['stack_hx_n'][i]>1:
                    h_mean = torch.mean(em_h_emb[1:cumsum_hx_n[i]],dim=0) if i < 1 else torch.mean(em_h_emb[1+cumsum_hx_n[i-1]:cumsum_hx_n[i]],dim=0)
                    h_emb_mean_.append(h_mean)
                # else:
                #     h_emb_mean_.append(em_PAD_emb.view(em_h_emb[0].shape))
                    
            h_emb_emb = torch.stack(h_emb_mean_)
            
            # for structual data
            s,sp, sm = sample['structure'],sample['structure_position_ids'], sample['structure_attention_mask']
                  
            if normalization is None:
                s_noise = s
            else:
                #normalization = torch.tensor(normalization).expand(s.shape).to(self.device)
                normalization = torch.ones(s.shape).to(device)
                noise_ = normalization*noise_scale*torch.randn_like(s,device=device)
                s_noise = s+noise_
                
            
            s_emb = stc2emb(inputs=s_noise,
                                 attention_mask=sm,
                                 position_ids=sp)
            s_emb_org = stc2emb(inputs=s,
                                 attention_mask=sm,
                                 position_ids=sp)
            
            # make EDisease input data
            things = {'s':{'emb':s_emb,
                           'attention_mask':torch.ones(s_emb.shape[:2],device=device),
                           'position_id':5*torch.ones(h_emb_emb.unsqueeze(1).shape[:2],device=device)
                           },
                      'c':{'emb':c_emb_emb.unsqueeze(1),
                           'attention_mask':torch.ones(c_emb_emb.unsqueeze(1).shape[:2],device=device),
                           'position_id':6*torch.ones(c_emb_emb.unsqueeze(1).shape[:2],device=device)
                           },
                      'h':{'emb':h_emb_emb.unsqueeze(1),
                           'attention_mask':torch.ones(h_emb_emb.unsqueeze(1).shape[:2],device=device),
                           'position_id':7*torch.ones(h_emb_emb.unsqueeze(1).shape[:2],device=device)
                           },
                      }

            outp = EDisease_Model(things,
                                  noise_scale=noise_scale,
                                  mask_ratio=mask_ratio
                                  )
            EDisease = outp['EDisease']
            input_emb_org = outp['input_emb_org']
            position_ids = outp['position_ids']
            attention_mask = outp['attention_mask']

            aug2 = 2*random.random()
            outp2 = EDisease_Model(things=things,
                                   noise_scale=aug2*noise_scale,
                                   mask_ratio=mask_ratio
                                   )
            EDisease2 = outp2['EDisease']
            EDiseaseFake = torch.cat([EDisease[1:],EDisease[:1]],dim=0)
            
            bs = len(sample['structure'])            

            mode = 'D' if batch_idx%2==0 else 'G'
            ptloss = True if batch_idx%99==3 else False
            
            things_e = {'e':{'emb':EDisease.unsqueeze(1),
                             'emb2':EDisease2.unsqueeze(1),
                             'embf':EDiseaseFake.unsqueeze(1),
                             'attention_mask':torch.ones(EDisease.unsqueeze(1).shape[:2],device=device),
                             'position_id':4*torch.ones(EDisease.unsqueeze(1).shape[:2],device=device)
                             }
                        }

            loss_dim = dim_model(things=things,
                                 things_e=things_e,
                                 soft=0.7, 
                                 mask_ratio=mask_ratio,
                                 mode=mode, 
                                 ptloss=ptloss,
                                 ep=ep)
            
            loss = loss_dim
                
            loss.sum().backward()
            model_optimizer.step()
            model_optimizer_dim.step()
            model_optimizer_s2e.step()
            model_optimizer_e2e.step()
                
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs

        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease_Model.pth',
                            model=EDisease_Model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='dim_model.pth',
                            model=dim_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='stc2emb.pth',
                            model=stc2emb,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='emb_emb.pth',
                            model=emb_emb,
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

    
    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
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
                 stc2emb=stc2emb,
                 emb_emb=emb_emb,
                 dim_model=dim_model,
                 baseBERT=baseBERT,
                 dloader=EDEW_DL_train,
                 lr=1e-5,
                 epoch=100,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 normalization=dm_normalization_np) 