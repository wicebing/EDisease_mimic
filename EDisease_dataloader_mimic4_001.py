import pandas as pd
import numpy as np
import glob
import os
import random
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
import torch

# === impoert BERT ===
BERT_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#BERT_model = BertModel.from_pretrained("bert-base-multilingual-cased")
# ====================

class mimic_Dataset(Dataset):
    def __init__(self,
                 set_hadmid,
                 icustays_select,
                 agegender,
                 vital_signs,
                 hadmid_first_lab,
                 diagnoses_icd_merge_dropna,
                 tokanizer,
                 train_set_lab_mean,
                 train_set_lab_std,
                 train_set_agegender_mean,
                 train_set_agegender_std,
                 train_set_vitalsign_mean,
                 train_set_vitalsign_std,
                 io_24_mean,
                 io_24_std,
                 structurals_idx,
                 dsidx=None):
       
        self.set_hadmid = set_hadmid
        self.icustays_select = icustays_select
        self.agegender = agegender
        self.vital_signs = vital_signs
        self.hadmid_first_lab = hadmid_first_lab
        self.diagnoses_icd_merge_dropna = diagnoses_icd_merge_dropna
        self.tokanizer = tokanizer
        
        self.train_set_lab_mean = train_set_lab_mean
        self.train_set_lab_std = train_set_lab_std
        self.train_set_agegender_mean = train_set_agegender_mean
        self.train_set_agegender_std = train_set_agegender_std
        self.train_set_vitalsign_mean = train_set_vitalsign_mean
        self.train_set_vitalsign_std = train_set_vitalsign_std
        self.io_24_mean = io_24_mean
        self.io_24_std = io_24_std
        
        self.structurals_idx = structurals_idx
        
        self.dsidx = dsidx
        
        if dsidx is None:
            self.len = len(set_hadmid)
        else:
            self.len = len(dsidx)
    
    def __getitem__(self, index):
        if self.dsidx is None:
            hadm_id = self.set_hadmid[index]
        else: 
            hadm_id = self.set_hadmid[self.dsidx[index]]        
        
        sample = self.icustays_select.loc[hadm_id]
        subject_id = sample['subject_id']
        stay_id = sample['stay_id']
        intime = sample['intime']
        
        los = sample['los']
        io24 = sample['io_24']
        io_norm = (io24 - self.io_24_mean)/self.io_24_std
        
        vs = self.vital_signs.loc[stay_id]
        vs_norm = (vs - self.train_set_vitalsign_mean)/self.train_set_vitalsign_std
        
        ag = self.agegender.loc[subject_id]
        ag_norm = (ag - self.train_set_agegender_mean)/self.train_set_agegender_std
        
        lab =self.hadmid_first_lab.loc[hadm_id]
        lab_norm =(lab-self.train_set_lab_mean)/self.train_set_lab_std
        
        structural_norm = pd.concat([ag_norm,vs_norm,lab_norm,io_norm],axis=0)
        self.structurals_idx['value'] = structural_norm
        
        self.structurals_idx['missing_value'] = (~self.structurals_idx['value'].isna()).astype(int)
        
        # for mean imputation
        self.structurals_idx = self.structurals_idx.fillna(0)
        
        # text data
        diagnoses_icd = self.diagnoses_icd_merge_dropna[self.diagnoses_icd_merge_dropna['hadm_id']==hadm_id]
        
        icd_n = len(diagnoses_icd)
                
        hx_token_ids = []
        
        for i in range(icd_n):
            h = diagnoses_icd.iloc[i]['long_title']
            hx_tokens = self.tokanizer.tokenize(h)
            
            # add BERT cls head
            hx_tokens = [self.tokanizer.cls_token, *hx_tokens]
            hx_tokens = hx_tokens[:512]
            hx_token_ids_ = self.tokanizer.convert_tokens_to_ids(hx_tokens)
            hx_token_ids.append(torch.tensor(hx_token_ids_,dtype=torch.float32))
        

            
        structure_tensor = torch.tensor(self.structurals_idx['value'],dtype=torch.float32)
        structure_attention_mask_tensor = torch.tensor(self.structurals_idx['missing_value'],dtype=torch.long)
        structure_position_ids_tensor = torch.tensor(self.structurals_idx['s_idx'],dtype=torch.long)
        
        trg = los
        trg_tensor = torch.tensor(los,dtype=torch.float32)

        
        datas = {'structure': structure_tensor,
                 'structure_attention_mask': structure_attention_mask_tensor,
                 'structure_position_ids': structure_position_ids_tensor,
                 'hx': hx_token_ids,
                 'trg':trg_tensor,
                 }             
          
        return datas
    
    def __len__(self):
        return self.len

def collate_fn(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    structure = [DD['structure'] for DD in datas]
    structure_attention_mask = [DD['structure_attention_mask'] for DD in datas]
    structure_position_ids = [DD['structure_position_ids'] for DD in datas]
    stack_hx_ = [DD['hx'] for DD in datas]
    trg = [DD['trg'] for DD in datas]
    
    stack_hx = []
    hx_n = []
    for shx in stack_hx_:
        hx_n.append(len(shx))
        for eshx in shx:
            stack_hx.append(eshx)
       
    ehx = stack_hx
    origin_ehx_length = [len(d) for d in ehx]
    # origin_pi_length = [len(d) for d in pi]
    
    ehx = pad_sequence(ehx,
                      batch_first = True,
                      padding_value=0)

    
    mask_padding_ehx = torch.zeros(ehx.shape,dtype=torch.long)
    
    for i,e in enumerate(origin_ehx_length):
        mask_padding_ehx[i,:e] = 1
        
    batch['structure'] = torch.stack(structure)
    batch['structure_attention_mask'] = torch.stack(structure_attention_mask)
    batch['structure_position_ids'] = torch.stack(structure_position_ids)
    batch['ehx'] = ehx
    
    batch['mask_ehx'] = mask_padding_ehx
    
    batch['stack_hx_n'] = torch.tensor(hx_n)
    batch['origin_ehx_length'] = torch.tensor(origin_ehx_length)
    
    batch['trg'] = torch.stack(trg)    
    return batch

