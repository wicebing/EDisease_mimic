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
            hadm_id = self.dsidx[index]
        
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
        hx_token_ids.append(torch.tensor([101,0,0,0,0],dtype=torch.float32))
        
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



class mimic_time_sequence_Dataset(Dataset):
    def __init__(self,
                 set_hadmid,
                 icustays_select,
                 agegender,
                 timesequence_vital_signs,
                 timesequence_lab,
                 diagnoses_icd_merge_dropna,
                 tokanizer,
                 structurals_idx,
                 dsidx=None):
       
        self.set_hadmid = set_hadmid
        self.icustays_select = icustays_select
        self.agegender = agegender
        self.timesequence_vital_signs = timesequence_vital_signs
        self.timesequence_lab = timesequence_lab
        self.diagnoses_icd_merge_dropna = diagnoses_icd_merge_dropna
        self.tokanizer = tokanizer            
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
            hadm_id = self.dsidx[index]
        
        sample = self.icustays_select.loc[hadm_id]
        subject_id = sample['subject_id']
        stay_id = sample['stay_id']
        intime = sample['intime']
        
        los = sample['los']
        
        io24 = sample['io_24']
       
        labevents_merge_dropna_clean_combine = self.timesequence_lab
        temp = labevents_merge_dropna_clean_combine[labevents_merge_dropna_clean_combine['hadm_id']==hadm_id]
        temp = temp.sort_values(by=['charttime'])
        # temp_first = temp.drop_duplicates(keep='first',subset=['bb_idx'])
          
        temp_filter_24 = (temp['charttime'] - intime) < datetime.timedelta(minutes=1440)
        temp_24 = temp[temp_filter_24]
        temp_24['time'] = (temp_24['charttime'] - intime)
        temp_24['time_day'] = temp_24['time'].dt.total_seconds()/(60*60*24)
        
        temp_select = temp_24[['bb_idx','valuenum','time_day']]
        t_idx = temp_select[temp_select['time_day']<0].index
        temp_select.loc[t_idx,['time_day']] = 0
        
        if len(temp_select)>500:
            temp_select = temp_select.iloc[:500]
 
        # add io
        temp_select = temp_select.append(pd.DataFrame([['io_24',io24,1.,]],columns=temp_select.columns))
        
        temp_select_idx_m_s = temp_select.merge(self.structurals_idx,how='left',on='bb_idx')
        
        temp_select_idx_m_s['n_value'] = (temp_select_idx_m_s['valuenum']-temp_select_idx_m_s['mean'])/temp_select_idx_m_s['std']



        temp_select_idx_m_s['missing_value'] = (~temp_select_idx_m_s['n_value'].isna()).astype(int)
        
        temp_select_idx_m_s = temp_select_idx_m_s.fillna(0)
                
        # time sequence lab & vital sitn  
        
        
        # text data
        diagnoses_icd = self.diagnoses_icd_merge_dropna[self.diagnoses_icd_merge_dropna['hadm_id']==hadm_id]
        
        icd_n = len(diagnoses_icd)
                
        hx_token_ids = []
        hx_token_ids.append(torch.tensor([101,0,0,0,0],dtype=torch.float32))
        
        for i in range(icd_n):
            h = diagnoses_icd.iloc[i]['long_title']
            hx_tokens = self.tokanizer.tokenize(h)
            
            # add BERT cls head
            hx_tokens = [self.tokanizer.cls_token, *hx_tokens]
            hx_tokens = hx_tokens[:512]
            hx_token_ids_ = self.tokanizer.convert_tokens_to_ids(hx_tokens)
            hx_token_ids.append(torch.tensor(hx_token_ids_,dtype=torch.float32))
        

            
        structure_tensor = torch.tensor(temp_select_idx_m_s['n_value'],dtype=torch.float32)
        structure_attention_mask_tensor = torch.tensor(temp_select_idx_m_s['missing_value'],dtype=torch.long)
        structure_position_ids_tensor = torch.tensor(temp_select_idx_m_s['s_idx'],dtype=torch.long)
        structure_time_ids_tensor = torch.tensor(temp_select_idx_m_s['time_day'],dtype=torch.long)
        
        trg = los
        trg_tensor = torch.tensor(los,dtype=torch.float32)

        
        datas = {'structure': structure_tensor,
                 'structure_attention_mask': structure_attention_mask_tensor,
                 'structure_position_ids': structure_position_ids_tensor,
                 'structure_time_ids': structure_time_ids_tensor,
                 'hx': hx_token_ids,
                 'trg':trg_tensor,
                 }
          
        return datas
    
    def __len__(self):
        return self.len
    
def collate_fn_time_sequence(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    structure = [DD['structure'] for DD in datas]
    structure_attention_mask = [DD['structure_attention_mask'] for DD in datas]
    structure_position_ids = [DD['structure_position_ids'] for DD in datas]
    structure_time_ids = [DD['structure_time_ids'] for DD in datas]
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
    
    # padding the structure
    structure = pad_sequence(structure,
                      batch_first = True,
                      padding_value=0)
    structure_attention_mask = pad_sequence(structure_attention_mask,
                      batch_first = True,
                      padding_value=0)
    structure_position_ids = pad_sequence(structure_position_ids,
                      batch_first = True,
                      padding_value=0)
    structure_time_ids = pad_sequence(structure_time_ids,
                      batch_first = True,
                      padding_value=0)
        
    batch['structure'] = structure
    batch['structure_attention_mask'] = structure_attention_mask
    batch['structure_position_ids'] = structure_position_ids
    batch['structure_time_ids'] = structure_time_ids
    
    batch['ehx'] = ehx
    batch['mask_ehx'] = mask_padding_ehx
    
    batch['stack_hx_n'] = torch.tensor(hx_n)
    batch['origin_ehx_length'] = torch.tensor(origin_ehx_length)
    
    batch['trg'] = torch.stack(trg)    
    return batch