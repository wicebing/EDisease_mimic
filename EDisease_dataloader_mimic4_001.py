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
        self.dsidx = dsidx
        
        if dsidx is None:
            self.len = len(set_hadmid)
        else:
            self.len = len(dsidx)
    
    def __getitem__(self, index):
        if self.dsidx is None:
            e_patient = self.ds.iloc[index]
        else: 
            e_patient = self.ds.iloc[self.dsidx.iloc[index]]        
       
        chief_complaint = e_patient['CHIEFCOMPLAIN']
        cc_tokens = self.tokanizer.tokenize(str(chief_complaint))
        # add BERT cls head
        cc_tokens = [self.tokanizer.cls_token, *cc_tokens]
        cc_tokens = cc_tokens[:512]
        cc_token_ids = self.tokanizer.convert_tokens_to_ids(cc_tokens)

        H_select = ['EDDIAL', 'DIABETES', 'DEMENTIA', 'MIHX', 'DVT',
            'CANCER', 'ETOHAB', 'ALZHD', 'ASTHMA', 'CEBVD', 'CKD', 'COPD',
            'CHF', 'CAD', 'DEPRN', 'DIABTYP1', 'DIABTYP2', 'DIABTYP0', 'ESRD',
            'HPE', 'EDHIV', 'HYPLIPID', 'HTN', 'OBESITY', 'OSA', 'OSTPRSIS','SUBSTAB',]

        H_content = ['Dialysis, HD, PD, H/D, P/D, ',
                     'Diabetes DM, ',
                     'Dementia, ',
                     'myocardial infarction (MI) (CAD), ',
                     'deep vein thrombosis (DVT), ',
                     'Cancer, ', 
                     'Alcoholism, Alcohol misuse, abuse, dependence, ', 
                     'Alzheimer’s disease, dementia, ',
                     'Asthma, ', 
                     'Cerebrovascular disease, stroke (CVA), transient ischemic attack (TIA), ', 
                     'Chronic kidney disease (CKD), ', 
                     'Chronic obstructive pulmonary disease (COPD), ',
                     'Congestive heart failure, CHF, ', 
                     'Coronary artery disease (CAD), ischemic heart disease (IHD), myocardial infarction (MI), ', 
                     'Depression, ', 
                     'Diabetes mellitus (DM) Type I',
                     'Diabetes mellitus (DM) Type 2', 
                     'Diabetes mellitus (DM)', 
                     'End-stage renal disease (ESRD), ',
                     'Pulmonary embolism (PE), deep vein thrombosis (DVT), venous thromboembolism (VTE), ', 
                     'HIV infection, AIDS',
                     'Hyperlipidemia, ', 
                     'Hypertension, ', 
                     'Obesity, ',
                     'Obstructive sleep apnea (OSA), ', 
                     'Osteoporosis, ',
                     'Substance abuse or dependence, ',]
        
        hxyn = e_patient[H_select].values
        
        history = ''
        
        for j, y in enumerate(hxyn):
            if y =='Yes' or y =='YES' or y =='yes':
                history += H_content[j]
                
        hx_token_ids = []
        hx_token_ids.append(torch.tensor([101,0,0,0,0],dtype=torch.float32)) 
               
        
        hx_tokens = self.tokanizer.tokenize(history)
        # add BERT cls head
        while len(hx_tokens)>511:
            hx_tokens_i,hx_tokens = hx_tokens[:511],hx_tokens[511:]
            hx_tokens_i = [self.tokanizer.cls_token, *hx_tokens_i]
            hx_token_ids_ = self.tokanizer.convert_tokens_to_ids(hx_tokens_i)
            hx_token_ids.append(torch.tensor(hx_token_ids_,dtype=torch.float32))
        hx_tokens = [self.tokanizer.cls_token, *hx_tokens]
        hx_tokens = hx_tokens[:512]
        hx_token_ids_ = self.tokanizer.convert_tokens_to_ids(hx_tokens)
        hx_token_ids.append(torch.tensor(hx_token_ids_,dtype=torch.float32))
        

        # ee_patient = e_patient.reshape(1,-1) 
        #          0   1      2      3   4  5  6   7 8  9 10  11     12      13      14
    # structure = y/o, sex, revisit,SBP,DBP,P,SPO2,R,T, H, W,Pain,Triage_C, Triage_H
    # structure = y/o, sex, revisit,SBP,DBP,P,SPO2,R,T, H, W,Pain,   BE,     BV      BM
        S_select = ['AGE',
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
        structure_df = e_patient[S_select]
        
        structure = np.array(structure_df,dtype='float')
        
        structure_attention_mask = np.ones(structure.shape)
        structure_position_ids = np.arange(0, len(S_select))
        
        for i,d in enumerate(structure):
            if str(d) == 'nan':
                structure[i] = 999
                structure_attention_mask[i]=0
                    
        nan = np.isnan(structure)
                    
        # # normalization
        structure_np = np.float32(structure)
        structure_mean = self.normalization[0]
#        structure_mean[0] = 43.34
        structure_mean[1] = 0.5
        structure_mean[2] = 0
        structure_std  = self.normalization[1]
#        structure_std[0] = 26.97
        structure_std[1] = 0.5
        structure_std[2] = 1
        structure_normalization = (structure_np-structure_mean)/(structure_std+1e-6)
        
        structure_normalization[nan] = 0
        nan = structure_np == 999       
        structure_normalization[nan] = 0
        if structure_np[3] == 999:
            structure_normalization[3] = -4
        if structure_np[4] == 999:
            structure_normalization[4] = -4
            
        structure_tensor = torch.tensor(np.float32(structure_normalization),dtype=torch.float32)
        structure_attention_mask_tensor = torch.tensor(structure_attention_mask,dtype=torch.long)
        structure_position_ids_tensor = torch.tensor(structure_position_ids,dtype=torch.long)
        
        chief_complaint_tensor = torch.tensor(cc_token_ids,dtype=torch.float32)

        target_select = ['DOA',
                         'DIEDED',
                         'DOA',
                         'ICU',
                         'DIEDED',
                         'AGE',
                         'SEX',
                         'DOA',
                         'DOA',
                         'query']
        
        trg = e_patient[target_select]
               
        trg_tensor = torch.tensor(trg,dtype=torch.float32)

        
        for i,d in enumerate(structure):
            if str(d) == 'nan':
                structure[i] = -1
        
        portal = {'CC':str(chief_complaint),
                  'HX':history,
                  'SS':structure}
        
        datas = {'structure': structure_tensor,
                 'structure_attention_mask': structure_attention_mask_tensor,
                 'structure_position_ids': structure_position_ids_tensor,
                 'cc': chief_complaint_tensor,
                 'hx': hx_token_ids,
                 'trg':trg_tensor,
                 'portal':portal,
                 'idx':e_patient['idx']
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
    cc = [DD['cc'] for DD in datas]
    stack_hx_ = [DD['hx'] for DD in datas]
    trg = [DD['trg'] for DD in datas]
    idx = [DD['idx'] for DD in datas]
    
    stack_hx = []
    hx_n = []
    for shx in stack_hx_:
        hx_n.append(len(shx))
        for eshx in shx:
            stack_hx.append(eshx)
       
    ehx = stack_hx
    origin_cc_length = [len(d) for d in cc]
    origin_ehx_length = [len(d) for d in ehx]
    # origin_pi_length = [len(d) for d in pi]
    
    cc = pad_sequence(cc,
                      batch_first = True,
                      padding_value=0)
    ehx = pad_sequence(ehx,
                      batch_first = True,
                      padding_value=0)

    
    mask_padding_cc = torch.zeros(cc.shape,dtype=torch.long)
    mask_padding_ehx = torch.zeros(ehx.shape,dtype=torch.long)
    
    for i,e in enumerate(origin_cc_length):
        mask_padding_cc[i,:e] = 1
    for i,e in enumerate(origin_ehx_length):
        mask_padding_ehx[i,:e] = 1
        
    batch['structure'] = torch.stack(structure)
    batch['structure_attention_mask'] = torch.stack(structure_attention_mask)
    batch['structure_position_ids'] = torch.stack(structure_position_ids)
    batch['cc'] = cc
    batch['ehx'] = ehx
    
    batch['mask_cc'] = mask_padding_cc
    batch['mask_ehx'] = mask_padding_ehx
    
    batch['stack_hx_n'] = torch.tensor(hx_n)
    batch['origin_cc_length'] = torch.tensor(origin_cc_length)
    batch['origin_ehx_length'] = torch.tensor(origin_ehx_length)
    
    batch['trg'] = torch.stack(trg)
    
    batch['idx'] = torch.tensor(idx)
    
    return batch

