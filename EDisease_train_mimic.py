import os
import time
import unicodedata
import random
import string
import re
import sys, pickle, math, tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc, accuracy_score

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
    
try:
    gpus = int(sys.argv[2])
    print('*****gpus = ', gpus)
except:
    gpus = 0
    
try:
    name = sys.argv[3]
    print('*****name = ', name)
except:
    name = None

batch_size = 128

parallel = False

alpha=1
beta=1
gamma=0.1

# fix the BERT version
model_name = "bert-base-multilingual-cased"
T_config = EDiseaseConfig()
S_config = StructrualConfig()

baseBERT = ED_model.adjBERTmodel(bert_ver=model_name,T_config=T_config,fixBERT=True)
BERT_tokenizer = AutoTokenizer.from_pretrained(model_name)

print(' ** pretrained BERT WEIGHT ** ')
print('baseBERT PARAMETERS: ' ,count_parameters(baseBERT))


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

# oversampling to balance +/-
pos = trainset_temp[trainset_temp['los'] > 7]['hadm_id'].values
neg = trainset_temp[trainset_temp['los'] <=7]['hadm_id'].values
ratio = len(neg) / len(pos)
balance_train_set_hadmid = round(ratio)*list(pos)+list(neg)

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
                                    dsidx=balance_train_set_hadmid)

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

DL_train = DataLoader(dataset = ds_train,
                     shuffle = True,
                     num_workers=4,
                     batch_size=batch_size,
                     collate_fn=dataloader.collate_fn)

DL_valid = DataLoader(dataset = ds_valid,
                     shuffle = False,
                     num_workers=4,
                     batch_size=batch_size,
                     collate_fn=dataloader.collate_fn)

DL_test = DataLoader(dataset = ds_test,
                     shuffle = False,
                     num_workers=4,
                     batch_size=batch_size,
                     collate_fn=dataloader.collate_fn)
    
def train_mimics(EDisease_Model,
                 stc2emb,
                 emb_emb,
                 dim_model,
                 baseBERT,
                 dloader,
                 dloader_v,
                 checkpoint_file,
                 device,
                 lr=1e-4,
                 epoch=100,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 noise=True,
                 gpus=0,
                 mlp=False,
                 name=None,
                 only_dx=False
                 ): 
    
    EDisease_Model.to(device)
    stc2emb.to(device)
    emb_emb.to(device)
    dim_model.to(device)
    baseBERT.to(device)
    
    baseBERT.eval()
        
    model_optimizer_eds = optim.Adam(EDisease_Model.parameters(), lr=lr)
    model_optimizer_s2e = optim.Adam(stc2emb.parameters(), lr=lr)
    model_optimizer_e2e = optim.Adam(emb_emb.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    
    criterion_em = nn.CrossEntropyLoss().to(device)
    
    if parallel:
        EDisease_Model = torch.nn.DataParallel(EDisease_Model)
        stc2emb = torch.nn.DataParallel(stc2emb)
        emb_emb = torch.nn.DataParallel(emb_emb)
        dim_model = torch.nn.DataParallel(dim_model)
        baseBERT = torch.nn.DataParallel(baseBERT)
    else:
        if device == 'cuda':
            torch.cuda.set_device(gpus)
            
    total_loss = []
    best_auc = 0
    auc_record = []
    
    s_type ='mlp' if mlp else 'spectrum'
    
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        
        for batch_idx, sample in enumerate(dloader):
            model_optimizer_eds.zero_grad()
            model_optimizer_dim.zero_grad()
            model_optimizer_s2e.zero_grad()
            model_optimizer_e2e.zero_grad()
            
            loss = torch.tensor(0.)

            sample = {k:v.to(device) for k,v in sample.items()}
            
            # for text data
            
            # c,cm = sample['cc'],sample['mask_cc']
            # output = baseBERT(c.long(),cm.long())
            # c_emb = output['heads']
            # c_emb_emb = emb_emb(c_emb)

            h,hm = sample['ehx'],sample['mask_ehx']
            output = baseBERT(h.long(),hm.long())
            h_emb = output['heads']
            em_h_emb = emb_emb(h_emb)
            
            stack_hx_n = sample['stack_hx_n']
            bs = len(stack_hx_n)
            
            hx_max = stack_hx_n.max()
            hx_padding = em_h_emb[:1]
            
            cumsum_hx_n = torch.cumsum(stack_hx_n,0)
            h_emb_cat_ = []
                    
            attention_mask_h = torch.ones([bs,hx_max],device=device)
            attention_mask_h[:,0] = 0
            
            for i,e in enumerate(cumsum_hx_n):
                hx_num = stack_hx_n[i]
                h_concat = em_h_emb[:cumsum_hx_n[i]] if i < 1 else em_h_emb[cumsum_hx_n[i-1]:cumsum_hx_n[i]]
                pad_num = hx_max - hx_num
                if pad_num>0:
                    hx_pads = [hx_padding]*pad_num
                    h_concat_pad = torch.cat([h_concat,*hx_pads],dim=0)
                    attention_mask_h[i,hx_num:] = 0
                else:
                    h_concat_pad = h_concat
                h_emb_cat_.append(h_concat_pad)
                
            h_emb_emb = torch.stack(h_emb_cat_)
            
            
            # for structual data
            s,sp, sm = sample['structure'],sample['structure_position_ids'], sample['structure_attention_mask']
                  
            if noise:
                normalization = torch.ones(s.shape).to(device)
                noise_ = normalization*noise_scale*torch.randn_like(s,device=device)
                s_noise = s+noise_
            else:
                s_noise = s
            
            s_emb = stc2emb(inputs=s_noise,
                                 attention_mask=sm,
                                 position_ids=sp)
            s_emb_org = stc2emb(inputs=s,
                                 attention_mask=sm,
                                 position_ids=sp)
            
            # make EDisease input data
            things = {}
            things_org ={}
            
            if not only_dx:
                things['s'] = {'emb':s_emb,
                               'attention_mask':torch.ones(s_emb.shape[:2],device=device).long(),
                               'position_id':5*torch.ones(s_emb.shape[:2],device=device).long()
                               }
                things_org['s'] = {'emb':s_emb_org,
                                   'attention_mask':torch.ones(s_emb.shape[:2],device=device).long(),
                                   'position_id':5*torch.ones(s_emb.shape[:2],device=device).long()
                                   }
            things['h'] = {'emb':h_emb_emb,
                           'attention_mask':attention_mask_h.long(),
                           'position_id':7*torch.ones(h_emb_emb.shape[:2],device=device).long()
                           }
            # thigns['c'] = {'emb':c_emb_emb.unsqueeze(1),
            #                'attention_mask':torch.ones(c_emb_emb.unsqueeze(1).shape[:2],device=device),
            #                'position_id':6*torch.ones(c_emb_emb.unsqueeze(1).shape[:2],device=device)
            #                }
            
            things_org['h'] = {'emb':h_emb_emb,
                               'attention_mask':attention_mask_h.long(),
                               'position_id':7*torch.ones(h_emb_emb.shape[:2],device=device).long()
                               }


            outp = EDisease_Model(things,
                                  mask_ratio=mask_ratio
                                  )
            EDisease = outp['EDisease']
            predict = outp['predict']

            outp2 = EDisease_Model(things=things_org,
                                   mask_ratio=mask_ratio
                                   )
            EDisease2 = outp2['EDisease']
            
            EDiseaseFake = torch.cat([EDisease[1:],EDisease[:1]],dim=0)
            
            mode = 'D' if batch_idx%2==0 else 'G'
            ptloss = True if batch_idx%299==3 else False
            
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
            
            trg = sample['trg']
            
            trg_bool = (trg >= 7).long()
            
            loss_cls = criterion_em(predict,trg_bool)
            
            loss = loss_dim+loss_cls
                
            loss.sum().backward()
            model_optimizer_eds.step()
            model_optimizer_dim.step()
            model_optimizer_s2e.step()
            model_optimizer_e2e.step()
                     
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs

            if ptloss:
                print('  ========================================================== ')
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L DIM:{:.4f} L CLS:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss_dim.item(), loss_cls.item()
                        )) 
                print(predict[:4],predict.shape)
                print(trg_bool[:4],trg_bool.shape)
                print('  ========================================================== \n')

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
            
            
            vEDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                                     S_config=S_config
                                                     )
            
            vstc2emb = ED_model.structure_emb(S_config) if not mlp else ED_model.structure_emb_mlp(S_config)
            vemb_emb = ED_model.emb_emb(T_config)
        
            vdim_model = ED_model.DIM(T_config=T_config,
                                      alpha=alpha,
                                      beta=beta,
                                      gamma=gamma)
            
            try: 
                vEDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model.pth',vEDisease_Model)
                print(' ** Complete Load CLS EDisease Model ** ')
            except:
                print('*** No Pretrain_EDisease_CLS_Model ***')
        
            try:     
                vdim_model = load_checkpoint(checkpoint_file,'dim_model.pth',vdim_model)
            except:
                print('*** No Pretrain_dim_model ***')
        
            try:     
                vstc2emb = load_checkpoint(checkpoint_file,'stc2emb.pth',vstc2emb)
            except:
                print('*** No Pretrain_stc2emb ***')
        
            try:     
                vemb_emb = load_checkpoint(checkpoint_file,'emb_emb.pth',vemb_emb)
            except:
                print('*** No Pretrain_emb_emb ***')

            try:
                valres= testt_mimics(vEDisease_Model,
                                     vstc2emb,
                                     vemb_emb,
                                     vdim_model,
                                     baseBERT,
                                     dloader_v,
                                     parallel=False,
                                     gpus=gpus,
                                     device=device,
                                     only_dx=only_dx)               

                fpr, tpr, _ = roc_curve(valres['ground_truth'].values, valres['probability'].values)
                
                roc_auc = auc(fpr,tpr)
                auc_record.append(roc_auc)

                print(f'auc: {roc_auc:.3f} ; === best is {best_auc:.3f} ')
                

                if roc_auc > best_auc:
                    best_auc = roc_auc
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease_Model_best.pth',
                                    model=EDisease_Model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='dim_model_best.pth',
                                    model=dim_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='stc2emb_best.pth',
                                    model=stc2emb,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='emb_emb_best.pth',
                                    model=emb_emb,
                                    parallel=parallel)
            except Exception as e:
                print(e)
            
            pd_total_auc = pd.DataFrame(auc_record)
            pd_total_auc.to_csv(f'./loss_record/total_auc_{s_type}_{name}.csv', sep = ',')
        
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv(f'./loss_record/total_loss_{s_type}_{name}.csv', sep = ',')
    print(total_loss) 


def testt_mimics(EDisease_Model,
                 stc2emb,
                 emb_emb,
                 dim_model,
                 baseBERT,
                 dloader,
                 device,
                 parallel=parallel,
                 gpus=0,
                 only_dx=False
                 ): 
    
    EDisease_Model.to(device)
    stc2emb.to(device)
    emb_emb.to(device)
    dim_model.to(device)
    baseBERT.to(device)
    
    baseBERT.eval()
    EDisease_Model.eval()
    stc2emb.eval()
    emb_emb.eval()
    dim_model.eval()
        
    if device == 'cuda':
        torch.cuda.set_device(gpus)
    
    total_res_ = []
    
    with torch.no_grad():  
        for batch_idx, sample in tqdm.tqdm(enumerate(dloader)):

            sample = {k:v.to(device) for k,v in sample.items()}
            
            # for text data
            
            # c,cm = sample['cc'],sample['mask_cc']
            # output = baseBERT(c.long(),cm.long())
            # c_emb = output['heads']
            # c_emb_emb = emb_emb(c_emb)

            h,hm = sample['ehx'],sample['mask_ehx']
            output = baseBERT(h.long(),hm.long())
            h_emb = output['heads']
            em_h_emb = emb_emb(h_emb)
            
            stack_hx_n = sample['stack_hx_n']
            bs = len(stack_hx_n)
            
            hx_max = stack_hx_n.max()
            hx_padding = em_h_emb[:1]
            
            cumsum_hx_n = torch.cumsum(stack_hx_n,0)
            h_emb_cat_ = []
                    
            attention_mask_h = torch.ones([bs,hx_max],device=device)
            attention_mask_h[:,0] = 0
            
            for i,e in enumerate(cumsum_hx_n):
                hx_num = stack_hx_n[i]
                h_concat = em_h_emb[:cumsum_hx_n[i]] if i < 1 else em_h_emb[cumsum_hx_n[i-1]:cumsum_hx_n[i]]
                pad_num = hx_max - hx_num
                if pad_num>0:
                    hx_pads = [hx_padding]*pad_num
                    h_concat_pad = torch.cat([h_concat,*hx_pads],dim=0)
                    attention_mask_h[i,hx_num:] = 0
                else:
                    h_concat_pad = h_concat
                h_emb_cat_.append(h_concat_pad)
                
            h_emb_emb = torch.stack(h_emb_cat_)
            
            
            # for structual data
            s,sp, sm = sample['structure'],sample['structure_position_ids'], sample['structure_attention_mask']
               
            s_emb = stc2emb(inputs=s,
                                 attention_mask=sm,
                                 position_ids=sp)
            
            # make EDisease input data
            things={}
            if not only_dx:
                things['s'] = {'emb':s_emb,
                               'attention_mask':torch.ones(s_emb.shape[:2],device=device).long(),
                               'position_id':5*torch.ones(s_emb.shape[:2],device=device).long()
                               }

            things['h'] = {'emb':h_emb_emb,
                           'attention_mask':attention_mask_h.long(),
                           'position_id':7*torch.ones(h_emb_emb.shape[:2],device=device).long()
                           }


            outp = EDisease_Model(things,
                                  mask_ratio=0.
                                  )
            predict = outp['predict']

            trg = sample['trg']
            trg_bool = (trg >= 7).long()
            
            predict_label = nn.functional.softmax(predict,dim=1)[:,1]

            columns=['probability', 'ground_truth']
            
            result = pd.concat([pd.Series(predict_label.cpu().detach().numpy()),
                                pd.Series(trg_bool.cpu().detach().numpy())],axis=1)
            result.columns = columns
            
            total_res_.append(result)
        total_res = pd.concat(total_res_,axis=0,ignore_index=True)
    return total_res
        
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
'  =======================================================================================================  '   
            
if task=='train':
    device = f'cuda:{gpus}'
    
    mlp = False
    checkpoint_file = '../checkpoint_EDs/EDisease_spectrum_flat'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')
    
    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)

    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

# ====
    train_mimics(EDisease_Model=EDisease_Model,
                 stc2emb=stc2emb,
                 emb_emb=emb_emb,
                 dim_model=dim_model,
                 baseBERT=baseBERT,
                 dloader=DL_train,
                 dloader_v=DL_valid, 
                 lr=1e-5,
                 epoch=200,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 noise=True,
                 gpus=gpus,
                 device=device,
                 mlp=mlp) 

if task=='test':
    device = f'cuda:{gpus}'
    
    mlp = False
    checkpoint_file = '../checkpoint_EDs/EDisease_spectrum_flat'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')
    
    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)

    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

# ====
    valres= testt_mimics(EDisease_Model,
                         stc2emb,
                         emb_emb,
                         dim_model,
                         baseBERT,
                         DL_test,
                         parallel=False,
                         gpus=gpus,
                         device=device)               

    fpr, tpr, _ = roc_curve(valres['ground_truth'].values, valres['probability'].values)
    
    roc_auc = auc(fpr,tpr)
    
    valres.to_pickle(f'./result_pickles/EDspectrumFlat_{roc_auc*1000:.0f}.pkl')

    print(f'auc: {roc_auc:.3f}')
    
if task=='train_mlp':

    only_dx = True if name=='only_dx' else False
    print(f' =========== name = {name}; only_dx = {only_dx}============')
    
    device = f'cuda:{gpus}'
    
    mlp = True
    checkpoint_file = f'../checkpoint_EDs/EDisease_spectrum_flat_oldstr2emb_{name}'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')

    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb_mlp(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)
    
    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

# ====
    train_mimics(EDisease_Model=EDisease_Model,
                 stc2emb=stc2emb,
                 emb_emb=emb_emb,
                 dim_model=dim_model,
                 baseBERT=baseBERT,
                 dloader=DL_train,
                 dloader_v=DL_valid, 
                 lr=1e-5,
                 epoch=200,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 noise=True,
                 gpus=gpus,
                 device=device,
                 mlp=mlp,
                 only_dx=only_dx) 

if task=='test_mlp':
    only_dx = True if name=='only_dx' else False
    print(f' =========== name = {name}; only_dx = {only_dx}============')
    
    device = f'cuda:{gpus}'
    
    mlp = True
    checkpoint_file = f'../checkpoint_EDs/EDisease_spectrum_flat_oldstr2emb_{name}'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')

    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb_mlp(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

# ====
    valres= testt_mimics(EDisease_Model,
                         stc2emb,
                         emb_emb,
                         dim_model,
                         baseBERT,
                         DL_test,
                         parallel=False,
                         gpus=gpus,
                         device=device,
                         only_dx=only_dx)               

    fpr, tpr, _ = roc_curve(valres['ground_truth'].values, valres['probability'].values)
    
    roc_auc = auc(fpr,tpr)
    
    valres.to_pickle(f'./result_pickles/EDmlpFlat_{name}_{roc_auc*1000:.0f}.pkl')

    print(f'auc: {roc_auc:.3f}')
        
if task=='train_mlp_ip':
    print(f' =========== imputation name = {name} ============')
    filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
    vital_signs_ip = pd.read_pickle(filepath)
    
    filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
    hadmid_first_lab_ip = pd.read_pickle(filepath)
    
    ds_train_ip = dataloader.mimic_Dataset(set_hadmid=train_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        vital_signs=vital_signs_ip,
                                        hadmid_first_lab=hadmid_first_lab_ip,
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
                                        dsidx=balance_train_set_hadmid)
    
    ds_valid_ip = dataloader.mimic_Dataset(set_hadmid=val_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        vital_signs=vital_signs_ip,
                                        hadmid_first_lab=hadmid_first_lab_ip,
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
    
    ds_test_ip  = dataloader.mimic_Dataset(set_hadmid=test_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        vital_signs=vital_signs_ip,
                                        hadmid_first_lab=hadmid_first_lab_ip,
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

    DL_train_ip = DataLoader(dataset = ds_train_ip,
                         shuffle = True,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
    
    DL_valid_ip = DataLoader(dataset = ds_valid_ip,
                         shuffle = False,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
    
    DL_test_ip = DataLoader(dataset = ds_test_ip,
                         shuffle = False,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
    
    device = f'cuda:{gpus}'
    mlp = True
    checkpoint_file = f'../checkpoint_EDs/EDisease_spectrum_flat_oldstr2emb_{name}'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')

    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb_mlp(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)

    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))    

    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

# ====
    train_mimics(EDisease_Model=EDisease_Model,
                 stc2emb=stc2emb,
                 emb_emb=emb_emb,
                 dim_model=dim_model,
                 baseBERT=baseBERT,
                 dloader=DL_train_ip,
                 dloader_v=DL_valid_ip, 
                 lr=1e-5,
                 epoch=200,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 noise=True,
                 gpus=gpus,
                 device=device,
                 mlp=mlp,
                 name=name) 

if task=='test_mlp_ip':
    print(f' =========== imputation name = {name} ============')
    filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
    vital_signs_ip = pd.read_pickle(filepath)
    
    filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
    hadmid_first_lab_ip = pd.read_pickle(filepath)
    
    ds_train_ip = dataloader.mimic_Dataset(set_hadmid=train_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        vital_signs=vital_signs_ip,
                                        hadmid_first_lab=hadmid_first_lab_ip,
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
                                        dsidx=balance_train_set_hadmid)
    
    ds_valid_ip = dataloader.mimic_Dataset(set_hadmid=val_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        vital_signs=vital_signs_ip,
                                        hadmid_first_lab=hadmid_first_lab_ip,
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
    
    ds_test_ip  = dataloader.mimic_Dataset(set_hadmid=test_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        vital_signs=vital_signs_ip,
                                        hadmid_first_lab=hadmid_first_lab_ip,
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

    DL_train_ip = DataLoader(dataset = ds_train_ip,
                         shuffle = True,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
    
    DL_valid_ip = DataLoader(dataset = ds_valid_ip,
                         shuffle = False,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
    
    DL_test_ip = DataLoader(dataset = ds_test_ip,
                         shuffle = False,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn)
    
    device = f'cuda:{gpus}'
    
    mlp = True
    checkpoint_file = f'../checkpoint_EDs/EDisease_spectrum_flat_oldstr2emb_{name}'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')

    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb_mlp(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)
    
    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

# ====
    valres= testt_mimics(EDisease_Model,
                         stc2emb,
                         emb_emb,
                         dim_model,
                         baseBERT,
                         DL_test_ip,
                         parallel=False,
                         gpus=gpus,
                         device=device)               

    fpr, tpr, _ = roc_curve(valres['ground_truth'].values, valres['probability'].values)
    
    roc_auc = auc(fpr,tpr)
    
    valres.to_pickle(f'./result_pickles/EDmlpFlat_{name}_{roc_auc*1000:.0f}.pkl')

    print(f'auc: {roc_auc:.3f}')
    
if task=='trainTS':

    # timesequence vitalsign
    filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_vitalsign_TS.pdpkl')
    stayid_vitalsign_TS = pd.read_pickle(filepath)
    
    # timesequence lab
    filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean_combine.pdpkl')
    labevents_merge_dropna_clean_combine = pd.read_pickle(filepath)

    # combime the idx with mean std
    df_train_set_vitalsign_mean = train_set_vitalsign_mean.to_frame()
    df_train_set_vitalsign_mean.columns = ['mean']
    df_train_set_agegender_mean = train_set_agegender_mean.to_frame()
    df_train_set_agegender_mean.columns = ['mean']
    df_train_set_lab_mean = train_set_lab_mean.to_frame()
    df_train_set_lab_mean.columns = ['mean']
    df_io_24_mean = io_24_mean.to_frame()
    df_io_24_mean.columns = ['mean']
    
    df_train_set_mean = pd.concat([df_train_set_agegender_mean,
                                   df_train_set_vitalsign_mean,
                                   df_train_set_lab_mean,
                                   df_io_24_mean],axis=0)
    
    df_train_set_vitalsign_std = train_set_vitalsign_std.to_frame()
    df_train_set_vitalsign_std.columns = ['std']
    df_train_set_agegender_std = train_set_agegender_std.to_frame()
    df_train_set_agegender_std.columns = ['std']
    df_train_set_lab_std = train_set_lab_std.to_frame()
    df_train_set_lab_std.columns = ['std']
    df_io_24_std = io_24_std.to_frame()
    df_io_24_std.columns = ['std']
    
    df_train_set_std = pd.concat([df_train_set_agegender_std,
                                   df_train_set_vitalsign_std,
                                   df_train_set_lab_std,
                                   df_io_24_std],axis=0)
    
    structurals_idx_mean_std = pd.concat([structurals_idx,df_train_set_mean,df_train_set_std],axis=1)
    # combime the idx with mean std

    device = f'cuda:{gpus}'
    
    mlp = False
    checkpoint_file = '../checkpoint_EDs/EDisease_spectrum_TS'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')
    
    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)

    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

    ds_train = dataloader.mimic_time_sequence_Dataset(set_hadmid=train_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        timesequence_vital_signs=stayid_vitalsign_TS,
                                        timesequence_lab=labevents_merge_dropna_clean_combine,
                                        diagnoses_icd_merge_dropna=diagnoses_icd_merge_dropna,
                                        tokanizer=BERT_tokenizer,
                                        structurals_idx=structurals_idx_mean_std,
                                        dsidx=balance_train_set_hadmid,
                                        test=False)
    
    ds_valid = dataloader.mimic_time_sequence_Dataset(set_hadmid=val_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        timesequence_vital_signs=stayid_vitalsign_TS,
                                        timesequence_lab=labevents_merge_dropna_clean_combine,
                                        diagnoses_icd_merge_dropna=diagnoses_icd_merge_dropna,
                                        tokanizer=BERT_tokenizer,
                                        structurals_idx=structurals_idx_mean_std,
                                        dsidx=None,
                                        test=True)
    
    DL_train = DataLoader(dataset = ds_train,
                         shuffle = True,
                         num_workers=8,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn_time_sequence)
    
    DL_valid = DataLoader(dataset = ds_valid,
                         shuffle = False,
                         num_workers=2,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn_time_sequence)


# ====
    train_mimics(EDisease_Model=EDisease_Model,
                 stc2emb=stc2emb,
                 dloader=DL_train,
                 dloader_v=DL_valid, 
                 lr=1e-5,
                 epoch=200,
                 log_interval=10,
                 noise_scale=0.002,
                 mask_ratio=0.33,
                 parallel=parallel,                     
                 checkpoint_file=checkpoint_file,
                 noise=True,
                 gpus=gpus,
                 device=device,
                 mlp=mlp) 

if task=='testTS':

    # timesequence vitalsign
    filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_vitalsign_TS.pdpkl')
    stayid_vitalsign_TS = pd.read_pickle(filepath)
    
    # timesequence lab
    filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean_combine.pdpkl')
    labevents_merge_dropna_clean_combine = pd.read_pickle(filepath)

    # combime the idx with mean std
    df_train_set_vitalsign_mean = train_set_vitalsign_mean.to_frame()
    df_train_set_vitalsign_mean.columns = ['mean']
    df_train_set_agegender_mean = train_set_agegender_mean.to_frame()
    df_train_set_agegender_mean.columns = ['mean']
    df_train_set_lab_mean = train_set_lab_mean.to_frame()
    df_train_set_lab_mean.columns = ['mean']
    df_io_24_mean = io_24_mean.to_frame()
    df_io_24_mean.columns = ['mean']
    
    df_train_set_mean = pd.concat([df_train_set_agegender_mean,
                                   df_train_set_vitalsign_mean,
                                   df_train_set_lab_mean,
                                   df_io_24_mean],axis=0)
    
    df_train_set_vitalsign_std = train_set_vitalsign_std.to_frame()
    df_train_set_vitalsign_std.columns = ['std']
    df_train_set_agegender_std = train_set_agegender_std.to_frame()
    df_train_set_agegender_std.columns = ['std']
    df_train_set_lab_std = train_set_lab_std.to_frame()
    df_train_set_lab_std.columns = ['std']
    df_io_24_std = io_24_std.to_frame()
    df_io_24_std.columns = ['std']
    
    df_train_set_std = pd.concat([df_train_set_agegender_std,
                                   df_train_set_vitalsign_std,
                                   df_train_set_lab_std,
                                   df_io_24_std],axis=0)
    
    structurals_idx_mean_std = pd.concat([structurals_idx,df_train_set_mean,df_train_set_std],axis=1)
    # combime the idx with mean std


    device = f'cuda:{gpus}'
    
    mlp = False
    checkpoint_file = '../checkpoint_EDs/EDisease_spectrum_TS'
    if not os.path.isdir(checkpoint_file):
        os.makedirs(checkpoint_file)
        print(f' make dir {checkpoint_file}')
    
    EDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                             S_config=S_config
                                             )

    stc2emb = ED_model.structure_emb(S_config)
    emb_emb = ED_model.emb_emb(T_config)

    dim_model = ED_model.DIM(T_config=T_config,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)

    print('dim_model PARAMETERS: ' ,count_parameters(dim_model))
    print('emb_emb PARAMETERS: ' ,count_parameters(emb_emb))
    print('stc2emb PARAMETERS: ' ,count_parameters(stc2emb))
    print('EDisease_Model PARAMETERS: ' ,count_parameters(EDisease_Model))
    
    try: 
        EDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model_best.pth',EDisease_Model)
        print(' ** Complete Load CLS EDisease Model ** ')
    except:
        print('*** No Pretrain_EDisease_CLS_Model ***')

    try:     
        dim_model = load_checkpoint(checkpoint_file,'dim_model_best.pth',dim_model)
    except:
        print('*** No Pretrain_dim_model ***')

    try:     
        stc2emb = load_checkpoint(checkpoint_file,'stc2emb_best.pth',stc2emb)
    except:
        print('*** No Pretrain_stc2emb ***')

    try:     
        emb_emb = load_checkpoint(checkpoint_file,'emb_emb_best.pth',emb_emb)
    except:
        print('*** No Pretrain_emb_emb ***')

    ds_test  = dataloader.mimic_time_sequence_Dataset(set_hadmid=test_set_hadmid,
                                        icustays_select=icustays_select_sort_dropduplicate,
                                        agegender=agegender,
                                        timesequence_vital_signs=stayid_vitalsign_TS,
                                        timesequence_lab=labevents_merge_dropna_clean_combine,
                                        diagnoses_icd_merge_dropna=diagnoses_icd_merge_dropna,
                                        tokanizer=BERT_tokenizer,
                                        structurals_idx=structurals_idx_mean_std,
                                        dsidx=None,
                                        test=True)
    DL_test = DataLoader(dataset = ds_test,
                         shuffle = False,
                         num_workers=4,
                         batch_size=batch_size,
                         collate_fn=dataloader.collate_fn_time_sequence)

# ====
    valres= testt_mimics(EDisease_Model,
                         stc2emb,
                         DL_test,
                         parallel=False,
                         gpus=gpus,
                         device=device)               

    fpr, tpr, _ = roc_curve(valres['ground_truth'].values, valres['probability'].values)
    
    roc_auc = auc(fpr,tpr)
    
    valres.to_pickle(f'./result_pickles/EDspectrumTS_OnlyS_{roc_auc*1000:.0f}.pkl')
