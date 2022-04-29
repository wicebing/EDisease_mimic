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

import EDisease_dataloader_mimic4_002 as dataloader

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


try:
    task = sys.argv[1]
except:
    task = 'test_nhamcs_cls'
print('*****task= ',task)
 
try:
    gpus = int(sys.argv[2])
except:
    gpus = 0
print('*****gpus = ', gpus)
    
try:
    random_state = int(sys.argv[3])
except:
    random_state = 0
print('*****random_state = ', random_state)

try:
    skemAdjust = sys.argv[4]
except:
    skemAdjust = 'origin'
print('*****skemAdjust = ', skemAdjust)

try:
    ablation = sys.argv[5]
except:
    ablation = None
print('*****ablation = ', ablation)

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

# timesequence vitalsign
# filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_vitalsign_TS.pdpkl')
# stayid_vitalsign_TS = pd.read_pickle(filepath)
filepath = os.path.join(db_file_path, 'data_EDis', 'vitalsigns_per_stay_id.pdpkl')
with open(filepath, 'rb') as f:
    stayid_vitalsign_TS = pickle.load(f)

# timesequence lab
# filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean_combine.pdpkl')
# labevents_merge_dropna_clean_combine = pd.read_pickle(filepath)
filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_per_hadm_id.pdpkl')
with open(filepath, 'rb') as f:
    labevents_merge_dropna_clean_combine = pickle.load(f)

# time point data
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
train_set_hadmid = hadmid_first_lab.sample(frac=0.80,random_state=random_state).index
temp_set_hadmid = hadmid_first_lab.drop(train_set_hadmid)
val_set_hadmid = temp_set_hadmid.sample(frac=0.50,random_state=random_state).index
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
structurals_idx.columns = ['bb_idx']
structurals_idx['s_idx'] = 10+np.arange(len(structurals))

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


# oversampling to balance +/-
pos = trainset_temp[trainset_temp['los'] > 7]['hadm_id'].values
neg = trainset_temp[trainset_temp['los'] <=7]['hadm_id'].values
ratio = len(neg) / len(pos)
balance_train_set_hadmid = round(ratio)*list(pos)+list(neg)

    
def train_mimics(EDisease_Model,
                 stc2emb,
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
                 ablation=None,
                 ): 
    
    EDisease_Model.to(device)
    stc2emb.to(device)
           
    model_optimizer_eds = optim.Adam(EDisease_Model.parameters(), lr=lr)
    model_optimizer_s2e = optim.Adam(stc2emb.parameters(), lr=lr)

    
    criterion_em = nn.CrossEntropyLoss().to(device)
    
    if parallel:
        EDisease_Model = torch.nn.DataParallel(EDisease_Model)
        stc2emb = torch.nn.DataParallel(stc2emb)
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
            model_optimizer_s2e.zero_grad()
            
            loss = torch.tensor(0.)

            sample = {k:v.to(device) for k,v in sample.items()}
            

            # for structual data
            s,sp, sm = sample['structure'],sample['structure_position_ids'], sample['structure_attention_mask']
            st = sample['structure_time_ids']
            
            bs = len(s)
                  
            if noise:
                normalization = torch.ones(s.shape).to(device)
                noise_ = normalization*noise_scale*torch.randn_like(s,device=device)
                s_noise = s+noise_
            else:
                s_noise = s
            
            if ablation is None:
                s_emb = stc2emb(inputs=s_noise,
                                     attention_mask=sm,
                                     position_ids=sp,
                                     time_ids=st)
            elif ablation == 'mask':
                s_emb = stc2emb(inputs=s_noise,
                                attention_mask=None,
                                     position_ids=sp,
                                     time_ids=st)
            elif ablation == 'vtype':
                s_emb = stc2emb(inputs=s_noise,
                                     attention_mask=sm,
                                     position_ids=sp*0,
                                     time_ids=st)
            elif ablation == 'vtime':
                s_emb = stc2emb(inputs=s_noise,
                                     attention_mask=sm,
                                     position_ids=sp,
                                     time_ids=st*0.)
            


            predict = EDisease_Model.classifier(s_emb[:,0,:])


            
            trg = sample['trg']
            
            trg_bool = (trg >= 7).long()
            
            loss_cls = criterion_em(predict,trg_bool)
            
            loss = loss_cls
                
            loss.sum().backward()
            model_optimizer_eds.step()
            model_optimizer_s2e.step()
                     
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
                
            ptloss = True if batch_idx%299==3 else False
            if ptloss:
                print('  ========================================================== ')
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L CLS:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss_cls.item()
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
                            checkpoint_path='stc2emb.pth',
                            model=stc2emb,
                            parallel=parallel)
            print('======= epoch:%i ========'%ep)
            
            
            vEDisease_Model = ED_model.EDisease_Model(T_config=T_config,
                                                     S_config=S_config
                                                     )
            
            vstc2emb = ED_model.structure_emb(S_config) if not mlp else ED_model.structure_emb_mlp(S_config)
        
            
            try: 
                vEDisease_Model = load_checkpoint(checkpoint_file,'EDisease_Model.pth',vEDisease_Model)
                print(' ** Complete Load CLS EDisease Model ** ')
            except:
                print('*** No Pretrain_EDisease_CLS_Model ***')
        
            try:     
                vstc2emb = load_checkpoint(checkpoint_file,'stc2emb.pth',vstc2emb)
            except:
                print('*** No Pretrain_stc2emb ***')
        
            try:
                valres= testt_mimics(vEDisease_Model,
                                     vstc2emb,
                                     dloader_v,
                                     parallel=False,
                                     gpus=gpus,
                                     device=device,
                                     ablation=ablation)               

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
                                    checkpoint_path='stc2emb_best.pth',
                                    model=stc2emb,
                                    parallel=parallel)
                elif roc_auc+0.15 < best_auc:
                    break
            except Exception as e:
                print(e)
            
            pd_total_auc = pd.DataFrame(auc_record)
            pd_total_auc.to_csv(f'./loss_record/{skemAdjust}/{random_state}/total_auc_s_TS_{s_type}_ablation_{ablation}.csv', sep = ',')
        
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv(f'./loss_record/{skemAdjust}/{random_state}/total_loss_s_TS_{s_type}_ablation_{ablation}.csv', sep = ',')
    print(total_loss) 


def testt_mimics(EDisease_Model,
                 stc2emb,
                 dloader,
                 device,
                 parallel=parallel,
                 gpus=0,
                 ablation=None,
                 ): 
    
    EDisease_Model.to(device)
    stc2emb.to(device)

    EDisease_Model.eval()
    stc2emb.eval()

        
    if device == 'cuda':
        torch.cuda.set_device(gpus)
    
    total_res_ = []
    
    with torch.no_grad():  
        for batch_idx, sample in tqdm.tqdm(enumerate(dloader)):

            sample = {k:v.to(device) for k,v in sample.items()}
            
                     
            # for structual data
            s,sp, sm = sample['structure'],sample['structure_position_ids'], sample['structure_attention_mask']
            st = sample['structure_time_ids']
               
            if ablation is None:
                s_emb = stc2emb(inputs=s,
                                     attention_mask=sm,
                                     position_ids=sp,
                                     time_ids=st)
            elif ablation == 'mask':
                s_emb = stc2emb(inputs=s,
                                attention_mask=None,
                                     position_ids=sp,
                                     time_ids=st)
            elif ablation == 'vtype':
                s_emb = stc2emb(inputs=s,
                                     attention_mask=sm,
                                     position_ids=sp*0.,
                                     time_ids=st)
            elif ablation == 'vtime':
                s_emb = stc2emb(inputs=s,
                                     attention_mask=sm,
                                     position_ids=sp,
                                     time_ids=st*0.)
            
            predict = EDisease_Model.classifier(s_emb[:,0,:])

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
            
if task=='train_':
    device = f'cuda:{gpus}'
    
    mlp = False
    checkpoint_file = f'../checkpoint_EDs_OnlyS/{skemAdjust}/{random_state}/{ablation}/EDisease_spectrum_TS'
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
                 mlp=mlp,
                 ablation=ablation) 

if task=='test_':
    device = f'cuda:{gpus}'
    
    mlp = False
    checkpoint_file = f'../checkpoint_EDs_OnlyS/{skemAdjust}/{random_state}/{ablation}/EDisease_spectrum_TS'
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
                         device=device,
                         ablation=ablation)               

    fpr, tpr, _ = roc_curve(valres['ground_truth'].values, valres['probability'].values)
    
    roc_auc = auc(fpr,tpr)
    
    valres.to_pickle(f'./result_pickles/{skemAdjust}/{random_state}/EDspectrumTS_OnlyS_{ablation}_{roc_auc*1000:.0f}.pkl')
