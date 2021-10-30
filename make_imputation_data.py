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

from ycimpute.imputer.mice import MICE
from ycimpute.imputer.knnimput import KNN
from ycimpute.imputer.mida import MIDA
from ycimpute.imputer.gain import GAIN
from ycimpute.imputer import EM

# load data
db_file_path = '../datahouse/mimic-iv-0.4'

filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
icustays_select = pd.read_pickle(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
agegender = pd.read_pickle(filepath)  # no missing values

filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
vital_signs = pd.read_pickle(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
hadmid_first_lab = pd.read_pickle(filepath)

agegender.isna().sum(axis=0).to_csv('./isna_agegender.csv')
vital_signs.isna().sum(axis=0).to_csv('./isna_vital_signs.csv')
hadmid_first_lab.isna().sum(axis=0).to_csv('./isna_hadmid_first_lab.csv')

(vital_signs.isna().sum(axis=0)/len(vital_signs)).to_csv('./isna_vital_signs_percent.csv')
(hadmid_first_lab.isna().sum(axis=0)/len(hadmid_first_lab)).to_csv('./isna_hadmid_first_lab_percent.csv')

# split the dataset
train_set_hadmid = hadmid_first_lab.sample(frac=0.85,random_state=0).index
temp_set_hadmid = hadmid_first_lab.drop(train_set_hadmid)
val_set_hadmid = temp_set_hadmid.sample(frac=0.25,random_state=0).index
test_set_hadmid = temp_set_hadmid.drop(val_set_hadmid).index

trainset_temp = pd.DataFrame(train_set_hadmid)
trainset_temp.columns = ['hadm_id']
trainset_temp = trainset_temp.merge(icustays_select,how='left',on=['hadm_id'])

trainset_stayid = trainset_temp[['stay_id']].drop_duplicates()
trainset_stayid = trainset_stayid.set_index('stay_id').index
trainset_hadmid = trainset_temp[['hadm_id']].drop_duplicates()
trainset_hadmid = trainset_hadmid.set_index('hadm_id').index

valtesat_stayid = vital_signs.drop(trainset_stayid).index
valtesat_hadmid = hadmid_first_lab.drop(trainset_hadmid).index

# select_lab_keys = ['BE', 'Cl', 'Ca', 'Glucose', 'Hb', 'Lac', 'PH', 'Na', 'ALT', 'ALB',
#         'ALP', 'AMY', 'AST', 'BIL-D', 'BIL-T', 'CK', 'Crea',
#         'Lipase', 'Mg', 'P', 'K', 'FreeT4', 'TnT', 'BUN', 'Band',
#         'Eosin', 'Hct', 'PTINR', 'Lym', 'MCH', 'MCHC', 'MCV', 'Myelo',
#         'Seg', 'PLT', 'PTT', 'WBC', 'UrineRBC', 'UrineWBC']

np_vital_signs = vital_signs.values

select_lab_keys = ['BE', 'Cl', 'Ca', 'Glucose', 'Hb', 'Lac', 'PH', 'Na', 'ALT', 'ALB',
       'ALP', 'NH3', 'AMY', 'AST', 'BIL-D', 'BIL-T', 'CK', 'Crea', 'Ddimer',
       'GGT', 'Lipase', 'Mg', 'P', 'K', 'FreeT4', 'TnT', 'BUN', 'Band',
       'Blast', 'Eosin', 'Hct', 'PTINR', 'Lym', 'MCH', 'MCHC', 'MCV', 'Myelo',
       'Seg', 'PLT', 'PTT', 'WBC', 'UrineRBC', 'UrineWBC']

np_hadmid_first_lab = hadmid_first_lab[select_lab_keys].values

# name = 'MICE'
# print(f'{name}_np_vital_signs')
# MICE_np_vital_signs = MICE().complete(np_vital_signs)
# MICE_vital_signs = vital_signs.copy()
# MICE_vital_signs.loc[:]=MICE_np_vital_signs

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
# MICE_vital_signs.to_pickle(filepath)

# print(f'{name}_np_hadmid_first_lab')
# MICE_np_hadmid_first_lab = MICE().complete(np_hadmid_first_lab)
# MICE_hadmid_first_lab = hadmid_first_lab.copy()
# MICE_hadmid_first_lab.loc[:,select_lab_keys]=MICE_np_hadmid_first_lab

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
# MICE_hadmid_first_lab.to_pickle(filepath)

name = 'KNN'
print(f'{name}_np_vital_signs')
MICE_np_vital_signs = KNN(k=4).complete(np_vital_signs)
MICE_vital_signs = vital_signs.copy()
MICE_vital_signs.loc[:]=MICE_np_vital_signs

filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
MICE_vital_signs.to_pickle(filepath)

print(f'{name}_np_hadmid_first_lab')
MICE_np_hadmid_first_lab = KNN(k=4).complete(np_hadmid_first_lab)
MICE_hadmid_first_lab = hadmid_first_lab.copy()
MICE_hadmid_first_lab.loc[:,select_lab_keys]=MICE_np_hadmid_first_lab

filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
MICE_hadmid_first_lab.to_pickle(filepath)

# select_lab_keys = ['BE', 'Cl', 'Ca', 'Glucose', 'Hb', 'Lac', 'PH', 'Na', 'ALT', 'ALB',
#         'ALP', 'AMY', 'AST', 'BIL-T', 'CK', 'Crea',
#         'Lipase', 'Mg', 'P', 'K', 'TnT', 'BUN', 'Band',
#         'Eosin', 'Hct', 'PTINR', 'Lym', 'MCH', 'MCHC', 'MCV', 'Myelo',
#         'Seg', 'PLT', 'PTT', 'WBC', 'UrineRBC', 'UrineWBC']

# np_hadmid_first_lab = hadmid_first_lab[select_lab_keys].values

# name = 'MIDA'
# print(f'{name}_np_vital_signs')
# MICE_np_vital_signs = MIDA().complete(np_vital_signs)
# MICE_vital_signs = vital_signs.copy()
# MICE_vital_signs.loc[:]=MICE_np_vital_signs

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
# MICE_vital_signs.to_pickle(filepath)

# print(f'{name}_np_hadmid_first_lab')
# MICE_np_hadmid_first_lab = MIDA().complete(np_hadmid_first_lab)
# MICE_hadmid_first_lab = hadmid_first_lab.copy()
# MICE_hadmid_first_lab.loc[:,select_lab_keys]=MICE_np_hadmid_first_lab

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
# MICE_hadmid_first_lab.to_pickle(filepath)

# name = 'GAIN'
# print(f'{name}_np_vital_signs')
# MICE_np_vital_signs = GAIN().complete(np_vital_signs)
# MICE_vital_signs = vital_signs.copy()
# MICE_vital_signs.loc[:]=MICE_np_vital_signs

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
# MICE_vital_signs.to_pickle(filepath)

# print(f'{name}_np_hadmid_first_lab')
# MICE_np_hadmid_first_lab = GAIN().complete(np_hadmid_first_lab)
# MICE_hadmid_first_lab = hadmid_first_lab.copy()
# MICE_hadmid_first_lab.loc[:,select_lab_keys]=MICE_np_hadmid_first_lab

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
# MICE_hadmid_first_lab.to_pickle(filepath)

# name = 'EM'
# print(f'{name}_np_vital_signs')
# MICE_np_vital_signs = EM().complete(np_vital_signs)
# MICE_vital_signs = vital_signs.copy()
# MICE_vital_signs.loc[:]=MICE_np_vital_signs

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'stayid_first_vitalsign_{name}.pdpkl')
# MICE_vital_signs.to_pickle(filepath)

# print(f'{name}_np_hadmid_first_lab')
# MICE_np_hadmid_first_lab = EM().complete(np_hadmid_first_lab)
# MICE_hadmid_first_lab = hadmid_first_lab.copy()
# MICE_hadmid_first_lab.loc[:,select_lab_keys]=MICE_np_hadmid_first_lab

# filepath = os.path.join(db_file_path, 'data_EDis_imputation', f'hadmid_first_lab_{name}.pdpkl')
# MICE_hadmid_first_lab.to_pickle(filepath)

