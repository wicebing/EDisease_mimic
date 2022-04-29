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
import tqdm

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# load data
db_file_path = '../datahouse/mimic-iv-0.4'

# timesequence vitalsign
filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_vitalsign_TS.pdpkl')
stayid_vitalsign_TS = pd.read_pickle(filepath)

# timesequence lab
filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean_combine.pdpkl')
labevents_merge_dropna_clean_combine = pd.read_pickle(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
vital_signs = pd.read_pickle(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
hadmid_first_lab = pd.read_pickle(filepath)

use_hadm_id = hadmid_first_lab.index
use_stay_id = vital_signs.index

labevents_per_hadm_id ={}
vitalsigns_per_stay_id ={}

for hadm_id in tqdm.tqdm(use_hadm_id):
    if hadm_id in labevents_per_hadm_id.keys():
        continue
    temp_lab = labevents_merge_dropna_clean_combine[labevents_merge_dropna_clean_combine['hadm_id']==hadm_id]
    temp_lab = temp_lab.sort_values(by=['charttime'])
    
    labevents_per_hadm_id[hadm_id]=temp_lab
    
filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_per_hadm_id.pdpkl')
# labevents_per_hadm_id.to_pickle(filepath)
with open(filepath, 'wb') as f:
    pickle.dump(labevents_per_hadm_id, f)


for stay_id in tqdm.tqdm(use_stay_id):
    if stay_id in vitalsigns_per_stay_id.keys():
        continue
    temp_vs = stayid_vitalsign_TS[stayid_vitalsign_TS['stay_id']==stay_id]
    temp_vs = temp_vs.sort_values(by=['charttime'])
    
    vitalsigns_per_stay_id[stay_id]=temp_vs
    
filepath = os.path.join(db_file_path, 'data_EDis', 'vitalsigns_per_stay_id.pdpkl')
# vitalsigns_per_stay_id.to_pickle(filepath)
with open(filepath, 'wb') as f:
    pickle.dump(vitalsigns_per_stay_id, f)
