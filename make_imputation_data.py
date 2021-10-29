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

filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
agegender = pd.read_pickle(filepath)  # no missing values

filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
vital_signs = pd.read_pickle(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
hadmid_first_lab = pd.read_pickle(filepath)

agegender.isna().sum(axis=0).to_csv('./isna_agegender.csv')
vital_signs.isna().sum(axis=0).to_csv('./isna_vital_signs.csv')
hadmid_first_lab.isna().sum(axis=0).to_csv('./isna_hadmid_first_lab.csv')

np_vital_signs = vital_signs.values
np_hadmid_first_lab = hadmid_first_lab.values

MICE_np_vital_signs = MICE().complete(np_vital_signs)
MICE_np_hadmid_first_lab = MICE().complete(np_hadmid_first_lab)