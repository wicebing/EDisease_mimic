import os, glob, datetime, tqdm

import pandas as pd

db_file_path = '../datahouse/mimic-iv-0.4'


# # =====================================================
# # step 3: add vital sign

# # =====================================================
# # step 3a: extract vital sign from chartevents
filepath = os.path.join(db_file_path, 'icu', 'chartevents.csv')
chartevents = pd.read_csv(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'b_d_items_intubation.csv')
b_d_items = pd.read_csv(filepath)
d_items_temp = b_d_items[['itemid', 'label','b_idx']]

chartevents_vs = chartevents.merge(d_items_temp,how='left',on=['itemid'])
chartevents_vs_dpna=chartevents_vs.dropna(axis=0,subset=['b_idx'])



# filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
# agegender = pd.read_pickle(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
# vital_signs = pd.read_pickle(filepath)


