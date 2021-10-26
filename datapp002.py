import os, glob, datetime, tqdm

import pandas as pd

db_file_path = '../datahouse/mimic-iv-0.4'



filepath = os.path.join(db_file_path, 'data_EDis', 'b_arrest_icds.csv')
arrest_icds = pd.read_csv(filepath)
arrest_icds['select'] = 1

# cardiac arrest/ > 18yrs, drop trauma/ sicu

filepath = os.path.join(db_file_path, 'hosp', 'diagnoses_icd.csv')
diagnoses_icd = pd.read_csv(filepath)


diagnoses_icd_arrest = diagnoses_icd.merge(arrest_icds,how='inner',on='icd_code')

sample = diagnoses_icd_arrest.iloc[0]

subject_id = sample['subject_id']
hadm_id = sample['hadm_id']


filepath = os.path.join(db_file_path, 'hosp', 'labevents.csv')
labevents = pd.read_csv(filepath)

filepath = os.path.join(db_file_path, 'icu', 'icustays.csv')
icustays = pd.read_csv(filepath)

filepath = os.path.join(db_file_path, 'icu', 'inputevents.csv')
inputevents = pd.read_csv(filepath)
inputevents.loc[:,['starttime']] = pd.to_datetime(inputevents['starttime'])
inputevents = inputevents.dropna(axis=0,subset=['totalamount'])

filepath = os.path.join(db_file_path, 'icu', 'outputevents.csv')
outputevents = pd.read_csv(filepath)
outputevents.loc[:,['charttime']] = pd.to_datetime(outputevents['charttime'])

# study group patient list is this icustays_select
icustays_select = icustays[(icustays['first_careunit']=='Medical Intensive Care Unit (MICU)') |
                           (icustays['first_careunit']=='Cardiac Vascular Intensive Care Unit (CVICU)') |
                           (icustays['first_careunit']=='Coronary Care Unit (CCU)')
                           ]

icustays_select.loc[:,['intime']] = pd.to_datetime(icustays_select['intime'])
icustays_select.loc[:,['outtime']] = pd.to_datetime(icustays_select['outtime'])
icustays_select.loc[:,['io_24']] = 0. 
icustays_select.loc[:,['i_24']] = 0. 
icustays_select.loc[:,['o_24']] = 0. 


length = len(icustays_select)
# merge the IO events
for i in tqdm.tqdm(range(length)):
    
    sample = icustays_select.iloc[i]
    
    subject_id = sample['subject_id']
    hadm_id = sample['hadm_id']
    stay_id = sample['stay_id']
    intime = sample['intime']
    
    inputevents_temp = inputevents[inputevents['stay_id']==stay_id]
    temp = (inputevents_temp['starttime'] - intime) < datetime.timedelta(minutes=1440)
    inputevents_temp_24 = inputevents_temp[temp]
    inputamount24 = inputevents_temp_24['totalamount'].sum()

    outputevents_temp = outputevents[outputevents['stay_id']==stay_id]
    temp = (outputevents_temp['charttime'] - intime) < datetime.timedelta(minutes=1440)
    outputevents_temp_24 = outputevents_temp[temp]
    outputamount24 = outputevents_temp_24['value'].sum()
    
    idx = icustays_select[icustays_select['stay_id']==stay_id].index
    
    icustays_select.loc[idx,['io_24','i_24','o_24']] = inputamount24-outputamount24,inputamount24,outputamount24

filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
icustays_select.to_pickle(filepath)
