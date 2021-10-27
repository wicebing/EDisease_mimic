import os, glob, datetime, tqdm

import pandas as pd

db_file_path = '../datahouse/mimic-iv-0.4'


# # step 1: drop trauma/ sicu

# filepath = os.path.join(db_file_path, 'icu', 'icustays.csv')
# icustays = pd.read_csv(filepath)

# filepath = os.path.join(db_file_path, 'icu', 'inputevents.csv')
# inputevents = pd.read_csv(filepath)
# inputevents.loc[:,['starttime']] = pd.to_datetime(inputevents['starttime'])
# inputevents = inputevents.dropna(axis=0,subset=['totalamount'])

# temp = inputevents[(inputevents['itemid']==226361)|
#                     (inputevents['itemid']==222168)|
#                     (inputevents['itemid']==225943)
#                     ]
# inputevents = inputevents.drop(temp.index)

# temp = inputevents[(inputevents['statusdescription']=='Stopped')|
#                     (inputevents['itemid']=='Paused')
#                     ]
# inputevents = inputevents.drop(temp.index)

# filepath = os.path.join(db_file_path, 'icu', 'outputevents.csv')
# outputevents = pd.read_csv(filepath)
# outputevents.loc[:,['charttime']] = pd.to_datetime(outputevents['charttime'])

# temp = outputevents[(outputevents['itemid']==227489)|
#                     (outputevents['itemid']==227488)|
#                     (outputevents['itemid']==226633)
#                     ]
# outputevents = outputevents.drop(temp.index)

# filepath = os.path.join(db_file_path, 'icu', 'd_items.csv')
# d_items = pd.read_csv(filepath)

# d_items_temp = d_items[['itemid', 'label']]



# # study group patient list is this icustays_select
# icustays_select = icustays[(icustays['first_careunit']=='Medical Intensive Care Unit (MICU)') |
#                             (icustays['first_careunit']=='Cardiac Vascular Intensive Care Unit (CVICU)') |
#                             (icustays['first_careunit']=='Coronary Care Unit (CCU)')
#                             ]

# icustays_select.loc[:,['intime']] = pd.to_datetime(icustays_select['intime'])
# icustays_select.loc[:,['outtime']] = pd.to_datetime(icustays_select['outtime'])
# icustays_select.loc[:,['io_24']] = 0. 
# icustays_select.loc[:,['i_24']] = 0. 
# icustays_select.loc[:,['o_24']] = 0. 

# # step 2: add io amount

# length = len(icustays_select)
# # merge the IO events
# for i in tqdm.tqdm(range(length)):
    
#     sample = icustays_select.iloc[i]
    
#     subject_id = sample['subject_id']
#     hadm_id = sample['hadm_id']
#     stay_id = sample['stay_id']
#     intime = sample['intime']
    
#     inputevents_temp = inputevents[inputevents['stay_id']==stay_id]
#     temp = (inputevents_temp['starttime'] - intime) < datetime.timedelta(minutes=1440)
#     inputevents_temp_24 = inputevents_temp[temp]
#     inputamount24 = inputevents_temp_24['totalamount'].sum()

#     outputevents_temp = outputevents[outputevents['stay_id']==stay_id]
#     temp = (outputevents_temp['charttime'] - intime) < datetime.timedelta(minutes=1440)
#     outputevents_temp_24 = outputevents_temp[temp]
#     outputamount24 = outputevents_temp_24['value'].sum()
    
#     idx = icustays_select[icustays_select['stay_id']==stay_id].index
    
#     icustays_select.loc[idx,['io_24','i_24','o_24']] = inputamount24-outputamount24,inputamount24,outputamount24

# filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
# icustays_select.to_pickle(filepath)

# # step 3: add vital sign
filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
icustays_select = pd.read_pickle(filepath)

# # step 3a: extract vital sign from chartevents
# filepath = os.path.join(db_file_path, 'icu', 'chartevents.csv')
# chartevents = pd.read_csv(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'b_d_items.csv')
# b_d_items = pd.read_csv(filepath)
# d_items_temp = b_d_items[['itemid', 'label','b_idx']]

# chartevents_vs = chartevents.merge(d_items_temp,how='left',on=['itemid'])
# chartevents_vs_dpna=chartevents_vs.dropna(axis=0,subset=['b_idx'])

# filepath = os.path.join(db_file_path, 'data_EDis', 'chartevents_vitalsisn.pdpkl')
# chartevents_vs_dpna.to_pickle(filepath)

# # step 3b: convert to general form value & data clean / extract vital sign from chartevents
filepath = os.path.join(db_file_path, 'data_EDis', 'chartevents_vitalsisn.pdpkl')
chartevents_vs_dpna = pd.read_pickle(filepath)
chartevents_vs_dpna.loc[:,['charttime']] = pd.to_datetime(chartevents_vs_dpna['charttime'])

# 3b1 f->c
temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='5'].index
chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value>8000].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/100

temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value>800].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/10

temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value>80].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=(pd.to_numeric(chartevents_vs_dpna.loc[temp_value_idx,'value'])-32)/1.8

chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='5b'].index
chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value>200].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/100

temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value>80].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=(chartevents_vs_dpna.loc[temp_value_idx,'value']-32)/1.8

chartevents_vs_dpna.loc[temp,'b_idx'] = '5'


temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='5'].index
temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# clean F TEMPF > 48 TRMPF < 10
chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>48].index)
chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<10].index)

# 3b2 inch->cm
temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='7'].index
chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value>200].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/10

temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value<120].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']*2.54

temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='7b'].index
chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
temp_value_idx = temp_value[temp_value<120].index
chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']*2.54

chartevents_vs_dpna.loc[temp,'b_idx'] = '7'

temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='7'].index
temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# clean height > 250 height < 20
chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>250].index)
chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<20].index)

length = len(icustays_select)
# merge the IO events
for i in tqdm.tqdm(range(length)):
    sample = icustays_select.iloc[i]
    
    subject_id = sample['subject_id']
    hadm_id = sample['hadm_id']
    stay_id = sample['stay_id']
    intime = sample['intime']

# ['SBP','DBP','HR','OXYGEN','RESPIRATION','BODYTEMPERATURE','BLOODSUGAR','HEIGHT','WEIGHT','GCSE','GCSV','GCSM','GCS']=
