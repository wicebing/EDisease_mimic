import os, glob, datetime, tqdm

import pandas as pd

db_file_path = '../datahouse/mimic-iv-0.4'


# # =====================================================
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

# # =====================================================
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

# # =====================================================
# # step 3: add vital sign

# # =====================================================
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

# # =====================================================
# # step 3b: convert to general form value & data clean / extract vital sign from chartevents
# filepath = os.path.join(db_file_path, 'data_EDis', 'chartevents_vitalsisn.pdpkl')
# chartevents_vs_dpna = pd.read_pickle(filepath)
# chartevents_vs_dpna.loc[:,['charttime']] = pd.to_datetime(chartevents_vs_dpna['charttime'])

# # 3b1 f->c
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='5'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>8000].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/100

# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>800].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/10

# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>80].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=(pd.to_numeric(chartevents_vs_dpna.loc[temp_value_idx,'value'])-32)/1.8

# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='5b'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>200].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/100

# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>80].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=(chartevents_vs_dpna.loc[temp_value_idx,'value']-32)/1.8

# chartevents_vs_dpna.loc[temp,'b_idx'] = '5'

# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='5'].index
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# # clean F TEMPF > 48 TRMPF < 10
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>48].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<10].index)

# # 3b2 inch->cm
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='7'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>200].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/10

# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value<120].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']*2.54

# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='7b'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value<120].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']*2.54

# chartevents_vs_dpna.loc[temp,'b_idx'] = '7'

# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='7'].index
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# # clean height > 250 height < 20
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>250].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<20].index)

# # 3b2 lbs->kg
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='8'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>0].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/2.2046

# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='8b'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value_idx = temp_value[temp_value>300].index
# chartevents_vs_dpna.loc[temp_value_idx,'value']=chartevents_vs_dpna.loc[temp_value_idx,'value']/10

# chartevents_vs_dpna.loc[temp,'b_idx'] = '8'

# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='8'].index
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# # clean weight > 400 weight < 3
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>400].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<3].index)

# # 3b3 0_clean SBP
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='0'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# # clean SBP > 300 < 0
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>300].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<0].index)

# # 3b3 1_clean DBP
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='1'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# # clean DBP > 300 < 0
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>300].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<0].index)

# # 3b3 2_clean HR
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='2'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# # clean HR > 300 < 0
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>300].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<0].index)

# # 3b3 3_clean OXYGEN
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='3'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# # clean OXYGEN > 100 < 0
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>100].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<0].index)

# # 3b3 4_clean RESPIRATION
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='4'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# # clean RESPIRATION > 100 < 0
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>100].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<0].index)

# # 3b3 6_clean BLOODSUGAR
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='6'].index
# chartevents_vs_dpna.loc[temp,'value'] = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])
# temp_value = pd.to_numeric(chartevents_vs_dpna.loc[temp,'value'])

# # clean RESPIRATION > 100 < 0
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value>5000].index)
# chartevents_vs_dpna = chartevents_vs_dpna.drop(temp_value[temp_value<0].index)

# # 3b3 9_clean GCSE
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='9'].index
# temp_value = chartevents_vs_dpna.loc[temp,'value']
# temp_value_idx = temp_value[temp_value=='Spontaneously'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 4

# temp_value_idx = temp_value[temp_value=='To Speech'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 3

# temp_value_idx = temp_value[temp_value=='To Pain'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 2

# temp_value_idx = temp_value[temp_value=='None'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 1

# # 3b3 10_clean GCSV
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='10'].index
# temp_value = chartevents_vs_dpna.loc[temp,'value']

# temp_value_idx = temp_value[temp_value=='Oriented'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 5

# temp_value_idx = temp_value[temp_value=='Confused'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 4

# temp_value_idx = temp_value[temp_value=='Inappropriate Words'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 3

# temp_value_idx = temp_value[temp_value=='Incomprehensible sounds'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 2

# temp_value_idx = temp_value[temp_value=='No Response'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 1

# temp_value_idx = temp_value[temp_value=='No Response-ETT'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = None

# # 3b3 11_clean GCSV
# temp = chartevents_vs_dpna[chartevents_vs_dpna['b_idx']=='11'].index
# temp_value = chartevents_vs_dpna.loc[temp,'value']

# temp_value_idx = temp_value[temp_value=='Obeys Commands'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 6

# temp_value_idx = temp_value[temp_value=='Localizes Pain'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 5

# temp_value_idx = temp_value[temp_value=='Flex-withdraws'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 4

# temp_value_idx = temp_value[temp_value=='Abnormal Flexion'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 3

# temp_value_idx = temp_value[temp_value=='Abnormal extension'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 2

# temp_value_idx = temp_value[temp_value=='No response'].index
# chartevents_vs_dpna.loc[temp_value_idx,'value'] = 1

# filepath = os.path.join(db_file_path, 'data_EDis', 'chartevents_vitalsisn_01.pdpkl')
# chartevents_vs_dpna.to_pickle(filepath)

# # =====================================================
# # step 3c: extract vital sign from chartevents

# filepath = os.path.join(db_file_path, 'data_EDis', 'chartevents_vitalsisn_clean.pdpkl')
# chartevents_vs_dpna = pd.read_pickle(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
# icustays_select = pd.read_pickle(filepath)

# vital_signs_ = []
# length = len(icustays_select)
# # merge the IO events
# for i in tqdm.tqdm(range(length)):
#     sample = icustays_select.iloc[i]
    
#     subject_id = sample['subject_id']
#     hadm_id = sample['hadm_id']
#     stay_id = sample['stay_id']
#     intime = sample['intime']
    
#     temp = chartevents_vs_dpna[chartevents_vs_dpna['stay_id']==stay_id]
#     temp = temp.sort_values(by=['charttime'])
#     temp_first = temp.drop_duplicates(keep='first',subset=['b_idx'])
      
#     temp_first_filter_24 = (temp_first['charttime'] - intime) < datetime.timedelta(minutes=1440)
#     temp_first = temp_first[temp_first_filter_24]
    
#     temp_first_select = temp_first[['b_idx','value']]
#     temp_first_select.loc[:,'b_idx'] = pd.to_numeric(temp_first_select.loc[:,'b_idx'])
    
#     vitalsign_temp = [subject_id,hadm_id,stay_id,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,None]
#     temp_vital_sign = pd.DataFrame([0,1,2,3,4,5,6,7,8,9,10,11,12],index=[0,1,2,3,4,5,6,7,8,9,10,11,12],columns=['b_idx'])
    
#     temp_vital_sign = temp_vital_sign.merge(temp_first_select,how='left',on=['b_idx'])
#     temp_vital_sign.loc[:,'value'] = pd.to_numeric(temp_vital_sign.loc[:,'value'])
#     temp_vital_sign = temp_vital_sign.T
#     columns = ['SBP','DBP','HR','OXYGEN','RESPIRATION','BODYTEMPERATURE','BLOODSUGAR','HEIGHT','WEIGHT','GCSE','GCSV','GCSM','GCS']
    
#     temp_vital_sign.columns = columns
    
#     temp_vital_sign = temp_vital_sign.loc['value']
#     temp_vital_sign['GCS'] = temp_vital_sign[['GCSE','GCSV','GCSM']].sum(skipna=False)

#     temp_vital_sign.name = stay_id
    
#     # temp_vital_sign['subject_id'] = f'{subject_id}'
#     # temp_vital_sign['hadm_id'] = f'{hadm_id}'
#     # temp_vital_sign['stay_id'] = f'{stay_id}'
    
#     vital_signs_.append(temp_vital_sign)

# vital_signs = pd.concat(vital_signs_,axis=1)
# vital_signs = vital_signs.T
# # vital_signs = vital_signs.set_index('stay_id')

# filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
# vital_signs.to_pickle(filepath)

# =====================================================
# step 3d: extract vital sign from chartevents_timesequence

filepath = os.path.join(db_file_path, 'data_EDis', 'chartevents_vitalsisn_clean.pdpkl')
chartevents_vs_dpna = pd.read_pickle(filepath)

filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
icustays_select = pd.read_pickle(filepath)

b_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12]
bb_idx = ['SBP','DBP','HR','OXYGEN','RESPIRATION','BODYTEMPERATURE','BLOODSUGAR','HEIGHT','WEIGHT','GCSE','GCSV','GCSM','GCS']

temp_b_idx = pd.DataFrame(b_idx,index=[0,1,2,3,4,5,6,7,8,9,10,11,12],columns=['b_idx'])
temp_bb_idx = pd.DataFrame(bb_idx,index=[0,1,2,3,4,5,6,7,8,9,10,11,12],columns=['bb_idx'])

temp_vs = pd.concat([temp_b_idx,temp_bb_idx],axis=1)

sel = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'valuenum', 'b_idx']
temp_select = chartevents_vs_dpna[sel]

temp_select.loc[:,'b_idx'] = pd.to_numeric(temp_select.loc[:,'b_idx'])
      
temp_vital_sign = temp_select.merge(temp_vs,how='left',on=['b_idx'])

filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_vitalsign_TS.pdpkl')
temp_vital_sign.to_pickle(filepath)

# # =====================================================
# # step 4: add age gender

# filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
# vital_signs = pd.read_pickle(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
# icustays_select = pd.read_pickle(filepath)

# filepath = os.path.join(db_file_path, 'core', 'patients.csv')
# patients = pd.read_csv(filepath)

# temp_idx = patients[patients['gender']=='F'].index
# patients.loc[temp_idx,'gender'] = 0
# temp_idx = patients[patients['gender']=='M'].index
# patients.loc[temp_idx,'gender'] = 1

# patients.loc[:,'anchor_age'] = pd.to_numeric(patients.loc[:,'anchor_age'])

# agegender_ = []
# length = len(icustays_select)
# # merge the IO events
# for i in tqdm.tqdm(range(length)):
#     sample = icustays_select.iloc[i]
    
#     subject_id = sample['subject_id']
#     hadm_id = sample['hadm_id']
#     stay_id = sample['stay_id']
#     intime = sample['intime']
    
#     temp = patients[patients['subject_id']==subject_id]
#     gender = temp['gender'].astype(int)
#     age = temp['anchor_age'].astype(int)
    
#     agegender_temp = pd.DataFrame([age,gender],index=['AGE','SEX'])
#     agegender_temp.columns = [subject_id]
    
#     agegender_temp = agegender_temp.T
    
#     agegender_.append(agegender_temp) 

# agegender = pd.concat(agegender_,axis=0)
# agegender = agegender.reset_index()
# agegender = agegender.drop_duplicates(subset=['index'])
# agegender = agegender.set_index(['index'])

# filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
# agegender.to_pickle(filepath)

# # step 5: add lab data

# filepath = os.path.join(db_file_path, 'data_EDis', 'agegender.pdpkl')
# agegender = pd.read_pickle(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'stayid_first_vitalsign.pdpkl')
# vital_signs = pd.read_pickle(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
# icustays_select = pd.read_pickle(filepath)

# # =====================================================
# # step 5a: downsize labevents

# filepath = os.path.join(db_file_path, 'hosp', 'labevents.csv')
# labevents = pd.read_csv(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'b_d_labitems.csv')
# b_d_labitems = pd.read_csv(filepath)

# labevents_merge = labevents.merge(b_d_labitems,how='left',on=['itemid'])

# tempidx= labevents_merge[labevents_merge['b_idx']==0].index
# labevents_merge_dropna = labevents_merge.drop(tempidx)

# filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_0.pdpkl')
# labevents_merge_dropna.to_pickle(filepath)

# # =====================================================
# # step 5a: clean labevents
# filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_0.pdpkl')
# labevents_merge_dropna = pd.read_pickle(filepath)

# # =====================================================
# # step 5a1: clean Base Excess
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50802].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<-100].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>100].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a2: clean Chloride, Whole Blood
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50806].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<50].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>150].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a3: clean Free Calcium
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50808].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>4.5].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a4: clean Glucose
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50809].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>3000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a5: clean Hemoglobin
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50811].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>40].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a6: clean Lactate
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50813].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>50].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a7: clean pH
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50820].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<4].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>9].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a8: clean Sodium, Whole Blood
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50824].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<70].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>250].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a9: clean Alanine Aminotransferase (ALT)
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50861].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>20000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a10: clean Albumin
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50862].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>10].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a11: clean Alkaline Phosphatase
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50863].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>6000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a12 skip Ammonia
# # # step 5a12: clean Amylase
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50867].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>12000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a13: skip Asparate Aminotransferase (AST) Bilirubin, Direct Bilirubin, Total
# # # step 5a13: clean Calcium, Total
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50893].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>40].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a14: skip Chloride Creatine Kinase (CK)
# # # step 5a14: clean Creatinine
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50912].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>45].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a15: skip D-Dimer
# # # step 5a15: clean Glucose =50809
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50931].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>6000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a16: clean Lipase
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50956].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>40000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a17: clean Magnesium
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50960].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>15].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a18: clean Phosphate
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50970].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>15].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a19: clean Potassium
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50971].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>15].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a20: skip Sodium Thyroxine (T4), Free Troponin T  Urea Nitrogen Bands Blasts Eosinophils
# # # step 5a20: clean Hematocrit
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51221].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>100].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a21: clean Hemoglobin = 50811
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51222].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>40].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a22: skip INR(PT) Lymphocytes Lymphocytes, Percent
# # # step 5a22: clean MCH 
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51248].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>100].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a23: clean MCHC 
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51249].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0.1].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>100].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a24: skip MCV Myelocytes Neutrophils Platelet Count PTT
# # # step 5a24: clean White Blood Cells 
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51301].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>1000].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a25: clean RBC 
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51493].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>200].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # # step 5a26: clean WBC 
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==51516].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# temp_value_idx = temp_value[temp_value<0].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)
# temp_value_idx = temp_value[temp_value>200].index
# labevents_merge_dropna = labevents_merge_dropna.drop(temp_value_idx)

# # step 5a27: skip Hematocrit High-Sensitivity CRP  INR(PT) White Blood Cells Creatinine, Whole Blood Chloride....# # =====================================================
# # step 5a27: convert Ca total to free Ca /4
# tempidx = labevents_merge_dropna[labevents_merge_dropna['itemid']==50893].index
# temp_value =labevents_merge_dropna.loc[tempidx,'valuenum']
# labevents_merge_dropna.loc[tempidx,'valuenum']=temp_value/4

# labevents_merge_dropna = labevents_merge_dropna.dropna(axis=0,subset=['valuenum'])


# filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean.pdpkl')
# labevents_merge_dropna.to_pickle(filepath)

# temp0 = labevents_merge_dropna.groupby('itemid')
# temp1 = temp0['valuenum'].aggregate(['count','min', 'max', 'mean', 'median', 'std'])
# temp1 = temp1.reset_index()
# temp1 = temp1.merge(b_d_labitems,how='left',on=['itemid'])

# filepath = os.path.join(db_file_path, 'data_EDis', 'b_lab_sel_check.csv')
# temp1.to_csv(filepath)

# # =====================================================
# # step 5B: Extract labdata/ combine b_lab_sel_check elements with addition bb_idx column by manual cluster

# filepath = os.path.join(db_file_path, 'data_EDis', 'b_lab_select.csv')
# b_lab_select = pd.read_csv(filepath)
# b_lab_select = b_lab_select[['itemid','bb_idx']]

# column_temp = b_lab_select['bb_idx']
# column_temp = column_temp.drop_duplicates()
# column_lab = list(column_temp.values)

# filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean.pdpkl')
# labevents_merge_dropna_clean = pd.read_pickle(filepath)

# labevents_merge_dropna_clean_combine = labevents_merge_dropna_clean.merge(b_lab_select,how='left',on=['itemid'])
# labevents_merge_dropna_clean_combine.loc[:,['charttime']] = pd.to_datetime(labevents_merge_dropna_clean_combine['charttime'])

# use_columns = ['subject_id', 'hadm_id', 'specimen_id', 
#        'charttime', 'valuenum','b_idx', 'bb_idx']

# filepath = os.path.join(db_file_path, 'data_EDis', 'labevents_merge_dropna_clean_combine.pdpkl')
# labevents_merge_dropna_clean_combine[use_columns].to_pickle(filepath)

# filepath = os.path.join(db_file_path, 'data_EDis', 'select_temp0.pdpkl')
# icustays_select = pd.read_pickle(filepath)

# labs_ = []
# length = len(icustays_select)
# # merge the IO events
# for i in tqdm.tqdm(range(length)):
#     sample = icustays_select.iloc[i]
    
#     subject_id = sample['subject_id']
#     hadm_id = sample['hadm_id']
#     stay_id = sample['stay_id']
#     intime = sample['intime']
    
#     temp = labevents_merge_dropna_clean_combine[labevents_merge_dropna_clean_combine['hadm_id']==hadm_id]
#     temp = temp.sort_values(by=['charttime'])
#     temp_first = temp.drop_duplicates(keep='first',subset=['bb_idx'])
      
#     temp_first_filter_24 = (temp_first['charttime'] - intime) < datetime.timedelta(minutes=1440)
#     temp_first = temp_first[temp_first_filter_24]
    
#     temp_first_select = temp_first[['bb_idx','valuenum']]
    
#     temp_vital_sign = pd.DataFrame(column_lab,index=column_lab,columns=['bb_idx'])
    
#     temp_vital_sign = temp_vital_sign.merge(temp_first_select,how='left',on=['bb_idx'])
#     temp_vital_sign.loc[:,'valuenum'] = pd.to_numeric(temp_vital_sign.loc[:,'valuenum'])
#     temp_vital_sign = temp_vital_sign.T
    
#     temp_vital_sign.columns = column_lab
    
#     temp_vital_sign = pd.to_numeric(temp_vital_sign.loc['valuenum'])

#     temp_vital_sign.name = hadm_id
    
#     labs_.append(temp_vital_sign)

# labs = pd.concat(labs_,axis=1)
# labs = labs.T
# labs = labs.reset_index()
# labs = labs.drop_duplicates(subset=['index'])
# labs = labs.set_index('index')

# filepath = os.path.join(db_file_path, 'data_EDis', 'hadmid_first_lab.pdpkl')
# labs.to_pickle(filepath)

# # =====================================================
# # step 6: Add ICD diagnosis

# filepath = os.path.join(db_file_path, 'hosp', 'diagnoses_icd.csv')
# diagnoses_icd = pd.read_csv(filepath)

# filepath = os.path.join(db_file_path, 'hosp', 'd_icd_diagnoses.csv')
# d_icd_diagnoses = pd.read_csv(filepath)

# diagnoses_icd_merge = diagnoses_icd.merge(d_icd_diagnoses,how='left',on=['icd_code','icd_version'])

# diagnoses_icd_merge = diagnoses_icd_merge.dropna(axis=0,subset=['long_title'])

# filepath = os.path.join(db_file_path, 'data_EDis', 'diagnoses_icd_merge_dropna.pdpkl')
# diagnoses_icd_merge.to_pickle(filepath)

# # =====================================================
# # step 6: all the preproceesing data

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



