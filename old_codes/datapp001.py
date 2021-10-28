import os, glob

import pandas as pd


# db_file_path = '../datahouse/mimic-iv-0.4'

# filepath = os.path.join(db_file_path, 'hosp')

# filelist = glob.glob(os.path.join(filepath,'*'))

# data = []
# for fn in filelist:
#     data.append(pd.read_csv(fn,sep=','))
    
# db_file_path = '../datahouse/mimic-iv-ed-1.0'
# filepath = os.path.join(db_file_path, 'ed')

# filelist = glob.glob(os.path.join(filepath,'*'))

# data2 = []
# for fn in filelist:
#     data2.append(pd.read_csv(fn,sep=','))

# import pickle
# db_file_path = '../datahouse/ntuh_data_20201120_new.pickle'

# data = pd.read_pickle(db_file_path)


db_file_path = '../datahouse/mimic-iv-0.4'

filepath = os.path.join(db_file_path, 'hosp', 'd_icd_diagnoses.csv')


d_icd_diagnoses = pd.read_csv(filepath)

arrest_icds = d_icd_diagnoses[d_icd_diagnoses['long_title'].str.contains("arrest|Arrest", na=False)]

savfilepath = os.path.join(db_file_path, 'data_EDis', 'b_arrest_icds.csv')
# arrest_icds.to_csv(savfilepath)

