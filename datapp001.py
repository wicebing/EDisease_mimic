import os, glob

import pandas as pd


db_file_path = '../datahouse/mimic-iv-0.4'

filepath = os.path.join(db_file_path, 'hosp')

filelist = glob.glob(os.path.join(filepath,'*'))

data = []
for fn in filelist:
    data.append(pd.read_csv(fn,sep=','))