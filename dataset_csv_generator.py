import pandas as pd
import os
import numpy as np
import csv

input_csv_path = "/work/scratch/abdul/CLAM/patches_RESULTS/process_list_autogen_EDITED_FEATURES.csv"
output_csv_path = '/work/scratch/abdul/CLAM/tumor_vs_normal_dummy_clean.csv'
df1 = pd.read_csv(input_csv_path)
slide_ids = df1['slide_id'].to_numpy()

df2 = pd.DataFrame(columns=['case_id','slide_id','label'])
case_ids, labels = [], []

for s in slide_ids:
    case_ids.append(s[8:12])
    if s[13] == "0":
        labels.append("tumor_tissue")
    else:
        labels.append("normal_tissue")

case_ids = np.array(case_ids)
labels = np.array(labels)

df2['case_id']=pd.Series(case_ids)
df2['slide_id']=pd.Series(slide_ids)
df2['label']=pd.Series(labels)


with open(output_csv_path, 'wb') as f:
    writer = csv.writer(f)
df2.to_csv(output_csv_path, index=False)
