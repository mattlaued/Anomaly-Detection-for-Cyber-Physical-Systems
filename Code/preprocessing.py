'''
Author: @mattlaued

This script will preprocess the 2 normal data files and produce 2 output files:
1. Resultant preprocessed csv file.
2. Pickle file for you to use the scaler in the future

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Change to Path of Folder with Data
path_to_data = "SWaT_Data"

df1 = pd.read_excel(f"{path_to_data}/SWaT_Dataset_Normal_v0.xlsx", skiprows=[0])
df2 = pd.read_excel(f"{path_to_data}/SWaT_Dataset_Normal_v1.xlsx", skiprows=[0])

df_new = df1.append(df2)

del df1
del df2

# Relabel
# df_new.replace(["A ttack", "Attack", "Normal"], [1, 1, 0], inplace=True)
df_new.drop(columns="Normal/Attack", inplace=True)

mv = []
p = []
cont = []

for col in df_new.columns:
    if col[0] == "M":
        # Categorical data for MV
        mv.append(col)
    elif col[0] == "P" and len(col) == 4:
        # Categorical data for Pump
        p.append(col)
    elif col[0:6] != "Normal" and col[0:5] != " Time":
        # Continuous Data for Non Label and Non Index Columns
        cont.append(col)

scaler = StandardScaler()
df_new[cont] = scaler.fit_transform(df_new[cont])

# Scale to 0, 0.5 and 1
df_new[mv] /= 2

# Make "2" into "1" for pump data
# "1" remains as "1"
df_new[p] = 2 - df_new[p]


# Save Preprocessed Data and Scaler
df_new.to_csv(f"{path_to_data}/SWaT_Dataset_Normal_2015_Combined_Scaled.csv", index=False)

with open(f"{path_to_data}/scalerX.pkl", 'wb') as f:
    pickle.dump(scaler, f)