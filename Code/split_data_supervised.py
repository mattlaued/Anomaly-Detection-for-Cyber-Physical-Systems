'''
Author: @mattlaued

This script will preprocess the 2 normal data files and produce 2 output files:
1. Resultant preprocessed csv file.
2. Pickle file for you to use the scaler in the future

'''

# TO BE CHANGED
# file_path should be a .xlsx file for the attack data
file_path = ""

df_attack = pd.read_excel(file_path)

# Relabel
df_attack.replace(["A ttack", "Attack", "Normal"], [1, 1, 0], inplace=True)

df_train = df_attack.iloc[50000:]
df_test = df_attack.iloc[:50000]

print("Training Data Split:\n", df_train["Normal/Attack"].value_counts())
print("\nTesting Data Split:\n", df_test["Normal/Attack"].value_counts())

# TO BE CHANGED
# With .csv extensions
file_path_train = ""
file_path_test = ""

df_train.to_csv(file_path_train, index=False)
df_test.to_csv(file_path_test, index=False)