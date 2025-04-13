from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle


df = pd.read_csv('diabetic_data.csv')

# # check unique values for each attribute
attributes = df.columns

# for col in attributes:
#     print(f'Unique values in', col,':', df[col].unique())

# # Preprocessing drop criteria: 
# # 1. missing val >= 30% threshold: 'weight', 'payer_code', 'medical_specialty'
# # 2. irrelevant: encounter_id
df_dropped = df.drop(columns=['weight', 'payer_code', 'medical_specialty'])


# keep only first encounter of patient
duplicates_before = df_dropped['patient_nbr'].value_counts()


df_sorted = df_dropped.sort_values(by=['patient_nbr', 'encounter_id'])
df_first_admission = df_sorted.drop_duplicates(subset='patient_nbr', keep='first')

duplicates_after = df_first_admission['patient_nbr'].value_counts()
# (71518, 47)

# drop encounter and patient number since they will not contribute to readmission rates
df_drop = df_first_admission.drop(columns=['encounter_id', 'patient_nbr'])
# (71518, 45)


# drop unknown/invalid genders
df_gender = df_drop[df_drop['gender'] != 'Unknown/Invalid'] # (71515, 45)
df_admission = df_gender[df_gender['admission_type_id'] != 6] #(66927, 45)


# find zero variance variables
zero_var = []
for col in df_drop.columns:
    if len(df_drop[col].unique()) == 1:
        zero_var.append(col)
    
# remove zero variance variables
df_zerovar = df_admission.drop(columns=['acetohexamide', 'tolbutamide', 'miglitol', 
                                 'troglitazone', 'tolazamide', 'examide', 
                                 'citoglipton', 'glipizide-metformin', 
                                 'glimepiride-pioglitazone', 
                                 'metformin-rosiglitazone', 'metformin-pioglitazone'])

# (71518, 34)

        
# MAPPING
    # for all variables:  NULL, not available, and not mapped are categorized as unknown

#******************************************************************************
#race
#******************************************************************************
race_map = {
    'AfricanAmerican': 1,
    'Asian': 2,
    'Caucasian': 3,
    'Hispanic': 4,
    'Other': 5,
    '?': 6 #classify as unknown
}
df_zerovar.loc[:, 'race'] = df_zerovar['race'].map(race_map)

#******************************************************************************
# Gender
#******************************************************************************
gender_map = {
    'Female': 0,
    'Male': 1
}
df_zerovar.loc[:, 'gender'] = df_zerovar['gender'].map(gender_map)

#******************************************************************************
# Age
#******************************************************************************
age_map = {
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 2,
    '[30-40)': 3,
    '[40-50)': 4,
    '[50-60)': 5,
    '[60-70)': 6,
    '[70-80)': 7,
    '[80-90)': 8,
    '[90-100)': 9
}
df_zerovar.loc[:, 'age'] = df_zerovar['age'].map(age_map)
# for col in df_zerovar.columns:
#     print(f'Unique values in', col,':', df_zerovar[col].unique())

#******************************************************************************
# Admission_type_id
#******************************************************************************
# map values for NULL, not available, and not mapped as unknown category
admission_type_map = {
    5: 0,
    8: 0,
    7: 5 #move Trauma admission up to fill gap of unknown categories
}
df_zerovar.loc[:, 'admission_type_id'] = df_zerovar['admission_type_id'].apply(lambda x: admission_type_map.get(x, x))


#******************************************************************************
# Discharge_disposition_id: 
    # changed to binary 0 --> unknown, 1 --> discharged
#******************************************************************************
# remove expired patients 
df_zerovar = df_zerovar[~df_zerovar['discharge_disposition_id'].isin([11, 19, 20, 21])]

# change all other values to discharged and mapped NULL, not available, and not mapped as unknwon
df_zerovar['discharge_disposition_id'] = np.where(
    df_zerovar['discharge_disposition_id'].isin([18, 25, 26]), 0, 1)

#******************************************************************************
# Admission_source_id: 
    # Unknown: 0
    # Referral: 1
    # Transfer: 2
    # Birth: 3
    # Other: 4
#******************************************************************************
df_zerovar['admission_source_id'] = np.where(
    df_zerovar['admission_source_id'].isin([9, 15, 17, 20, 21]), 0, 
    df_zerovar['admission_source_id'])

df_zerovar['admission_source_id'] = np.where(
    df_zerovar['admission_source_id'].isin([1, 2, 3]), 1, 
    df_zerovar['admission_source_id'])

df_zerovar['admission_source_id'] = np.where(
    df_zerovar['admission_source_id'].isin([4, 5, 6, 10, 18, 22, 25, 26]), 2, 
    df_zerovar['admission_source_id'])

df_zerovar['admission_source_id'] = np.where(
    df_zerovar['admission_source_id'].isin([11, 12, 13, 14, 23, 24]), 3, 4)

#******************************************************************************
# For all 3 diagnoses
    # unknown: 0
    # Circulatory: 1
    # Respiratory: 2
    # Digestive: 3
    # Diabetes: 4
    # Injury: 5
    # Musculoskeletal: 6
    # Genitourinary: 7
    # Neoplasms: 8
    # Other: 9

# Diag 1
#******************************************************************************
df_zerovar['diag_1_numeric'] = pd.to_numeric(df_zerovar['diag_1'], errors='coerce')

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'] == 789) |
    (df_zerovar['diag_1_numeric'] == 783) |
    (df_zerovar['diag_1'].str.contains(r'\?')), 0,  
    df_zerovar['diag_1']  
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(390, 459, inclusive='both')) | 
    (df_zerovar['diag_1_numeric'] == 785), 1, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(460, 519, inclusive='both')) | 
    (df_zerovar['diag_1_numeric'] == 786), 2, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(520, 579, inclusive='both')) | 
    (df_zerovar['diag_1_numeric'] == 787), 3, 
    df_zerovar['diag_1']

)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(250.00, 250.99, inclusive='both')), 4, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(800, 999, inclusive='both')), 5, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(710, 739, inclusive='both')), 6, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(580, 629, inclusive='both')) |
    (df_zerovar['diag_1_numeric'] == 788), 7, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'].between(140, 239, inclusive='both')), 8, 
    df_zerovar['diag_1']
)

df_zerovar['diag_1'] = np.where(
    (df_zerovar['diag_1_numeric'] == 780) |
    (df_zerovar['diag_1_numeric'] == 781) |
    (df_zerovar['diag_1_numeric'] == 782) |
    (df_zerovar['diag_1_numeric'] == 784) |
    (df_zerovar['diag_1_numeric'].between(790, 799, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(240, 249, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(251, 279, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(680, 709, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(1, 139, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(290, 319, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(280, 289, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(320, 359, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(630, 679, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(360, 389, inclusive='both')) |
    (df_zerovar['diag_1_numeric'].between(740, 759, inclusive='both')) |
    (df_zerovar['diag_1'].str.contains(r'^[E-V]')), 9,  
    df_zerovar['diag_1']  
)

df_zerovar = df_zerovar.drop(columns=['diag_1_numeric'])

#******************************************************************************
# Diag 2
#******************************************************************************

df_zerovar['diag_2_numeric'] = pd.to_numeric(df_zerovar['diag_2'], errors='coerce')

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'] == 789) |
    (df_zerovar['diag_2_numeric'] == 783) |
    (df_zerovar['diag_2'].str.contains(r'\?')), 0,  
    df_zerovar['diag_2']  
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(390, 459, inclusive='both')) | 
    (df_zerovar['diag_2_numeric'] == 785), 1, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(460, 519, inclusive='both')) | 
    (df_zerovar['diag_2_numeric'] == 786), 2, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(520, 579, inclusive='both')) | 
    (df_zerovar['diag_2_numeric'] == 787), 3, 
    df_zerovar['diag_2']

)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(250.00, 250.99, inclusive='both')), 4, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(800, 999, inclusive='both')), 5, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(710, 739, inclusive='both')), 6, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(580, 629, inclusive='both')) |
    (df_zerovar['diag_2_numeric'] == 788), 7, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'].between(140, 239, inclusive='both')), 8, 
    df_zerovar['diag_2']
)

df_zerovar['diag_2'] = np.where(
    (df_zerovar['diag_2_numeric'] == 780) |
    (df_zerovar['diag_2_numeric'] == 781) |
    (df_zerovar['diag_2_numeric'] == 782) |
    (df_zerovar['diag_2_numeric'] == 784) |
    (df_zerovar['diag_2_numeric'].between(790, 799, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(240, 249, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(251, 279, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(680, 709, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(1, 139, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(290, 319, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(280, 289, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(320, 359, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(630, 679, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(360, 389, inclusive='both')) |
    (df_zerovar['diag_2_numeric'].between(740, 759, inclusive='both')) |
    (df_zerovar['diag_2'].str.contains(r'^[E-V]')), 9,  
    df_zerovar['diag_2']  
)

df_zerovar = df_zerovar.drop(columns=['diag_2_numeric'])

#******************************************************************************
# Diag 3
#******************************************************************************

df_zerovar['diag_3_numeric'] = pd.to_numeric(df_zerovar['diag_3'], errors='coerce')

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'] == 789) |
    (df_zerovar['diag_3_numeric'] == 783) |
    (df_zerovar['diag_3'].str.contains(r'\?')), 0,  
    df_zerovar['diag_3']  
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(390, 459, inclusive='both')) | 
    (df_zerovar['diag_3_numeric'] == 785), 1, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(460, 519, inclusive='both')) | 
    (df_zerovar['diag_3_numeric'] == 786), 2, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(520, 579, inclusive='both')) | 
    (df_zerovar['diag_3_numeric'] == 787), 3, 
    df_zerovar['diag_3']

)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(250.00, 250.99, inclusive='both')), 4, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(800, 999, inclusive='both')), 5, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(710, 739, inclusive='both')), 6, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(580, 629, inclusive='both')) |
    (df_zerovar['diag_3_numeric'] == 788), 7, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'].between(140, 239, inclusive='both')), 8, 
    df_zerovar['diag_3']
)

df_zerovar['diag_3'] = np.where(
    (df_zerovar['diag_3_numeric'] == 780) |
    (df_zerovar['diag_3_numeric'] == 781) |
    (df_zerovar['diag_3_numeric'] == 782) |
    (df_zerovar['diag_3_numeric'] == 784) |
    (df_zerovar['diag_3_numeric'].between(790, 799, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(240, 249, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(251, 279, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(680, 709, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(1, 139, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(290, 319, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(280, 289, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(320, 359, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(630, 679, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(360, 389, inclusive='both')) |
    (df_zerovar['diag_3_numeric'].between(740, 759, inclusive='both')) |
    (df_zerovar['diag_3'].str.contains(r'^[E-V]')), 9,  
    df_zerovar['diag_3']  
)

df_zerovar = df_zerovar.drop(columns=['diag_3_numeric'])

#******************************************************************************
# Convert str outputs to binary 
#******************************************************************************

max_glu_serum_map = {
    'Norm': 1, 
    '>200': 2, 
    '>300': 3
}
df_zerovar['max_glu_serum'] = df_zerovar['max_glu_serum'].map(max_glu_serum_map)
df_zerovar['max_glu_serum'] = df_zerovar['max_glu_serum'].fillna(0).astype(int)


A1Cresult_map = {
    'Norm': 1, 
    '>7': 2, 
    '>8': 3
}
df_zerovar['A1Cresult'] = df_zerovar['A1Cresult'].map(A1Cresult_map)
df_zerovar['A1Cresult'] = df_zerovar['A1Cresult'].fillna(0).astype(int)


change_map = {
    'No': 0,
    'Ch': 1, 
}
df_zerovar['change'] = df_zerovar['change'].map(change_map)


diabetes_med_map = {
    'No': 0,
    'Yes': 1, 
}
df_zerovar['diabetesMed'] = df_zerovar['diabetesMed'].map(diabetes_med_map)

readmitted_map = {
    'NO': 0,
    '<30': 1, 
    '>30': 1, 
}
df_zerovar['readmitted'] = df_zerovar['readmitted'].map(readmitted_map)

medication_map = {
    'No': 0,
    'Steady': 1, 
    'Up': 2, 
    'Down': 3
}

medications = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 
    'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin'
]
for col in medications:
    df_zerovar[col] = df_zerovar[col].map(medication_map)


# for col in df_zerovar.columns:
#     print(df_zerovar[col].unique())
# print(df_zerovar.shape)

# (65901, 34)

df_zerovar.to_csv('diabetes_clean.csv', index=False)

