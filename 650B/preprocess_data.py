from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# Load your dataset
original_data = pd.read_csv('diabetic_data.csv')

# Preprocessing drop criteria: 
# 1. missing val >= 70% threshold: 'max_glu_serum', 'A1Cresult', 'weight'
# 2. irrelevant: 24 medication types, payment, patient num ...
required_vars = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency", "number_inpatient", 
                "number_diagnoses", "change", "diabetesMed", "readmitted"]

data = (original_data[required_vars.copy()])

# check unique values for each attribute
# for col in data.columns:
#     print(f'Unique values in', col,':', data[col].unique())

# remove unknown values
data = data.loc[:, 'race'] = data[data['race'] != '?']
data = data.loc[:, 'gender'] = data[data['gender'] != 'Unknown/Invalid']

# print(data.info())

# maps objects to int
categorical_columns = data.select_dtypes(include=["object"]).columns

label_encoders = {}
mappings = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# print(mappings)

sampled_data = data.sample(n=10000, random_state=42).reset_index(drop=True)

train_data, test_data = train_test_split(sampled_data, test_size=0.2, 
                                         random_state=42)

# save train data for eval
test_data.to_csv('test_preprocessed_data.csv', index=False)

# Train CTGAN
ctgan = CTGAN(
    epochs = 300,
    discriminator_dim=(64, 64),
    generator_dim = (256, 256),
    verbose = True,
    batch_size = 256,
    pac = 2
)

ctgan.fit(train_data, discrete_columns=list(categorical_columns))

with open('ctgan_model2.pkl', 'wb') as f:
    pickle.dump(ctgan, f)

# Generate synthetic data
train_data_size = len(train_data)
synthetic_data = ctgan.sample(train_data_size)

# save synthetic
synthetic_data.to_csv('synthetic_data.csv', index=False)

