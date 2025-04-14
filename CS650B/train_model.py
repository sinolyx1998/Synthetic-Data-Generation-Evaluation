from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle


df=pd.read_csv('CS650B/datasets/diabetes_clean_final.csv')

train_data, test_data = train_test_split(df, test_size=0.2, 
                                         random_state=42)

# save train data for eval
test_data.to_csv('test_data_final.csv', index=False)
test_data.to_csv('train_data_final.csv', index=False)


# Train CTGAN
ctgan = CTGAN(
    epochs = 300,
    discriminator_dim=(64, 64),
    generator_dim = (256, 256),
    verbose = True,
    batch_size = 256,
    pac = 2
)

ctgan.fit(train_data)

with open('ctgan_model.pkl', 'wb') as f:
    pickle.dump(ctgan, f)

# Generate synthetic data
train_data_size = len(train_data)
synthetic_data = ctgan.sample(train_data_size)

# save synthetic
synthetic_data.to_csv('synthetic_data_final.csv', index=False)

