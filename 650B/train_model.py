from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle

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

