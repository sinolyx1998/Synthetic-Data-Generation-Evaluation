from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your dataset
original_data = pd.read_csv('diabetic_data.csv')

# Preprocessing
data = original_data.drop(columns=["encounter_id", "patient_nbr"])
categorical_columns = data.select_dtypes(include=["object"]).columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

sampled_data = data.sample(n=10000, random_state=42).reset_index(drop=True)

train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42)

test_data.to_csv('synthetic_test.csv', index=False)

# Train CTGAN
ctgan = CTGAN(
    epochs=200,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    verbose=True
)
#verbose=True pac=10 batch_size=1024,
ctgan.fit(train_data, discrete_columns=list(categorical_columns))

# Generate synthetic data
synthetic_data = ctgan.sample(len(test_data))
synthetic_data.to_csv('synthetic_test.csv', index=False)

