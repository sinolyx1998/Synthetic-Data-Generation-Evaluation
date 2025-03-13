from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import sys
sys.path.append("modules/TableDiffusion/tablediffusion")
from models import table_diffusion

# Load your dataset
original_data = pd.read_csv('patient.csv')

"""
# Preprocessing
data = original_data.drop(columns=["encounter_id", "patient_nbr"])
categorical_columns = data.select_dtypes(include=["object"]).columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
"""

epsilon_target = 1.0

# Train TableDiffusion
td = table_diffusion.TableDiffusion_Synthesiser(
    epsilon_target=epsilon_target,
)
td.fit(
    df=original_data,
    epsilon=epsilon_target,
)

# Generate synthetic data
synthetic_data = td.sample()
synthetic_data.to_csv('synthetic_data.csv', index=False)

# Membership Inference Attack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create labels for membership: 1 for training data, 0 for synthetic
train_data["membership"] = 1
synthetic_data["membership"] = 0

# Combine and shuffle
combined_data = pd.concat([train_data, synthetic_data]).sample(frac=1, random_state=42)
X = combined_data.drop(columns=["membership"])
y = combined_data["membership"]

# Split for attack model
X_train, X_attack, y_train, y_attack = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple attack model
attack_model = RandomForestClassifier(random_state=42)
attack_model.fit(X_train, y_train)

# Evaluate the attack
y_pred = attack_model.predict(X_attack)
attack_results = {
    "accuracy": accuracy_score(y_attack, y_pred),
    "precision": precision_score(y_attack, y_pred),
    "recall": recall_score(y_attack, y_pred)
}

print("Attack Results:", attack_results)

