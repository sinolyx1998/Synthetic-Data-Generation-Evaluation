from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from dp_cgans import DP_CGAN


original_data = pd.read_csv('diff_priv/diabetic_data.csv')

# Preprocessing drop criteria: 
# 1. missing val >= 70% threshold: 'max_glu_serum', 'A1Cresult', 'weight'
# 2. irrelevant: 24 medication types, payment, patient num ...
required_vars = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency", "number_inpatient", 
                "number_diagnoses", "change", "diabetesMed", "readmitted"]

data = original_data[required_vars].copy()

# categorical encoding
categorical_columns = data.select_dtypes(include=["object"]).columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Run this smaller batch to test and see if it works I hope its okkkkkk :(
# *********************************************************************************************
small_data = data.sample(1000, random_state=42)
train_data, test_data = train_test_split(small_data, test_size=0.2, random_state=42)

dp_model = DP_CGAN(
    epochs = 2,
    batch_size = 32,
    verbose = True,
    generator_dim = (256, 256),
    discriminator_dim = (256, 256),
    generator_lr = 1e-4, 
    discriminator_lr = 1e-4,
    discriminator_steps = 1, 
    private = True,
    pac=4
)
dp_model.fit(small_data)
print("done")
#*********************************************************************************************

# # Train 
# dp_model = DP_CGAN(
#     epochs = 200,
#     batch_size = 1024,
#     verbose = True,
#     generator_dim = (256, 256),
#     discriminator_dim = (256, 256),
#     generator_lr = 1e-4, 
#     discriminator_lr = 1e-4,
#     discriminator_steps = 1, 
#     private = True
# )

# dp_model.fit(train_data)

# # # Generate synthetic data
# # synthetic_data = dp_model.sample(len(test_data))
# # synthetic_data.to_csv('synthetic_data.csv', index=False)


# # # Create labels for membership: 1 for training data, 0 for synthetic
# # train_data["membership"] = 1
# # synthetic_data["membership"] = 0

# # # Combine and shuffle
# # combined_data = pd.concat([train_data, synthetic_data]).sample(frac=1, random_state=42)
# # X = combined_data.drop(columns=["membership"])
# # y = combined_data["membership"]

# # # Split for attack model
# # X_train, X_attack, y_train, y_attack = train_test_split(X, y, test_size=0.3, random_state=42)

# # # Train a simple attack model
# # attack_model = RandomForestClassifier(random_state=42)
# # attack_model.fit(X_train, y_train)

# # # Evaluate the attack
# # y_pred = attack_model.predict(X_attack)
# # attack_results = {
# #     "accuracy": accuracy_score(y_attack, y_pred),
# #     "precision": precision_score(y_attack, y_pred),
# #     "recall": recall_score(y_attack, y_pred)
# # }

# # print("Attack Results:", attack_results)
