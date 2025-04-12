import pandas as pd
import matplotlib.pyplot as plt

original_data = pd.read_csv('diabetic_data.csv')
synthetic_data = pd.read_csv('synthetic_data.csv')

required_vars = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency", "number_inpatient", 
                "number_diagnoses", "change", "diabetesMed", "readmitted"]

data = (original_data[required_vars.copy()])

numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    plt.hist(original_data[col], bins=30, alpha=0.5, label='Real', color='blue', density=True)
    plt.hist(synthetic_data[col], bins=30, alpha=0.5, label='Synthetic', color='orange', density=True)
    plt.title(f"Distribution of {col}")
    plt.legend()
    plt.show()