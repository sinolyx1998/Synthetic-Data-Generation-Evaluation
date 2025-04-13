import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

original_data = pd.read_csv('diabetic_data.csv')
synthetic_data = pd.read_csv('synthetic_data.csv')
test_data = pd.read_csv('test_preprocessed_data.csv')

# required_vars = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
#                 "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
#                 "num_medications", "number_outpatient", "number_emergency", "number_inpatient", 
#                 "number_diagnoses", "change", "diabetesMed", "readmitted"]

# data = (original_data[required_vars.copy()])

# numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns

# for col in numerical_columns:
#     plt.figure(figsize=(8, 4))
#     plt.hist(original_data[col], bins=30, alpha=0.5, label='Real', color='blue', density=True)
#     plt.hist(synthetic_data[col], bins=30, alpha=0.5, label='Synthetic', color='orange', density=True)
#     plt.title(f"Distribution of {col}")
#     plt.legend()
#     plt.show()



def hellinger_distance(p, q):
        return np.sqrt(0.5*np.sum((np.sqrt(p) - np.sqrt(q))**2))

numeric_cols = ['age', 'num_medications', 'num_lab_procedures', 
                    'time_in_hospital', 'number_inpatient']
    
print("\nFidelity Evaluation (Hellinger distances):")
    # for col in numeric_cols:
    #     holdout_hist, bin_edges = np.histogram(train_df[col], bins=10, density=True)
    #     synthetic_hist, _ = np.histogram(synthetic_data[col], bins=bin_edges, density=True)
    #     holdout_hist = holdout_hist / holdout_hist.sum()
    #     synthetic_hist = synthetic_hist / synthetic_hist.sum()
    #     h_dist = hellinger_distance(holdout_hist, synthetic_hist)
    #     print("  {}: Hellinger distance = {:.8f}".format(col, h_dist))
for col in numeric_cols:
    holdout_hist, bin_edges = np.histogram(test_data[col], bins=10, density=True)
    synthetic_hist, _ = np.histogram(synthetic_data[col], bins=bin_edges, density=True)
    holdout_hist = holdout_hist / holdout_hist.sum()
    synthetic_hist = synthetic_hist / synthetic_hist.sum()
    h_dist = hellinger_distance(holdout_hist, synthetic_hist)
    print("  {}: Hellinger distance = {:.8f}".format(col, h_dist))