from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
import pandas as pd

raw_data = pd.read_csv('diabetic_data.csv')
synthetic_data = pd.read_csv('synthetic_diabetic_data_650B.csv')

# isolate relevant variables
required_vars = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency", "number_inpatient", 
                "number_diagnoses", "change", "diabetesMed", "readmitted"]

synth_df = synthetic_data[required_vars].copy()
raw_df = raw_data[required_vars].copy()

# remove unknown gender from real dataset
real_df = raw_df[raw_df['gender'] != 'Unknown/Invalid']

# find encoder to remap values
# for col in ['gender', 'readmitted']:
#     le = LabelEncoder()
#     le.fit(real_df[col])  # use the original, non-encoded data
#     print(f"\nMapping for '{col}':")
#     for i, cls in enumerate(le.classes_):
#         print(f"  {cls} --> {i}")

# change to same dtype and remap values of readmission/gender
readmit_map = {'<30': 0, '>30': 1, 'NO': 2}
real_df.loc[:, 'readmitted'] = real_df['readmitted'].map(readmit_map)
synth_df['readmitted'] = synth_df['readmitted'].astype(int)

gender_map = {}
synth_df['gender'] = synth_df['gender'].map({0: 'Female', 1: 'Male'})

# downsample to match sample sizes
real_df_resampled = real_df.sample(n=len(synth_df), random_state=42)

def evaluate_fairness(real_df, synth_df, sensitive_attr, outcome):

    fairness_results = {}
    groups = real_df[sensitive_attr].unique()

    for group in groups:
        real_group = real_df[real_df[sensitive_attr] == group][outcome]
        synth_group = synth_df[synth_df[sensitive_attr] == group][outcome]

        real_dist = real_group.value_counts(normalize=True).sort_index()
        synth_dist = synth_group.value_counts(normalize=True).sort_index()

        all_outcomes = sorted(set(real_dist.index).union(set(synth_dist.index)))

        real_probs = real_dist.reindex(all_outcomes, fill_value=0).values
        synth_probs = synth_dist.reindex(all_outcomes, fill_value=0).values

        jsd_score = jensenshannon(real_probs, synth_probs, base=2)
        fairness_results[group] = jsd_score

    return fairness_results

# Adjust the following column names based on your dataset.
sensitive_attribute = "gender"    # example sensitive attribute
outcome_variable = "readmitted"   # example outcome variable

fairness_results = evaluate_fairness(real_df_resampled, synth_df, sensitive_attribute, outcome_variable)
print("\nFairness Evaluation (JSD) Results:")
print(fairness_results)
