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

# remove unknown values for all sensitive attributes
real_df_gen = raw_df[raw_df['gender'] != 'Unknown/Invalid']
synth_df_gen = synth_df[synth_df['gender'] != 2]

real_df = real_df_gen[real_df_gen['race'] != '?']
synth_df = synth_df_gen[synth_df_gen['race'] != 0]

# find encoder to remap values // can add readmitted to see
for col in ['age', 'gender', 'race']:
    le = LabelEncoder()
    le.fit(real_df[col]) 
    print(f"\nMapping for '{col}':")
    for i, clss in enumerate(le.classes_):
        print(f"  {clss} --> {i}")

# mapping readmission
readmit_map = {'<30': 0, '>30': 1, 'NO': 2}
real_df.loc[:, 'readmitted'] = real_df['readmitted'].map(readmit_map)
synth_df['readmitted'] = synth_df['readmitted'].astype(int)

# remap values for age, gender, race
age_map = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, 
           '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
real_df.loc[:, 'age'] = real_df['age'].map(age_map)

gender_map = {'Female': 0, 'Male': 1}
real_df.loc[:, 'gender'] = real_df['gender'].map(gender_map)

race_map = {'AfricanAmerican': 1, 'Asian': 2, 'Caucasian': 3, 'Hispanic': 4, 
            'Other': 5}
real_df.loc[:, 'race'] = real_df['race'].map(race_map)


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

    sorted_fairness = dict(sorted(fairness_results.items(), key=lambda item: item[0]))

    return sorted_fairness

# Adjust the following column names based on your dataset.
outcome_variable = "readmitted"   # example outcome variable

fairness_age = evaluate_fairness(real_df_resampled, synth_df, 'age', outcome_variable)
fairness_gender = evaluate_fairness(real_df_resampled, synth_df, 'gender', outcome_variable)
fairness_race = evaluate_fairness(real_df_resampled, synth_df, 'race', outcome_variable)

print(f"\nJSD values for age: {fairness_age}\n\n"
      f"JSD values for gender: {fairness_gender}\n\n"
      f"JSD values for race: {fairness_race}\n\n")
