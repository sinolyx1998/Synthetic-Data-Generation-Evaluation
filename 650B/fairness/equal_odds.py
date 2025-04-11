from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
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

def equal_odds(real_df, synth_df, sensitive_attr, outcome):
    equal_odds_results = {}

    groups = real_df[sensitive_attr].unique()

    for group in groups:
        # Extract the true and predicted outcomes for this group
        real_group = real_df[real_df[sensitive_attr] == group]
        synth_group = synth_df[synth_df[sensitive_attr] == group]

        y_true = real_group[outcome]
        y_pred = synth_group[outcome]

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate FPR and FNR
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # Avoid division by zero
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # Avoid division by zero

        equal_odds_results[group] = {"FPR": fpr, "FNR": fnr}

    return equal_odds_results