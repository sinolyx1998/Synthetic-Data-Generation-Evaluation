from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

real_df = pd.read_csv('real_df_resampled.csv')
synth_df = pd.read_csv('synth_df_processed.csv')


def equal_odds(real_df, synth_df, sensitive_attr, outcome):
    equal_odds_results = {}

    groups = real_df[sensitive_attr].unique()

    for group in groups:
        # Extract the true and predicted outcomes for this group
        real_group = real_df[real_df[sensitive_attr] == group]
        synth_group = synth_df[synth_df[sensitive_attr] == group]

        y_true = real_group[outcome]
        y_pred = synth_group[outcome]

        print(len(y_true), len(y_pred))  

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate FPR and FNR
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # Avoid division by zero
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # Avoid division by zero

        equal_odds_results[group] = {"FPR": fpr, "FNR": fnr}

    return equal_odds_results

EO = equal_odds(real_df, synth_df, 'gender', 'readmitted')
