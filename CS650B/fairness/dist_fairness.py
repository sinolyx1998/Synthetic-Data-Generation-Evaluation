from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import LabelEncoder
import pandas as pd

real_df = pd.read_csv('real_df_resampled.csv')
synth_df = pd.read_csv('synth_df_processed.csv')

def evaluate_fairness(real_df, synth_df, sensitive_attr, outcome):

    fairness_JSD = {}
    fairness_W = {}
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
        w_dist = wasserstein_distance(real_group.values, synth_group.values)

        fairness_JSD[group] = jsd_score
        fairness_W[group] = w_dist

        sorted_JSD = dict(sorted(fairness_JSD.items(), key=lambda item: item[0]))
        sorted_W = dict(sorted(fairness_W.items(), key=lambda item: item[0]))

    return sorted_JSD, sorted_W

# Adjust the following column names based on your dataset.
outcome_variable = "readmitted"   # example outcome variable

fairness_age = evaluate_fairness(real_df, synth_df, 'age', outcome_variable)
fairness_gender = evaluate_fairness(real_df, synth_df, 'gender', outcome_variable)
fairness_race = evaluate_fairness(real_df, synth_df, 'race', outcome_variable)

print(f"\nJSD values for age: {fairness_age}\n\n"
      f"JSD values for gender: {fairness_gender}\n\n"
      f"JSD values for race: {fairness_race}\n\n")