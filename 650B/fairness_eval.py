from scipy.spatial.distance import jensenshannon

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

fairness_results = evaluate_fairness(data, synthetic_data, sensitive_attribute, outcome_variable)
print("\nFairness Evaluation (JSD) Results:")
print(fairness_results)
