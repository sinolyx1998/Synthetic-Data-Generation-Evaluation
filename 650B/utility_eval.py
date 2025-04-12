from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

synthetic_data = pd.read_csv('synthetic_test.csv')
synth_data = pd.read_csv('synthetic_diabetic_data_650B.csv')
test_data = pd.read_csv('test_preprocessed_data.csv')

# Define target column
target_column = "readmitted"  
sensitive_attributes = ['age', 'race', 'gender']

# Extract features and target
X_synthetic = synthetic_data.drop(columns=[target_column] + sensitive_attributes)
y_synthetic = synthetic_data[target_column]

X_real = test_data.drop(columns=[target_column] + sensitive_attributes)
y_real = test_data[target_column]

# Train a Random Forest model on synthetic data
utility_model = RandomForestClassifier(random_state=42)
utility_model.fit(X_synthetic, y_synthetic)

# Test the model on real data
y_pred_utility = utility_model.predict(X_real)
y_prob_utility = utility_model.predict_proba(X_real)[:, 1]

# Evaluate performance
utility_results = {
    "accuracy": accuracy_score(y_real, y_pred_utility),
    "precision": precision_score(y_real, y_pred_utility, average="weighted"),
    "recall": recall_score(y_real, y_pred_utility, average="weighted"),
    "f1_score": f1_score(y_real, y_pred_utility, average="weighted")
}

if len(set(y_real)) == 2:
    print("ROC AUC:", roc_auc_score(y_real, y_prob_utility))
else:
     print("ROC AUC (multi-class):", roc_auc_score(y_real, utility_model.predict_proba(X_real), multi_class='ovo'))

print("Utility Results:", utility_results)

