from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define target column
target_column = "readmitted"  # Replace with the correct target column

# Extract features and target
X_synthetic = synthetic_data.drop(columns=[target_column, "membership"])
y_synthetic = synthetic_data[target_column]
X_real = test_data.drop(columns=[target_column])
y_real = test_data[target_column]

# Train a Random Forest model on synthetic data
utility_model = RandomForestClassifier(random_state=42)
utility_model.fit(X_synthetic, y_synthetic)

# Test the model on real data
y_pred_utility = utility_model.predict(X_real)

# Evaluate performance
utility_results = {
    "accuracy": accuracy_score(y_real, y_pred_utility),
    "precision": precision_score(y_real, y_pred_utility, average="weighted"),
    "recall": recall_score(y_real, y_pred_utility, average="weighted"),
    "f1_score": f1_score(y_real, y_pred_utility, average="weighted")
}
print("Utility Results:", utility_results)

