import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = "./data/eICU_test.csv"
df = pd.read_csv(file_path)

# Replace '> 89' in age with 90
df['age'] = df['age'].replace('> 89', 90)
# Convert age to numeric (in case there are other issues)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Only keep the first 10000 rows and 5 columns
cols = ["patientunitstayid", "gender", "age", "ethnicity", "hospitaldischargestatus"]
df = df[cols].head(10000)

# Encode categorical columns
for col in ["gender", "ethnicity", "hospitaldischargestatus"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Save back to CSV
# Overwrite the original file
# No header, no index for compatibility with np.loadtxt
# If you want header, set header=True

df.to_csv(file_path, index=False, header=True)
print("Preprocessing complete. Encoded categorical columns as numbers.")
