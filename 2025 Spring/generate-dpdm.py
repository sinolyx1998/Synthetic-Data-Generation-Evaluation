from docopt import docopt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import os
import sys
sys.path.append("modules/TableDiffusion/tablediffusion")
from models import table_diffusion

usage = '''
Usage: generate-dpdm.py [--epsilon <value>] [--delta <value>] [--cuda <value>] [-i <value>] [-o <value>]

Options:

    --epsilon EPSILON       Specify ε (default is 1.0)
    --delta DELTA           Specify δ (default is 10^-5)
    --cuda BOOL             Whether to use CUDA (default is False)
    -i, --input FILE        Input file (default is patient.csv)
    -o, --output FILE       Output file (default is synthetic_data.csv)
    -h, --help              Show this screen

'''

args = docopt(usage)

# Initialize variables with default values
epsilon = 1.0
delta = 1e-5
cuda = False
infile = "patient.csv"
outfile = "synthetic_data.csv"

# Set variables to user input if applicable
if not args["--epsilon"] is None:
    epsilon = float(args["--epsilon"])

if not args["--delta"] is None:
    delta = float(args["--delta"])

if not args["--cuda"] is None:
    cuda = args["--cuda"].lower() == "true"

if not args["--input"] is None:
    infile = args["--input"]
    assert(os.path.isfile(infile))

if not args["--output"] is None:
    outfile = args["--output"]

# Load your dataset
original_data = pd.read_csv(infile)

# TODO: Preprocessing
# Some things to consider:
# - What do we not need? (IDs, etc.)
# - How do we handle mixed data? (E.g., age values include "> 89". Do we normalize this to "90" or something, or do we treat age as categorical?)
# - We should treat times as numbers. We should probably convert them to seconds out of 86400.
# - We seem to get issues when values are empty, so preprocessing should probably ensure we have a value everywhere.

# TODO: Set delta adaptively. 10^-5 is fine for smaller data sets, but delta should be less than 1/n or ideally less than 1/(n^2).

"""
# Preprocessing
data = original_data.drop(columns=["encounter_id", "patient_nbr"])
categorical_columns = data.select_dtypes(include=["object"]).columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
"""

# Train TableDiffusion
td = table_diffusion.TableDiffusion_Synthesiser(
    epsilon_target=epsilon,
    delta=delta,
    cuda=False,
)
td.fit(
    df=original_data,
    epsilon=epsilon,
    discrete_columns = [
        "gender",
        "age", # should not be categorical
        "ethnicity",
        "apacheadmissiondx",
        "hospitaladmittime24", # should not be categorical
        "hospitaladmitsource",
        "hospitaldischargetime24", # should not be categorical
        "hospitaldischargelocation",
        "hospitaldischargestatus",
        "unittype",
        "unitadmittime24", # should not be categorical
        "unitadmitsource",
        "unitstaytype",
        "unitdischargetime24", # should not be categorical
        "unitdischargelocation",
        "unitdischargestatus",
        "uniquepid", # should be removed
    ],
)

# Generate synthetic data
synthetic_data = td.sample()
synthetic_data.to_csv(outfile, index=False)

# Exit here and do evaluation separately
sys.exit(0)



# Membership Inference Attack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create labels for membership: 1 for training data, 0 for synthetic
train_data["membership"] = 1
synthetic_data["membership"] = 0

# Combine and shuffle
combined_data = pd.concat([train_data, synthetic_data]).sample(frac=1, random_state=42)
X = combined_data.drop(columns=["membership"])
y = combined_data["membership"]

# Split for attack model
X_train, X_attack, y_train, y_attack = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple attack model
attack_model = RandomForestClassifier(random_state=42)
attack_model.fit(X_train, y_train)

# Evaluate the attack
y_pred = attack_model.predict(X_attack)
attack_results = {
    "accuracy": accuracy_score(y_attack, y_pred),
    "precision": precision_score(y_attack, y_pred),
    "recall": recall_score(y_attack, y_pred)
}

print("Attack Results:", attack_results)

