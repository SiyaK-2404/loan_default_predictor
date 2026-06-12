from pathlib import Path
import joblib
import pandas as pd

from data_prep import clean_data

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")

# Load raw data
df = pd.read_csv(BASE_DIR / "data" / "raw" / "application_train.csv")

# Apply same cleaning used during training
df = clean_data(df)

# Take one sample applicant
sample = df.drop("TARGET", axis=1).iloc[[0]]

# Predict probability of default
default_probability = model.predict_proba(sample)[0][1]

print(f"Default Probability: {default_probability:.4f}")