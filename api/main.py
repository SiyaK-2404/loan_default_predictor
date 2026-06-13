from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model = joblib.load(
    BASE_DIR / "models" / "xgb_model.pkl"
)

# Load saved sample
sample = pd.read_pickle(
    BASE_DIR / "models" / "sample.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Loan Default Predictor API Running"
    }


@app.get("/predict")
def predict():

    probability = float(
        model.predict_proba(sample)[0][1]
    )

    return {
        "default_probability": probability
    }