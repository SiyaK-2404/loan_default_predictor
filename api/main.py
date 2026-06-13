from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI

from src.data_prep import clean_data

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")


@app.get("/")
def home():
    return {"message": "Loan Default Predictor API Running"}


@app.get("/predict")
def predict():

    df = pd.read_csv(
        BASE_DIR / "data" / "raw" / "application_train.csv"
    )

    df = clean_data(df)

    sample = df.drop("TARGET", axis=1).iloc[[0]]

    probability = float(
        model.predict_proba(sample)[0][1]
    )

    return {
        "default_probability": probability
    }