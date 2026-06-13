from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class LoanApplication(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float

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


@app.post("/predict")
def predict(data: LoanApplication):

    input_df = sample.copy()

    input_df["AMT_INCOME_TOTAL"] = data.AMT_INCOME_TOTAL
    input_df["AMT_CREDIT"] = data.AMT_CREDIT
    input_df["AMT_ANNUITY"] = data.AMT_ANNUITY
    input_df["AMT_GOODS_PRICE"] = data.AMT_GOODS_PRICE

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    return {
        "default_probability": probability
    }