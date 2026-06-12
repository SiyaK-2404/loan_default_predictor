import numpy as np
import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath)


def clean_data(df):
    df = df.copy()

    if "SK_ID_CURR" in df.columns:
        df = df.drop("SK_ID_CURR", axis=1)

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    missing_pct = df.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > 60].index

    df = df.drop(columns=cols_to_drop)

    return df