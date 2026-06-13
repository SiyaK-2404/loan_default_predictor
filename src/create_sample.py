import pandas as pd
from data_prep import load_data, clean_data

df = load_data("data/raw/application_train.csv")
df = clean_data(df)

sample = df.drop("TARGET", axis=1).iloc[[0]]

sample.to_pickle("models/sample.pkl")

print("sample.pkl saved")