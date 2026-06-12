from pathlib import Path
import joblib
import pandas as pd
import shap

from data_prep import clean_data

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load model
model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")

# Load and clean data
df = pd.read_csv(BASE_DIR / "data" / "raw" / "application_train.csv")
df = clean_data(df)


sample = df.drop("TARGET", axis=1).sample(1000, random_state=42)

# Transform using same preprocessor
X_transformed = model.named_steps["preprocessor"].transform(sample)

# Extract trained XGBoost model
xgb_model = model.named_steps["classifier"]

# Feature names after preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# SHAP
explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_transformed)

# Top contributors
shap_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values[0]
})

shap_df["abs_shap"] = shap_df["shap_value"].abs()

top_features = shap_df.sort_values(
    "abs_shap",
    ascending=False
).head(10)

import matplotlib.pyplot as plt

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig(BASE_DIR / "models" / "shap_summary.png")

print("SHAP plot saved.")

