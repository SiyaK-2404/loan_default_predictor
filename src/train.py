from data_prep import load_data, clean_data

# training code here
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from data_prep import load_data, clean_data


# Load and clean data
df = load_data("../data/raw/application_train.csv")
df = clean_data(df)

# Features and target
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Column types
cat_cols = X.select_dtypes(include=["object", "string"]).columns
num_cols = X.select_dtypes(exclude=["object", "string"]).columns

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print(f"ROC-AUC: {auc:.4f}")

# Save model
joblib.dump(model, "../models/xgb_model.pkl")

print("Model saved successfully.")