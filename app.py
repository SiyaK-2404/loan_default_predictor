
import streamlit as st
import requests

API_URL = "https://loan-default-predictor-fauu.onrender.com/predict"

st.title("Loan Default Predictor")

st.write(
    "Enter applicant details to estimate the probability of loan default."
)

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=150000.0
)

credit = st.number_input(
    "Credit Amount",
    min_value=0.0,
    value=500000.0
)

annuity = st.number_input(
    "Annuity Amount",
    min_value=0.0,
    value=25000.0
)

goods_price = st.number_input(
    "Goods Price",
    min_value=0.0,
    value=12000.0
)

if st.button("Predict Default Risk"):

    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    if response.status_code == 200:

        probability = response.json()["default_probability"]

        st.success(
            f"Default Probability: {probability:.2%}"
        )

        if probability < 0.3:
            st.success("Low Risk Applicant")

        elif probability < 0.6:
            st.warning("Moderate Risk Applicant")

        else:
            st.error("High Risk Applicant")

    else:
        st.error(
            "Unable to get prediction from API."
        )

st.subheader("Model Explainability")

st.image(
    "models/shap_summary.png",
    caption="SHAP Feature Importance Summary"
)
