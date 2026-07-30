import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("models/fraud_detection.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# Title
# -----------------------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
This application predicts whether a credit card transaction is **Fraudulent** or **Legitimate**
using a trained **XGBoost Machine Learning Model**.

### Instructions

1. Upload a CSV file.
2. The file should contain all feature columns used during training.
3. Click **Predict Fraud**.
""")

# -----------------------------
# Upload File
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    # -------------------------
    # Preprocessing
    # -------------------------

    if "Amount" in df.columns:

        df["scaled_amount"] = scaler.transform(
            df[["Amount"]]
        )

        df.drop("Amount", axis=1, inplace=True)

        scaled = df.pop("scaled_amount")

        df["scaled_amount"] = scaled

    # Remove target column if present

    if "Class" in df.columns:

        df = df.drop("Class", axis=1)

    # -------------------------
    # Prediction
    # -------------------------

    if st.button("Predict Fraud"):

        prediction = model.predict(df)

        probability = model.predict_proba(df)[:,1]

        result = df.copy()

        result["Prediction"] = prediction

        result["Fraud Probability"] = probability

        result["Prediction"] = result["Prediction"].map({
            0:"Legitimate",
            1:"Fraud"
        })

        st.subheader("Prediction Results")

        st.dataframe(result)

        fraud = (prediction == 1).sum()

        genuine = (prediction == 0).sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Transactions",
            fraud
        )

        col2.metric(
            "Legitimate Transactions",
            genuine
        )

        csv = result.to_csv(index=False)

        st.download_button(
            "Download Predictions",
            csv,
            "fraud_prediction_results.csv",
            "text/csv"
        )

        st.success("Prediction Completed Successfully!")