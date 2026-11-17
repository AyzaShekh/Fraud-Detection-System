
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.explainability import create_explainer, explain_transaction
from src.model_evaluation import get_model_evaluation


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

# Create SHAP explainer
explainer = create_explainer(model)


# -----------------------------
# Title
# -----------------------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
This application predicts whether a credit card transaction is
**Fraudulent** or **Legitimate** using a trained
**XGBoost Machine Learning Model**.

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

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    # Get the exact feature order expected by the model
    expected_features = list(model.feature_names_in_)

    # Remove optional target column
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Prevent ambiguous input
    if "Amount" in df.columns and "scaled_amount" in df.columns:
        st.error("Please provide either Amount or scaled_amount, not both.")
        st.stop()

    # Scale Amount if the original column is provided
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

        if df["Amount"].isna().any():
            st.error("Amount contains missing or non-numeric values.")
            st.stop()

        try:
            df["scaled_amount"] = scaler.transform(
                df[["Amount"]]
            ).ravel()
        except Exception as e:
            st.error(f"Amount scaling failed: {e}")
            st.stop()

        df = df.drop(columns=["Amount"])

    # Check required features
    missing_features = [
        feature for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        st.stop()

    # Warn about and remove unnecessary columns
    extra_features = [
        column for column in df.columns
        if column not in expected_features
    ]

    if extra_features:
        st.warning(f"Ignoring extra columns: {extra_features}")

    # Keep the exact feature order expected by XGBoost
    df = df[expected_features].copy()

    # Check that every feature contains valid numeric values
    df = df.apply(pd.to_numeric, errors="coerce")

    invalid_columns = df.columns[df.isna().any()].tolist()

    if invalid_columns:
        st.error(
            f"Missing or non-numeric values found in: {invalid_columns}"
        )
        st.stop()

    st.success("CSV validation successful! Ready for fraud prediction.")

    # -------------------------
    # Prediction
    # -------------------------

    if st.button("Predict Fraud"):
        st.session_state["prediction_ready"] = True

    if st.session_state.get("prediction_ready", False):

        prediction = model.predict(df)

        probability = model.predict_proba(df)[:, 1]

        result = df.copy()

        # Add model predictions and scores
        result["Prediction"] = prediction
        result["Fraud Probability"] = probability


        # Assign review risk categories
        def classify_risk(score):
            if score >= 0.70:
                return "High Risk"
            elif score >= 0.30:
                return "Medium Risk"
            else:
                return "Low Risk"


        result["Risk Level"] = [
            classify_risk(score) for score in probability
        ]

        # Recommended review actions

        def recommend_action(risk_level):

            if risk_level == "High Risk":
                return "Priority Manual Review"

            elif risk_level == "Medium Risk":
                return "Secondary Review"

            else:
                return "Routine Monitoring"

        result["Recommended Action"] = result[
            "Risk Level"
        ].apply(recommend_action)


        # Convert prediction numbers into readable labels
        result["Prediction"] = result["Prediction"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        # Filter transactions by risk level

        st.subheader("🔎 Filter Transactions")

        risk_filter = st.selectbox(
            "Select Risk Level",
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )

        if risk_filter == "All":
            filtered_result = result
        else:
            filtered_result = result[
                result["Risk Level"] == risk_filter
                ]

        st.write(f"Showing {len(filtered_result)} transactions")

        st.dataframe(
            filtered_result,
            width="stretch",
            hide_index=True
        )

        # Fraud Detection Summary Report

        st.subheader("📋 Fraud Detection Summary")

        total = len(result)
        fraud_count = (result["Prediction"] == "Fraud").sum()
        legitimate_count = (result["Prediction"] == "Legitimate").sum()
        fraud_percentage = (fraud_count / total) * 100 if total > 0 else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)
        col2.metric("Predicted Fraud", fraud_count)
        col3.metric("Legitimate Transactions", legitimate_count)

        st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

        if fraud_percentage >= 10:
            st.error("Elevated fraud activity detected.")
        elif fraud_percentage >= 3:
            st.warning("Moderate fraud activity detected.")
        else:
            st.success("Low predicted fraud activity.")

        # -------------------------
        # Prediction Results
        # -------------------------

        st.subheader("Prediction Results")




        # -------------------------
        # Fraud Statistics
        # -------------------------

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


        # -------------------------
        # SHAP Explainability
        # -------------------------


        # -------------------------
        # Transaction Risk Analysis
        # -------------------------

        st.divider()

        st.header("📊 Transaction Risk Analysis")

        st.markdown("""
        Review transactions using their model-estimated fraud scores.
        Higher scores indicate that the model considers a transaction
        more likely to belong to the fraud class.
        """)

        # Summary Metrics

        total_transactions = len(df)

        high_risk = (result["Risk Level"] == "High Risk").sum()
        medium_risk = (result["Risk Level"] == "Medium Risk").sum()
        low_risk = (result["Risk Level"] == "Low Risk").sum()

        st.subheader("Risk Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total_transactions)
        col2.metric("High-Risk Transactions", high_risk)
        col3.metric("Medium-Risk Transactions", medium_risk)

        col1, col2, col3 = st.columns(3)

        col1.metric("Low-Risk Transactions", low_risk)
        col2.metric("Highest Fraud Score", f"{probability.max():.2%}")
        col3.metric("Average Fraud Score", f"{probability.mean():.2%}")

        # MODEL PERFORMANCE DASHBOARD

        st.divider()
        st.header("📊 Model Performance Dashboard")

        training_metrics, testing_metrics, matrix = get_model_evaluation()

        # Display performance metrics
        st.subheader("Training vs Testing Performance")

        comparison = pd.DataFrame({
            "Training": training_metrics,
            "Testing": testing_metrics
        })

        st.dataframe(comparison.style.format("{:.2%}"))

        st.subheader("Confusion Matrix")

        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"]
        )



        st.dataframe(confusion_df)




        # -------------------------
        # Fraud Score Distribution
        # -------------------------

        st.subheader("Fraud Score Distribution")

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(
            probability,
            bins=20,
            edgecolor="black"
        )

        ax.set_xlabel("Model-estimated fraud score")
        ax.set_ylabel("Number of transactions")
        ax.set_title("Distribution of Fraud Scores")

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------
        # Highest Scoring Transactions
        # -------------------------

        st.subheader("Transactions Requiring Review")

        review_results = result.copy()

        review_results.insert(
            0,
            "Transaction ID",
            range(1, len(review_results) + 1)
        )

        top_transactions = review_results.sort_values(
            by="Fraud Probability",
            ascending=False
        ).head(10)

        st.dataframe(
            top_transactions,
            width="stretch"
        )

        # -------------------------
        # Select Transaction
        # -------------------------

        st.divider()

        st.header("🤖 Explainable AI Analysis")

        st.markdown("""
        Select a transaction to understand which features
        influenced the XGBoost model's prediction.
        """)


        # -------------------------
        # Global SHAP Feature Importance
        # -------------------------

        st.subheader("🌍 Global Feature Importance")

        st.markdown("""
        This chart shows which features have the greatest
        overall influence on the model's predictions.
        """)

        # Select a sample of transactions
        sample_size = min(500, len(df))

        sample_data = df.sample(
            n=sample_size,
            random_state=42
        )

        # Calculate SHAP values for the sample
        global_explanation = explain_transaction(
            explainer,
            sample_data
        )

        # Display global SHAP importance
        shap.plots.bar(
            global_explanation,
            max_display=15,
            show=False
        )

        fig = plt.gcf()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        st.caption(
            "Features are ranked by their mean absolute SHAP values "
            "across the sampled transactions."
        )

        st.divider()

        # -------------------------
        # Class-Specific SHAP Analysis
        # -------------------------

        st.subheader("🔍 Class-Specific Feature Importance")

        st.markdown("""
        Compare the features influencing transactions predicted
        as fraudulent and those predicted as legitimate.
        """)

        # Predict classes for the sampled transactions
        sample_predictions = model.predict(sample_data)

        # Separate transactions by predicted class
        fraud_sample = sample_data.loc[
            sample_predictions == 1
        ]

        legitimate_sample = sample_data.loc[
            sample_predictions == 0
        ]

        st.write(
            f"Fraud-predicted transactions: {len(fraud_sample)}"
        )

        st.write(
            f"Legitimate-predicted transactions: {len(legitimate_sample)}"
        )

        fraud_col, legitimate_col = st.columns(2)

        # Explain predicted fraud transactions
        with fraud_col:

            st.markdown("#### 🚨 Predicted Fraud")

            if len(fraud_sample) > 0:

                fraud_explanation = explain_transaction(
                    explainer,
                    fraud_sample
                )

                shap.plots.bar(
                    fraud_explanation,
                    max_display=10,
                    show=False
                )

                fig = plt.gcf()

                st.pyplot(
                    fig,
                    clear_figure=True
                )

                plt.close(fig)

            else:
                st.info(
                    "No fraud-predicted transactions found in this sample."
                )

        # Explain predicted legitimate transactions
        with legitimate_col:

            st.markdown("#### ✅ Predicted Legitimate")

            if len(legitimate_sample) > 0:

                legitimate_explanation = explain_transaction(
                    explainer,
                    legitimate_sample
                )

                shap.plots.bar(
                    legitimate_explanation,
                    max_display=10,
                    show=False
                )

                fig = plt.gcf()

                st.pyplot(
                    fig,
                    clear_figure=True
                )

                plt.close(fig)

            else:
                st.info(
                    "No legitimate-predicted transactions found in this sample."
                )

        st.caption(
            "Feature importance is calculated using mean absolute SHAP values "
            "within each predicted group. These groups are based on model "
            "predictions, not verified transaction labels."
        )

        st.divider()



        selected_index = st.selectbox(
            "Select Transaction",
            options=list(range(len(df))),
            format_func=lambda i: (
                f"Transaction {i + 1} "
                f"— Fraud score: {probability[i]:.2%}"
            )
        )

        transaction = df.iloc[[selected_index]]

        selected_prediction = prediction[selected_index]

        st.write(
            "### Selected Transaction Prediction:",
            "Fraud" if selected_prediction == 1 else "Legitimate"
        )

        st.write(
            f"Model-estimated fraud score: "
            f"{probability[selected_index]:.4%}"
        )

        # Generate SHAP explanation

        explanation = explain_transaction(
            explainer,
            transaction
        )

        # -------------------------
        # Top 5 Influential Features
        # -------------------------

        st.subheader("📌 Top 5 Influential Features")

        single_explanation = explanation[0]

        shap_values = single_explanation.values
        feature_values = single_explanation.data
        feature_names = single_explanation.feature_names

        feature_analysis = pd.DataFrame({
            "Feature": feature_names,
            "Feature Value": feature_values,
            "SHAP Contribution": shap_values
        })

        feature_analysis["Influence"] = feature_analysis[
            "SHAP Contribution"
        ].apply(
            lambda value: (
                "Pushes towards Fraud"
                if value > 0
                else "Pushes away from Fraud"
            )
        )

        feature_analysis["Absolute Impact"] = (
            feature_analysis["SHAP Contribution"].abs()
        )

        top_features = feature_analysis.sort_values(
            by="Absolute Impact",
            ascending=False
        ).head(5)

        st.dataframe(
            top_features[
                [
                    "Feature",
                    "Feature Value",
                    "SHAP Contribution",
                    "Influence"
                ]
            ].round(4),
            width="stretch",
            hide_index=True
        )


        st.subheader("SHAP Waterfall Explanation")

        shap.plots.waterfall(
            explanation[0],
            show=False
        )

        fig = plt.gcf()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        # -------------------------

        # -------------------------
        # Download Prediction Reports
        # -------------------------

        st.subheader("📥 Download Reports")

        # Download all transactions
        all_csv = result.to_csv(index=False)

        st.download_button(
            label="Download All Predictions",
            data=all_csv,
            file_name="all_fraud_prediction_results.csv",
            mime="text/csv"
        )

        # Download filtered transactions
        filtered_csv = filtered_result.to_csv(index=False)

        st.download_button(
            label="Download Filtered Predictions",
            data=filtered_csv,
            file_name="filtered_fraud_prediction_results.csv",
            mime="text/csv"
        )

        st.success(
            "Prediction and SHAP analysis completed successfully!"
        )
