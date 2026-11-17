
from pathlib import Path
import joblib
import pandas as pd

from explainability import create_explainer, explain_transaction

# Locate the project folder
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load existing model and scaler
model = joblib.load(PROJECT_DIR / "models" / "fraud_detection.pkl")
scaler = joblib.load(PROJECT_DIR / "models" / "scaler.pkl")

# Load one transaction from your existing dataset
df = pd.read_csv(PROJECT_DIR / "data" / "creditcard.csv")

# Apply the same preprocessing used during model training
df["scaled_amount"] = scaler.transform(
    df[["Amount"]].to_numpy()
)

df.drop("Amount", axis=1, inplace=True)

scaled = df.pop("scaled_amount")
df["scaled_amount"] = scaled

# Separate features from target
X = df.drop("Class", axis=1)

# Select one transaction
transaction = X.iloc[[0]]

# Make prediction
prediction = model.predict(transaction)
probability = model.predict_proba(transaction)[:, 1]

print("Prediction:", "Fraud" if prediction[0] == 1 else "Legitimate")
print("Fraud probability:", probability[0])

# Generate SHAP explanation
explainer = create_explainer(model)
explanation = explain_transaction(explainer, transaction)

# Display the five most influential features
shap_values = explanation.values[0]

importance = pd.Series(
    shap_values,
    index=transaction.columns
)

importance = importance.reindex(
    importance.abs().sort_values(ascending=False).index
)

print("\nTop 5 contributing features:")
print(importance.head(5))
