import joblib

model = joblib.load("models/fraud_detection.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model:", type(model).__name__)
print("Scaler:", type(scaler).__name__)

if hasattr(model, "get_params"):
    print("\nModel Parameters:")
    print(model.get_params())

if hasattr(model, "n_features_in_"):
    print("\nNumber of Input Features:", model.n_features_in_)

print("\nModel verification completed!")