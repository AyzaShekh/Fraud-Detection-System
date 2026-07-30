import joblib


def predict_transaction(features):

    model = joblib.load("../models/fraud_detection.pkl")

    prediction = model.predict([features])

    if prediction[0] == 1:

        print("Fraudulent Transaction")

    else:

        print("Legitimate Transaction")