
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1 Score": f1_score(y, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities)
    }


def get_model_evaluation():

    model = joblib.load("models/fraud_detection.pkl")

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")["Class"]

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")["Class"]

    training_metrics = calculate_metrics(model, X_train, y_train)
    testing_metrics = calculate_metrics(model, X_test, y_test)

    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    return training_metrics, testing_metrics, matrix
