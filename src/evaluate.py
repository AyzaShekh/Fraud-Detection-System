from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate(model, X_test, y_test):

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(
            y_test, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_test, prediction, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, prediction, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probability)
    }

    matrix = confusion_matrix(y_test, prediction)

    report = classification_report(
        y_test,
        prediction,
        zero_division=0
    )

    return metrics, matrix, report
