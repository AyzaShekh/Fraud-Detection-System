from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def evaluate(model, X_test, y_test):

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:,1]

    print("Accuracy :", accuracy_score(y_test, prediction))

    print("Precision :", precision_score(y_test, prediction))

    print("Recall :", recall_score(y_test, prediction))

    print("F1 Score :", f1_score(y_test, prediction))

    print("ROC AUC :", roc_auc_score(y_test, probability))

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report\n")

    print(classification_report(y_test, prediction))