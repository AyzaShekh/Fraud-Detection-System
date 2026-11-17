
import shap


def create_explainer(model):
    """
    Creates a SHAP TreeExplainer for the trained XGBoost model.
    """
    explainer = shap.TreeExplainer(model)

    return explainer


def explain_transaction(explainer, transaction):
    """
    Calculates SHAP values for a transaction.

    transaction must contain the same features
    and feature order used during model training.
    """
    shap_explanation = explainer(transaction)

    return shap_explanation
