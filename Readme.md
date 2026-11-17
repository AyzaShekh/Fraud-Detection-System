# 💳 Explainable AI-Based Fraud Detection and Risk Analysis System

## 📌 Project Overview

Financial fraud is a major challenge faced by banks and digital payment systems. Since fraudulent transactions are significantly less common than legitimate transactions, fraud detection is a highly imbalanced classification problem.

This project develops a **Machine Learning-based Credit Card Fraud Detection System** using an XGBoost classifier. It identifies potentially fraudulent transactions and provides risk analysis through an interactive Streamlit web application.

The system also integrates **Explainable Artificial Intelligence (XAI) using SHAP (SHapley Additive exPlanations)** to help users understand which transaction features influence the model's predictions.

## 🎯 Objectives

- Detect potentially fraudulent credit card transactions using machine learning.
- Handle class imbalance during model development using SMOTE.
- Train and compare multiple machine learning algorithms.
- Evaluate model performance using appropriate classification metrics.
- Classify transactions into Low, Medium and High Risk categories.
- Provide recommended review actions based on predicted risk.
- Explain model predictions using SHAP.
- Develop an interactive Streamlit application for fraud analysis.

## ✨ Key Features

- **CSV Upload:** Upload transaction datasets for batch prediction.
- **Input Validation:** Check required features, missing values and invalid numeric inputs.
- **Fraud Prediction:** Predict whether transactions are Legitimate or Fraud.
- **Fraud Score:** Display the model-estimated fraud score for each transaction.
- **Risk Classification:** Categorize transactions into High, Medium and Low Risk.
- **Recommended Actions:** Suggest manual review or routine monitoring based on risk category.
- **Risk Overview:** Display transaction counts and fraud-score summaries.
- **Risk Filtering:** Filter transactions by their assigned risk category.
- **Model Performance Dashboard:** Display evaluation metrics and a confusion matrix.
- **Fraud Score Distribution:** Visualize the distribution of predicted fraud scores.
- **Transaction Review:** Identify transactions with the highest fraud scores.
- **Global SHAP Explanation:** Understand overall feature importance.
- **Class-Specific SHAP Analysis:** Compare feature importance for fraud-predicted and legitimate-predicted transactions.
- **Individual SHAP Explanation:** Inspect influential features and a waterfall explanation for a selected transaction.
- **CSV Download:** Download prediction results.

## 📂 Project Structure

```text
Fraud_Detection_System/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
└── src/
    ├── data_preprocessing.py
    ├── train_model.py
    ├── evaluate.py
    ├── model_evaluation.py
    ├── predict.py
    └── explainability.py
```

*Note: This structure describes the main project files. Update it if your actual folders or filenames differ.*

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**, containing anonymized credit card transactions.

**Dataset source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download the dataset and place `creditcard.csv` inside the `data/` folder.

### Dataset Information

| Description | Value |
|---|---:|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Total columns | 31 |
| Target variable | Class |

### Dataset Features

- `Time`: Time elapsed between transactions and the first transaction in the dataset.
- `V1` to `V28`: Anonymized numerical features.
- `Amount`: Transaction amount.
- `Class`: Target variable.

Target values:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

The dataset is highly imbalanced, making fraud detection a challenging classification task.

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- SHAP
- Joblib
- Streamlit

## 🔄 Project Workflow

1. Data Collection
2. Data Exploration
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Scaling
6. Train-Test Split
7. Handling Class Imbalance using SMOTE
8. Model Training
9. Model Evaluation
10. Model Comparison
11. SHAP-Based Explainability
12. Streamlit Application Development
13. Fraud Prediction and Risk Analysis

## 📈 Exploratory Data Analysis

The following analyses were performed:

- Dataset inspection
- Missing-value analysis
- Duplicate-record detection
- Statistical summary
- Class distribution visualization
- Transaction amount distribution
- Transaction time distribution
- Correlation heatmap
- Feature importance visualization

## 🤖 Machine Learning Models

The following algorithms were trained and evaluated during model development:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- XGBoost Classifier
- Isolation Forest

The saved XGBoost model is integrated into the Streamlit application for fraud prediction.

## ⚖ Handling Imbalanced Data

Fraudulent transactions form a very small proportion of the dataset.

To address class imbalance during model development, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

The test data was kept separate from the oversampling process to support a more reliable evaluation.

## 📏 Model Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

The application also provides a Model Performance Dashboard with evaluation metrics and visualizations.

## 🧠 Explainable Artificial Intelligence (XAI)

The system integrates **SHAP (SHapley Additive exPlanations)** to make machine learning predictions easier to interpret.

The Explainable AI module provides:

### 1. Global Feature Importance

Shows which features have the greatest overall influence on model predictions.

### 2. Class-Specific Feature Importance

Compares feature importance across transactions predicted as fraudulent and those predicted as legitimate.

### 3. Individual Transaction Explanation

Explains the features influencing a selected transaction's prediction.

### 4. Top Five Influential Features

Displays the five features with the largest absolute SHAP contributions for the selected transaction.

### 5. SHAP Waterfall Plot

Visualizes how feature contributions move the model output from its baseline for an individual transaction.

## 📊 Risk Analysis

The application classifies transactions using model-estimated fraud scores:

| Fraud Score | Risk Category |
|---|---|
| 70% or above | High Risk |
| 30% to below 70% | Medium Risk |
| Below 30% | Low Risk |

Recommended review actions:

| Risk Category | Recommended Action |
|---|---|
| High Risk | Priority Manual Review |
| Medium Risk | Secondary Review |
| Low Risk | Routine Monitoring |

**Note:** These thresholds are provisional project settings and should be calibrated before any real-world financial use. Risk categories are model-based indicators, not confirmed fraud decisions.

## 🌐 Streamlit Web Application

The Streamlit application provides an interactive interface for analysing credit card transactions.

Users can:

- Upload a CSV file containing transaction features.
- Validate the uploaded data.
- Generate fraud predictions and scores.
- View transaction risk categories and recommended actions.
- Filter transactions by risk category.
- Explore model evaluation metrics and visualizations.
- View global, class-specific and individual SHAP explanations.
- Download prediction results.

### Run the Application

```bash
streamlit run app.py
```

The application opens locally in your web browser.

## 📦 Installation

### 1. Clone the Repository

After publishing the project on GitHub, replace the placeholder below with your actual repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd Fraud_Detection_System
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare the Dataset

Download `creditcard.csv` from the Kaggle dataset source and place it inside the `data/` directory.

### 6. Run the Application

```bash
streamlit run app.py
```

**Note:** The saved model and scaler files must be available in the `models/` directory. The application expects transaction data with the required model features.

## 🚀 Future Improvements

- Hyperparameter tuning and model optimization.
- Integration with real-time transaction data.
- REST API integration.
- Cloud deployment.
- Model monitoring and drift detection.
- Automated model retraining pipeline.
- Further calibration of risk thresholds.
- Additional explainability methods such as LIME.

## 👨‍💻 Authors

**Ayza Shekh**  
**Dhruv Mishra**

Artificial Intelligence and Data Science  
Academic Mini Project

## 📄 License

This project is developed for educational, academic and portfolio purposes.
