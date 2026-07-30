# 💳 Fraud Detection System using Machine Learning

## 📌 Project Overview

Financial fraud has become one of the major challenges faced by banks and online payment systems. Fraudulent transactions are extremely rare compared to legitimate transactions, making fraud detection a highly imbalanced classification problem.

This project develops a machine learning-based Fraud Detection System capable of identifying fraudulent credit card transactions using multiple classification algorithms. Various machine learning models are trained, evaluated, and compared to determine the most effective model for fraud detection. A Streamlit web application is also developed to provide an interactive interface for making predictions.

---

# 🎯 Objectives

* Detect fraudulent credit card transactions using machine learning.
* Handle highly imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique).
* Compare multiple machine learning algorithms.
* Evaluate model performance using appropriate classification metrics.
* Deploy the trained model through a Streamlit web application.

---

# 📂 Project Structure

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
├── images/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── model_comparison.png
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
    └── predict.py
```

---

# 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**, which contains anonymized credit card transactions.

Download the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded `creditcard.csv` file inside this `data/` folder before running the project.

### Dataset Information

* Total Transactions: **284,807**
* Legitimate Transactions: **284,315**
* Fraudulent Transactions: **492**
* Total Features: **31**
* Target Variable: **Class**

Target Values:

* **0** → Legitimate Transaction
* **1** → Fraudulent Transaction

The dataset is highly imbalanced, making fraud detection a challenging machine learning task.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Joblib
* Streamlit

---

# 🔄 Project Workflow

1. Data Collection
2. Data Exploration
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Scaling
6. Feature Selection
7. Train-Test Split
8. Handling Imbalanced Data using SMOTE
9. Model Training
10. Model Evaluation
11. Model Comparison
12. Model Saving
13. Streamlit Deployment

---

# 📈 Exploratory Data Analysis

The following analyses were performed:

* Dataset inspection
* Missing value analysis
* Duplicate record detection
* Statistical summary
* Class distribution visualization
* Transaction amount distribution
* Transaction time distribution
* Correlation heatmap
* Feature importance visualization

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Extra Trees Classifier
* XGBoost Classifier
* Isolation Forest

---

# 📏 Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report
* ROC Curve

---

# ⚖ Handling Imbalanced Data

Since fraudulent transactions represent only a small fraction of the dataset, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the training data before model training.

---

# 🌐 Streamlit Web Application

A user-friendly Streamlit application was developed to allow users to:

* Load the trained machine learning model.
* Enter transaction feature values.
* Predict whether a transaction is **Legitimate** or **Fraudulent**.
* Display prediction results instantly.

Run the application locally:

```bash
streamlit run app.py
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Fraud_Detection_System.git
```

Move into the project directory:

```bash
cd Fraud_Detection_System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📊 Results

Multiple machine learning models were compared using several evaluation metrics to determine the best-performing model for fraud detection.

The final selected model was saved using **Joblib** and integrated into the Streamlit application for real-time predictions.

---

# 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Real-time fraud detection with streaming transaction data.
* Model deployment on cloud platforms.
* Integration with REST APIs.
* Explainable AI using SHAP or LIME.
* Automated model retraining pipeline.

---

# 👨‍💻 Author

**Ayza Shekh**

Student | Artificial Intelligence & Data Science

---

# 📄 License

This project is developed for educational, internship, and portfolio purposes.
