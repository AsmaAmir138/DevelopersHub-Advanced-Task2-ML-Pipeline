# Advanced Task 2: End-to-End ML Pipeline (Customer Churn)

This repository contains my submission for **Advanced Task 2** of the AI/ML Engineering Internship at DevelopersHub Corporation.

##  Objective
To build a production-ready, reusable machine learning pipeline using `scikit-learn` to preprocess features and predict customer churn.

## Methodology & Tech Stack
- **Data Preprocessing:** Standard Scaling for numerical data, One-Hot Encoding for categorical features via `ColumnTransformer`.
- **Model:** Random Forest Classifier wrapped inside an end-to-end `Pipeline`.
- **Hyperparameter Tuning:** Performed via `GridSearchCV` to optimize estimators and tree depth.
- **Serialization:** Complete trained pipeline exported using `joblib`.

##  Key Results
- Successfully achieved automatic pre-processing and prediction in a single execution line.
- Saved the entire structure as `.pkl` for easy API deployment.# DevelopersHub-Advanced-Task2-ML-Pipeline
