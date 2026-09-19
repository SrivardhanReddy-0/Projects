# 📉 Customer Churn Prediction

A Machine Learning classification project that predicts whether a customer is likely to leave a service.

## Project Overview

The model uses customer information such as:

- Age
- Tenure
- Monthly charges
- Total charges
- Contract type
- Support calls

to predict whether the customer will churn.

## Machine Learning Algorithm

Logistic Regression

## Workflow

Dataset
→ Data Analysis
→ Feature Selection
→ Categorical Encoding
→ Train/Test Split
→ Logistic Regression
→ Prediction
→ Evaluation
→ Model Saving

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Project Structure

```text
002-customer-churn-prediction/
│
├── data/
│   └── customers.csv
│
├── models/
│   ├── customer_churn_model.pkl
│   └── confusion_matrix.png
│
├── src/
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt