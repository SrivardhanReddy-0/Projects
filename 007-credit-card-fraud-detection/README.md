# 007 - Credit Card Fraud Detection

## Project Overview

This project uses Machine Learning to identify potentially fraudulent credit card transactions. The dataset contains transaction-related features and a target column named `Class`, where `0` represents a legitimate transaction and `1` represents a fraudulent transaction.

## Dataset

The dataset was obtained from Kaggle.

Dataset link:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

## Machine Learning Algorithm

Logistic Regression with class balancing.

## Workflow

1. Load the transaction dataset.
2. Analyze the distribution of legitimate and fraudulent transactions.
3. Separate the features and target.
4. Split the data into training and testing sets.
5. Scale the features.
6. Train a balanced Logistic Regression model.
7. Evaluate the model using precision, recall, F1-score, ROC-AUC, and Average Precision.
8. Visualize the confusion matrix and precision-recall curve.
9. Save the trained model and scaler.

## Important Concepts Learned

* Binary classification
* Imbalanced datasets
* Class weighting
* Feature scaling
* Precision and recall
* F1-score
* Confusion matrix
* ROC-AUC
* Precision-recall curve
* Model persistence

## Important Note

Fraud detection is an imbalanced classification problem. Accuracy alone can be misleading because legitimate transactions greatly outnumber fraudulent transactions. In a real production system, model thresholds, false positives, false negatives, cost of fraud, and additional validation would need to be considered.

This project is for educational purposes.
