# 008 - Loan Approval Prediction

## Project Overview

This project uses Machine Learning to predict loan approval based on applicant information. It compares Logistic Regression and Decision Tree models after preprocessing numerical and categorical features.

## Dataset

The dataset was obtained from Kaggle.

Dataset link:

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

## Machine Learning Algorithms

* Logistic Regression
* Decision Tree Classifier

## Workflow

1. Load the Kaggle dataset.
2. Inspect the data and missing values.
3. Clean and standardize column names.
4. Identify the target column.
5. Handle missing numerical values using median imputation.
6. Handle missing categorical values using most-frequent imputation.
7. Convert categorical features using one-hot encoding.
8. Split the dataset into training and testing sets.
9. Train two classification models.
10. Compare model accuracy.
11. Generate a confusion matrix.
12. Save the selected model.

## Important Concepts Learned

* Data preprocessing
* Missing-value imputation
* Categorical encoding
* Logistic Regression
* Decision Trees
* Model comparison
* Classification metrics
* Confusion matrices
* Machine learning pipelines

## Disclaimer

This is an educational project. A real loan approval system requires additional validation, fairness analysis, regulatory compliance, privacy protections, and human oversight. Model performance on this dataset does not guarantee safe or appropriate real-world lending decisions.
