# ============================================
# CUSTOMER CHURN PREDICTION - DAY 2
# Machine Learning Classification Project
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("data/customers.csv")

print("\n============================================")
print("CUSTOMER CHURN PREDICTION")
print("============================================")

print("\n--- Dataset ---")
print(data)


# ============================================
# 2. DATA INFORMATION
# ============================================

print("\n--- Dataset Information ---")
data.info()

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Churn Distribution ---")
print(data["churn"].value_counts())


# ============================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================

X = data.drop("churn", axis=1)

y = data["churn"]


# ============================================
# 4. IDENTIFY COLUMN TYPES
# ============================================

categorical_features = [
    "contract_type"
]

numerical_features = [
    "age",
    "tenure",
    "monthly_charges",
    "total_charges",
    "support_calls"
]


# ============================================
# 5. PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================
# 6. CREATE MODEL PIPELINE
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# ============================================
# 7. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Dataset Split ---")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================
# 8. TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")


# ============================================
# 9. MAKE PREDICTIONS
# ============================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


print("\n--- Predictions ---")

for actual, predicted, probability in zip(
    y_test,
    predictions,
    probabilities
):
    print(
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Churn Probability: {probability:.2%}"
    )


# ============================================
# 10. MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n--- Model Evaluation ---")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================
# 11. CLASSIFICATION REPORT
# ============================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================
# 12. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n--- Confusion Matrix ---")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png"
)

plt.show()


# ============================================
# 13. PREDICT A NEW CUSTOMER
# ============================================

new_customer = pd.DataFrame({
    "age": [27],
    "tenure": [4],
    "monthly_charges": [95],
    "total_charges": [380],
    "contract_type": ["monthly"],
    "support_calls": [5]
})


new_prediction = model.predict(new_customer)

new_probability = model.predict_proba(
    new_customer
)[0][1]


print("\n--- New Customer Prediction ---")

if new_prediction[0] == 1:
    result = "Customer is likely to churn"
else:
    result = "Customer is likely to stay"


print(result)

print(
    f"Churn Probability: {new_probability:.2%}"
)


# ============================================
# 14. SAVE MODEL
# ============================================

joblib.dump(
    model,
    "models/customer_churn_model.pkl"
)

print("\n--- Model Saved ---")

print(
    "models/customer_churn_model.pkl"
)


# ============================================
# 15. COMPLETED
# ============================================

print("\n============================================")
print("PROJECT COMPLETED")
print("============================================")