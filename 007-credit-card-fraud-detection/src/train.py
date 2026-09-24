
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------
# 1. Project paths
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------
# 2. Load dataset
# ---------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place creditcard.csv inside the data folder."
    )

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())


# ---------------------------------
# 3. Analyze class distribution
# ---------------------------------

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print(df["Class"].value_counts(normalize=True) * 100)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Class",
)

plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution.png"
)

plt.close()


# ---------------------------------
# 4. Prepare features and target
# ---------------------------------

X = df.drop(columns=["Class"])
y = df["Class"]


# ---------------------------------
# 5. Split the dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ---------------------------------
# 6. Scale numerical features
# ---------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------
# 7. Train Logistic Regression
# ---------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model.fit(
    X_train_scaled,
    y_train,
)

print("Training completed.")


# ---------------------------------
# 8. Make predictions
# ---------------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ---------------------------------
# 9. Evaluate the model
# ---------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)

roc_auc = roc_auc_score(
    y_test,
    y_probability,
)

average_precision = average_precision_score(
    y_test,
    y_probability,
)

print(f"ROC-AUC Score: {roc_auc:.4f}")
print(
    f"Average Precision Score: "
    f"{average_precision:.4f}"
)


# ---------------------------------
# 10. Confusion matrix
# ---------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Fraud Detection Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png"
)

plt.close()


# ---------------------------------
# 11. Precision-recall curve
# ---------------------------------

precision, recall, thresholds = (
    precision_recall_curve(
        y_test,
        y_probability,
    )
)

plt.figure(figsize=(8, 5))

plt.plot(
    recall,
    precision,
)

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curve.png"
)

plt.close()


# ---------------------------------
# 12. Save model and scaler
# ---------------------------------

model_bundle = {
    "model": model,
    "scaler": scaler,
    "feature_names": X.columns.tolist(),
}

model_path = (
    MODEL_DIR / "fraud_detection_model.pkl"
)

joblib.dump(
    model_bundle,
    model_path,
)

print(f"\nModel saved at: {model_path}")


# ---------------------------------
# 13. Test one transaction
# ---------------------------------

sample_transaction = X_test.iloc[[0]]

sample_transaction_scaled = (
    scaler.transform(sample_transaction)
)

sample_prediction = model.predict(
    sample_transaction_scaled
)[0]

sample_probability = model.predict_proba(
    sample_transaction_scaled
)[0][1]

print("\nSample Transaction Prediction:")

if sample_prediction == 1:
    print("Prediction: Potential fraud")
else:
    print("Prediction: Likely legitimate")

print(
    f"Fraud probability: "
    f"{sample_probability:.2%}"
)

print("\nProject completed successfully.")