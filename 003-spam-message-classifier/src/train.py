
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "messages.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"


# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv(DATA_PATH)

print("\n============================================")
print("AI SPAM MESSAGE CLASSIFIER")
print("============================================")

print("\n--- Dataset Preview ---")
print(data.head())

print("\n--- Dataset Information ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Class Distribution ---")
print(data["label"].value_counts())


# ============================================
# CONVERT LABELS TO NUMBERS
# ============================================

# ham  = 0
# spam = 1

data["target"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

if data["target"].isnull().any():
    raise ValueError("Dataset contains an invalid label.")

X = data["message"]
y = data["target"]


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n--- Dataset Split ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# BUILD AI PIPELINE
# ============================================

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================
# MODEL TRAINING
# ============================================

model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")


# ============================================
# PREDICTIONS
# ============================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- Test Predictions ---")

for message, actual, predicted, probability in zip(
    X_test,
    y_test,
    predictions,
    probabilities
):
    actual_label = "SPAM" if actual == 1 else "HAM"
    predicted_label = "SPAM" if predicted == 1 else "HAM"

    print("\nMessage:", message)
    print("Actual:", actual_label)
    print("Predicted:", predicted_label)
    print(f"Spam Probability: {probability:.2%}")


# ============================================
# MODEL EVALUATION
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
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)


# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, predictions)

print("\n--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Spam Classifier Confusion Matrix")

plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)


# ============================================
# TEST CUSTOM MESSAGES
# ============================================

custom_messages = [
    "Congratulations you won a free cash prize click now",
    "Hey, please send me the assignment when you finish",
    "Claim your exclusive reward immediately",
    "Are we meeting at the library tomorrow?"
]

print("\n--- Custom Message Predictions ---")

for message in custom_messages:
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0][1]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print("\nMessage:", message)
    print("Prediction:", result)
    print(f"Spam Probability: {probability:.2%}")


# ============================================
# SAVE MODEL
# ============================================

joblib.dump(model, MODEL_PATH)

print("\n--- Model Saved ---")
print(MODEL_PATH)

print("\n============================================")
print("PROJECT COMPLETED")
print("============================================")