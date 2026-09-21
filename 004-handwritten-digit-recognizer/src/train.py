
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "digit_recognizer.pkl"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
SAMPLE_PREDICTIONS_PATH = OUTPUT_DIR / "sample_predictions.png"


# ============================================
# LOAD DIGITS DATASET
# ============================================

digits = load_digits()

X = digits.data
y = digits.target

images = digits.images

print("\n============================================")
print("AI HANDWRITTEN DIGIT RECOGNIZER")
print("============================================")

print("\n--- Dataset Information ---")
print("Number of images:", len(X))
print("Image dimensions:", images.shape[1:])
print("Number of features per image:", X.shape[1])
print("Available digit classes:", np.unique(y))

print("\n--- Pixel Data Example ---")
print(X[0])

print("\n--- First Image Label ---")
print("Actual digit:", y[0])


# ============================================
# NORMALIZE PIXEL VALUES
# ============================================

# Original pixel values range from 0 to 16.
# Dividing by 16 scales them approximately
# between 0 and 1.

X = X / 16.0


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Dataset Split ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# BUILD NEURAL NETWORK PIPELINE
# ============================================

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "neural_network",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42
            )
        )
    ]
)


# ============================================
# TRAIN THE MODEL
# ============================================

print("\n--- Training Neural Network ---")

model.fit(X_train, y_train)

print("Neural network training completed.")
print(
    "Training iterations:",
    model.named_steps["neural_network"].n_iter_
)


# ============================================
# MAKE PREDICTIONS
# ============================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

print("\n--- Sample Predictions ---")

for index in range(min(15, len(X_test))):
    predicted_digit = predictions[index]
    confidence = np.max(probabilities[index])

    print(
        f"Sample {index + 1}: "
        f"Predicted = {predicted_digit}, "
        f"Confidence = {confidence:.2%}"
    )


# ============================================
# MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy:.2%}")

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, predictions)

print("\n--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("Handwritten Digit Recognition Confusion Matrix")

plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close()

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)


# ============================================
# VISUALIZE SAMPLE PREDICTIONS
# ============================================

sample_count = 12

fig, axes = plt.subplots(
    3,
    4,
    figsize=(10, 8)
)

axes = axes.ravel()

# Recover the original pixel scale for display.
# The test images are selected using their
# original indices to display correctly.

_, test_indices = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=42,
    stratify=y
)

for index in range(sample_count):
    image_index = test_indices[index]

    image = images[image_index]
    actual_digit = y[image_index]

    # X_test is normalized and follows the same
    # ordering as test_indices.
    predicted_digit = predictions[index]

    axes[index].imshow(image, cmap="gray")
    axes[index].set_title(
        f"Actual: {actual_digit} | Predicted: {predicted_digit}"
    )
    axes[index].axis("off")

plt.tight_layout()
plt.savefig(SAMPLE_PREDICTIONS_PATH, dpi=150)
plt.close()

print("\nSample predictions saved to:")
print(SAMPLE_PREDICTIONS_PATH)


# ============================================
# CUSTOM IMAGE-LIKE PREDICTION
# ============================================

sample_index = 0

sample_image = images[sample_index]
sample_features = (sample_image.reshape(1, -1)) / 16.0

custom_prediction = model.predict(sample_features)[0]
custom_probability = np.max(
    model.predict_proba(sample_features)[0]
)

print("\n--- Single Digit Prediction ---")
print("Actual digit:", y[sample_index])
print("Predicted digit:", custom_prediction)
print(f"Confidence: {custom_probability:.2%}")


# ============================================
# SAVE TRAINED MODEL
# ============================================

joblib.dump(model, MODEL_PATH)

print("\n--- Model Saved ---")
print(MODEL_PATH)


# ============================================
# COMPLETION
# ============================================

print("\n============================================")
print("PROJECT COMPLETED")
print("============================================")