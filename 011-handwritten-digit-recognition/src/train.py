import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. PROJECT SETUP
# ============================================================

print("=" * 60)
print("HANDWRITTEN DIGIT RECOGNITION")
print("=" * 60)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("\nTensorFlow version:", tf.__version__)


# ============================================================
# 2. LOAD MNIST DATASET
# ============================================================

print("\n--- Loading MNIST Dataset ---")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)


# ============================================================
# 3. NORMALIZE IMAGES
# ============================================================

# Original pixel values:
# 0 to 255
#
# Neural networks work better with smaller values.
#
# Convert:
# 0-255 → 0-1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ============================================================
# 4. DISPLAY SAMPLE IMAGES
# ============================================================

plt.figure(figsize=(8, 8))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/sample_digits.png",
    dpi=150
)

plt.close()


# ============================================================
# 5. BUILD NEURAL NETWORK
# ============================================================

print("\n--- Building Neural Network ---")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(28, 28)
        ),

        # Convert 28x28 image into one vector
        tf.keras.layers.Flatten(),

        # Hidden layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Dropout helps reduce overfitting
        tf.keras.layers.Dropout(0.2),

        # Second hidden layer
        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        # Output layer
        #
        # 10 neurons = digits 0-9
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ]
)


# ============================================================
# 6. DISPLAY MODEL STRUCTURE
# ============================================================

model.summary()


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n--- Training Neural Network ---")

history = model.fit(
    x_train,
    y_train,

    epochs=10,

    batch_size=128,

    validation_split=0.1,

    verbose=1
)


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

print("\n--- Evaluating Model ---")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

print("\n--- Making Predictions ---")

probabilities = model.predict(
    x_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 11. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n--- Sample Predictions ---")

for i in range(20):

    print(
        f"Image {i + 1}: "
        f"Actual = {y_test[i]}, "
        f"Predicted = {predictions[i]}"
    )


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("MNIST Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 13. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/accuracy.png",
    dpi=150
)

plt.close()


# ============================================================
# 14. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/loss.png",
    dpi=150
)

plt.close()


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

model_path = (
    "models/mnist_digit_model.keras"
)

model.save(model_path)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print("outputs/sample_digits.png")
print("outputs/confusion_matrix.png")
print("outputs/accuracy.png")
print("outputs/loss.png")

print("\nSaved model:")
print(model_path)