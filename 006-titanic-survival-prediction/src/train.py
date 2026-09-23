
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# 1. Define project paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# 2. Load dataset
# -----------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please place train.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "survival_distribution.png")
plt.close()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "survival_by_gender.png")
plt.close()


# -----------------------------
# 4. Select features and target
# -----------------------------

target = "Survived"

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

X = df[features]
y = df[target]


# -----------------------------
# 5. Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# -----------------------------
# 6. Define preprocessing
# -----------------------------

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

categorical_features = [
    "Sex",
    "Embarked",
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# -----------------------------
# 7. Create the ML pipeline
# -----------------------------

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)


# -----------------------------
# 8. Train the model
# -----------------------------

print("\nTraining model...")

model_pipeline.fit(X_train, y_train)

print("Training completed.")


# -----------------------------
# 9. Make predictions
# -----------------------------

y_pred = model_pipeline.predict(X_test)


# -----------------------------
# 10. Evaluate the model
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
    )
)


# -----------------------------
# 11. Create confusion matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()


# -----------------------------
# 12. Save the model
# -----------------------------

model_path = MODEL_DIR / "titanic_survival_model.pkl"

joblib.dump(
    model_pipeline,
    model_path,
)

print(f"\nModel saved at: {model_path}")


# -----------------------------
# 13. Test with a new passenger
# -----------------------------

new_passenger = pd.DataFrame(
    [
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8.05,
            "Embarked": "S",
        }
    ]
)

prediction = model_pipeline.predict(
    new_passenger
)[0]

probability = model_pipeline.predict_proba(
    new_passenger
)[0][1]

print("\nNew Passenger Prediction:")

if prediction == 1:
    print("Prediction: Passenger may survive")
else:
    print("Prediction: Passenger may not survive")

print(
    f"Survival probability: {probability:.2%}"
)

print("\nProject completed successfully.")