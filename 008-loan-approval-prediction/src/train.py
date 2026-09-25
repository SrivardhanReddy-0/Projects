
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------
# 1. Project paths
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "loan_sanction_train.csv"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------
# 2. Load dataset
# ---------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Place the CSV file inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ---------------------------------
# 3. Standardize column names
# ---------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nStandardized columns:")
print(df.columns.tolist())


# ---------------------------------
# 4. Detect target column
# ---------------------------------

possible_targets = [
    "Loan_Status",
    "loan_status",
    "Loan_Approved",
    "loan_approved",
    "Approved",
    "approved",
]

target = None

for column in possible_targets:
    if column in df.columns:
        target = column
        break

if target is None:
    raise ValueError(
        "Could not find the loan approval target column. "
        f"Available columns: {df.columns.tolist()}"
    )

print(f"\nTarget column: {target}")


# ---------------------------------
# 5. Prepare target
# ---------------------------------

df = df.dropna(subset=[target]).copy()

target_values = (
    df[target]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_mapping = {
    "y": 1,
    "yes": 1,
    "1": 1,
    "approved": 1,
    "true": 1,
    "n": 0,
    "no": 0,
    "0": 0,
    "rejected": 0,
    "false": 0,
}

y = target_values.map(target_mapping)

if y.isnull().any():
    raise ValueError(
        "Unexpected target values found: "
        f"{target_values.unique().tolist()}"
    )

y = y.astype(int)


# ---------------------------------
# 6. Prepare features
# ---------------------------------

drop_columns = [
    target,
    "Loan_ID",
    "loan_id",
    "LoanID",
    "loan_id_",
]

drop_columns = [
    column
    for column in drop_columns
    if column in df.columns
]

X = df.drop(columns=drop_columns)

# Remove columns that contain only missing values
X = X.dropna(axis=1, how="all")

print("\nFeature columns:")
print(X.columns.tolist())


# ---------------------------------
# 7. Identify feature types
# ---------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ---------------------------------
# 8. Create preprocessing pipelines
# ---------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
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


# ---------------------------------
# 9. Split the dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ---------------------------------
# 10. Define models
# ---------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    ),
}


# ---------------------------------
# 11. Train and evaluate models
# ---------------------------------

results = []
trained_models = {}

for model_name, classifier in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
        }
    )

    trained_models[model_name] = pipeline


# ---------------------------------
# 12. Compare models
# ---------------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy",
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png"
)

plt.close()


# ---------------------------------
# 13. Select model
# ---------------------------------

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model",
]

best_model = trained_models[best_model_name]

print(
    f"\nSelected model based on test accuracy: "
    f"{best_model_name}"
)


# ---------------------------------
# 14. Confusion matrix
# ---------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png"
)

plt.close()


# ---------------------------------
# 15. Save best model
# ---------------------------------

model_path = (
    MODEL_DIR / "loan_approval_model.pkl"
)

joblib.dump(
    best_model,
    model_path,
)

print(f"\nModel saved at: {model_path}")

print("\nProject completed successfully.")