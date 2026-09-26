from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n============================================")
print("HOUSE PRICE PREDICTION - XGBOOST")
print("============================================")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Place Kaggle train.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
missing = df.isnull().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
    .head(20)
)


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "SalePrice"

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} column was not found."
    )

# SalePrice is highly right-skewed.
# We train on log1p(SalePrice).

y = np.log1p(df[TARGET])

X = df.drop(columns=[TARGET])


# ============================================================
# 4. REMOVE IDENTIFIER
# ============================================================

if "Id" in X.columns:
    X = X.drop(columns=["Id"])


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\n--- Numeric Features ---")
print(len(numeric_features))

print("\n--- Categorical Features ---")
print(len(categorical_features))


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- Dataset Split ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\n--- Training XGBoost ---")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 11. PREDICTION
# ============================================================

log_predictions = pipeline.predict(X_test)

predictions = np.expm1(log_predictions)
actual_prices = np.expm1(y_test)


# ============================================================
# 12. EVALUATION
# ============================================================

mae = mean_absolute_error(
    actual_prices,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predictions
    )
)

print("\n============================================")
print("MODEL EVALUATION")
print("============================================")

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")


# ============================================================
# 13. SAMPLE PREDICTIONS
# ============================================================

comparison = pd.DataFrame(
    {
        "Actual Price": actual_prices.values,
        "Predicted Price": predictions
    }
)

comparison["Difference"] = (
    comparison["Predicted Price"]
    - comparison["Actual Price"]
)

print("\n--- Sample Predictions ---")
print(comparison.head(10).round(2))


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

comparison.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ============================================================
# 15. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    actual_prices,
    predictions,
    alpha=0.6
)

minimum = min(
    actual_prices.min(),
    predictions.min()
)

maximum = max(
    actual_prices.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# 16. RESIDUAL ANALYSIS
# ============================================================

residuals = (
    actual_prices.values
    - predictions
)

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_plot.png",
    dpi=150
)

plt.close()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

trained_model = pipeline.named_steps["model"]
trained_preprocessor = pipeline.named_steps["preprocessor"]

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

importance = trained_model.feature_importances_

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(20)
)

print("\n--- Top 20 Important Features ---")
print(feature_importance.to_string(index=False))


# ============================================================
# 18. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"][::-1],
    feature_importance["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importances")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# 19. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "house_price_xgboost_model.pkl"
)

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n============================================")
print("PROJECT COMPLETED")
print("============================================")

print("\nGenerated files:")

print(
    OUTPUT_DIR
    / "predictions.csv"
)

print(
    OUTPUT_DIR
    / "actual_vs_predicted.png"
)

print(
    OUTPUT_DIR
    / "residual_plot.png"
)

print(
    OUTPUT_DIR
    / "feature_importance.png"
)

print("\nSaved model:")

print(model_path)