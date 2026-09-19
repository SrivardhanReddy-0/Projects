# ============================================
# HOUSE PRICE PREDICTION - DAY 1
# Machine Learning Project
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("data/housing.csv")

print("\n============================================")
print("HOUSE PRICE PREDICTION")
print("============================================")

print("\nDataset:")
print(data)


# ============================================
# 2. DATA INFORMATION
# ============================================

print("\n--- Dataset Information ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Statistical Summary ---")
print(data.describe())


# ============================================
# 3. DEFINE FEATURES AND TARGET
# ============================================

X = data[[
    "area",
    "bedrooms",
    "bathrooms",
    "age"
]]

y = data["price"]

print("\n--- Features ---")
print(X.head())

print("\n--- Target ---")
print(y.head())


# ============================================
# 4. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Dataset Split ---")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================
# 5. CREATE MACHINE LEARNING MODEL
# ============================================

model = LinearRegression()


# ============================================
# 6. TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")


# ============================================
# 7. MAKE PREDICTIONS
# ============================================

predictions = model.predict(X_test)

print("\n--- Predictions ---")

for actual, predicted in zip(y_test, predictions):
    print(
        f"Actual: ₹{actual:,.0f} | "
        f"Predicted: ₹{predicted:,.0f}"
    )


# ============================================
# 8. EVALUATE MODEL
# ============================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")

print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================
# 9. MODEL COEFFICIENTS
# ============================================

print("\n--- Learned Coefficients ---")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:,.2f}")

print(f"Intercept: {model.intercept_:,.2f}")


# ============================================
# 10. PREDICT A NEW HOUSE
# ============================================

new_house = pd.DataFrame({
    "area": [2000],
    "bedrooms": [3],
    "bathrooms": [3],
    "age": [5]
})

new_prediction = model.predict(new_house)

print("\n--- New House Prediction ---")

print(
    f"House: 2000 sq.ft | 3 bedrooms | "
    f"3 bathrooms | 5 years old"
)

print(f"Predicted Price: ₹{new_prediction[0]:,.0f}")


# ============================================
# 11. SAVE MODEL
# ============================================

joblib.dump(model, "models/house_price_model.pkl")

print("\n--- Model Saved ---")
print("models/house_price_model.pkl")


# ============================================
# 12. VISUALIZE ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()

plt.savefig("models/actual_vs_predicted.png")

plt.show()

print("\n============================================")
print("PROJECT COMPLETED")
print("============================================")