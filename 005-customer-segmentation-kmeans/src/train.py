
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Mall_Customers.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "customer_segmentation_model.pkl"
SEGMENTED_DATA_PATH = OUTPUT_DIR / "segmented_customers.csv"
ELBOW_PATH = OUTPUT_DIR / "elbow_curve.png"
SILHOUETTE_PATH = OUTPUT_DIR / "silhouette_scores.png"
CLUSTER_PLOT_PATH = OUTPUT_DIR / "customer_clusters.png"


# ============================================
# LOAD DATASET
# ============================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please place Mall_Customers.csv inside the data folder."
    )

data = pd.read_csv(DATA_PATH)

print("\n============================================")
print("AI CUSTOMER SEGMENTATION USING K-MEANS")
print("============================================")

print("\n--- Dataset Preview ---")
print(data.head())

print("\n--- Dataset Shape ---")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\n--- Column Names ---")
print(list(data.columns))

print("\n--- Dataset Information ---")
data.info()

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Duplicate Rows ---")
print(data.duplicated().sum())


# ============================================
# VALIDATE REQUIRED COLUMNS
# ============================================

required_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================
# SELECT FEATURES
# ============================================

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = data[features].copy()

if X.isnull().any().any():
    raise ValueError(
        "Selected features contain missing values."
    )

print("\n--- Selected Features ---")
print(features)

print("\n--- Statistical Summary ---")
print(X.describe())


# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================
# FIND BEST NUMBER OF CLUSTERS
# ============================================

cluster_range = range(2, 11)

inertia_values = []
silhouette_values = []

print("\n--- Testing Cluster Counts ---")

for k in cluster_range:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f"Clusters: {k} | "
        f"Inertia: {inertia:.2f} | "
        f"Silhouette Score: {silhouette:.4f}"
    )


# ============================================
# SELECT BEST CLUSTER COUNT
# ============================================

best_index = int(np.argmax(silhouette_values))
best_k = list(cluster_range)[best_index]
best_silhouette = silhouette_values[best_index]

print("\n--- Best Cluster Count ---")
print("Selected clusters:", best_k)
print(f"Best silhouette score: {best_silhouette:.4f}")


# ============================================
# ELBOW CURVE
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Curve for K-Means Clustering")
plt.xticks(list(cluster_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(ELBOW_PATH, dpi=150)
plt.close()

print("\nElbow curve saved to:")
print(ELBOW_PATH)


# ============================================
# SILHOUETTE SCORE GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores by Cluster Count")
plt.xticks(list(cluster_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(SILHOUETTE_PATH, dpi=150)
plt.close()

print("\nSilhouette graph saved to:")
print(SILHOUETTE_PATH)


# ============================================
# TRAIN FINAL K-MEANS MODEL
# ============================================

final_model = KMeans(
    n_clusters=best_k,
    init="k-means++",
    n_init=10,
    random_state=42
)

cluster_labels = final_model.fit_predict(X_scaled)

data["Cluster"] = cluster_labels


# ============================================
# CLUSTER SUMMARY
# ============================================

cluster_summary = (
    data.groupby("Cluster")[features]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\n--- Cluster Summary ---")
print(cluster_summary)


# ============================================
# CLUSTER VISUALIZATION
# ============================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

centers_original_scale = scaler.inverse_transform(
    final_model.cluster_centers_
)

plt.scatter(
    centers_original_scale[:, 0],
    centers_original_scale[:, 1],
    marker="X",
    s=250,
    color="red",
    edgecolor="black",
    label="Cluster Centers"
)

plt.title(
    f"Customer Segmentation using K-Means "
    f"({best_k} Clusters)"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()

plt.savefig(CLUSTER_PLOT_PATH, dpi=150)
plt.close()

print("\nCluster visualization saved to:")
print(CLUSTER_PLOT_PATH)


# ============================================
# SAVE SEGMENTED DATA
# ============================================

data.to_csv(
    SEGMENTED_DATA_PATH,
    index=False
)

print("\nSegmented dataset saved to:")
print(SEGMENTED_DATA_PATH)


# ============================================
# SAVE MODEL
# ============================================

model_bundle = {
    "model": final_model,
    "scaler": scaler,
    "features": features,
    "n_clusters": best_k
}

joblib.dump(model_bundle, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)


# ============================================
# PREDICT A NEW CUSTOMER SEGMENT
# ============================================

new_customer = pd.DataFrame({
    "Annual Income (k$)": [75],
    "Spending Score (1-100)": [80]
})

new_customer_scaled = scaler.transform(new_customer)

new_cluster = final_model.predict(
    new_customer_scaled
)[0]

print("\n--- New Customer Prediction ---")
print(new_customer)
print("Assigned cluster:", new_cluster)


# ============================================
# COMPLETION
# ============================================

print("\n============================================")
print("PROJECT COMPLETED")
print("============================================")