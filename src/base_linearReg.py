from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import os, json, datetime, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths (project-relative) ---
ROOT = os.path.dirname(os.path.dirname(__file__))         # repo root
data_dir = os.path.join(ROOT, "data")
results_dir = os.path.join(ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

# IMPORTANT: point to the PARQUET DIRECTORY (not an inner file)
train_path = os.path.join(data_dir, "train_2024.parquet")
test_path  = os.path.join(data_dir, "test_2024.parquet")

# --- Spark & load ---
spark = SparkSession.builder.appName("Baseline_LinearRegression").getOrCreate()
train_df = spark.read.parquet(train_path)
test_df  = spark.read.parquet(test_path)

print(f"Train rows: {train_df.count():,}")
print(f"Test rows : {test_df.count():,}")
train_df.printSchema()

# --- Features ---
feature_cols = [
    "DEW_C","SLP_hPa","WND_SPD_MS",
    "VIS_DIST_M","CIG_HEIGHT_M",
    "LATITUDE","LONGITUDE","ELEVATION"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_vec = assembler.transform(train_df).select("features", "TMP_C")
test_vec  = assembler.transform(test_df).select("features", "TMP_C")
print("Features assembled.")

# --- Model ---
lr = LinearRegression(featuresCol="features", labelCol="TMP_C", predictionCol="prediction", maxIter=100)
lr_model = lr.fit(train_vec)
predictions = lr_model.transform(test_vec)

# --- Metrics ---
evaluator = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction")
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae  = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2   = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

metrics = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "Baseline Linear Regression",
    "rmse": round(rmse, 4),
    "mae": round(mae, 4),
    "r2": round(r2, 4),
    "intercept": float(lr_model.intercept),
}
coefs = {name: float(coef) for name, coef in zip(feature_cols, lr_model.coefficients)}
results = {"metrics": metrics, "coefficients": coefs}

# Save metrics (JSON + CSV)
with open(os.path.join(results_dir, "base_linear_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)
pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "base_linear_metrics.csv"), index=False)
pd.DataFrame(list(coefs.items()), columns=["feature","coefficient"]).to_csv(
    os.path.join(results_dir, "base_linear_coefficients.csv"), index=False)

# --- Plots ---
pdf = predictions.select("TMP_C","prediction").sample(False, 0.002, seed=42).toPandas()
pdf["residual"] = pdf["TMP_C"] - pdf["prediction"]

# Actual vs Predicted
plt.figure(figsize=(5,5))
plt.scatter(pdf["TMP_C"], pdf["prediction"], alpha=0.4, s=10)
plt.plot([-80, 60], [-80, 60], "r--")
plt.xlabel("Actual TMP_C"); plt.ylabel("Predicted TMP_C")
plt.title("Actual vs Predicted Temperature")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "base_linear_reg_actual_vs_pred.png"), dpi=150)
plt.close()

# Residual histogram
plt.figure(figsize=(6,3))
plt.hist(pdf["residual"], bins=40, color="skyblue", edgecolor="black")
plt.xlabel("Residual (Actual - Predicted)"); plt.ylabel("Count")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "base_linear_reg_residual_dist.png"), dpi=150)
plt.close()

# QQ plot (optional if SciPy available)
try:
    import scipy.stats as stats
    plt.figure(figsize=(5,5))
    stats.probplot(pdf["residual"], dist="norm", plot=plt)
    plt.title("Residual QQ Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "base_linear_reg_residual_qq.png"), dpi=150)
    plt.close()
except Exception as e:
    print("Skipping QQ plot:", e)

print("Saved results to:", os.path.abspath(results_dir))