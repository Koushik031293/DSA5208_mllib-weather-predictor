#!/usr/bin/env python3
# base_linearReg.py — cloud-friendly I/O for GCS

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import os, json, datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------
# I/O helpers (GCS or local)
# -----------------------------
def is_gcs_path(p: str) -> bool:
    return str(p).startswith("gs://")

def split_gs(uri: str):
    # returns (bucket, prefix-without-leading-slash)
    no_scheme = uri[5:]
    bucket, _, prefix = no_scheme.partition("/")
    return bucket, prefix

_gcs_client = None
def gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client

def save_text(path: str, text: str, content_type="text/plain"):
    if is_gcs_path(path):
        bkt, key = split_gs(path)
        gcs_client().bucket(bkt).blob(key).upload_from_string(text, content_type=content_type)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

def save_bytes(path: str, data: bytes, content_type="application/octet-stream"):
    if is_gcs_path(path):
        bkt, key = split_gs(path)
        gcs_client().bucket(bkt).blob(key).upload_from_string(data, content_type=content_type)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

def save_csv(path: str, df: pd.DataFrame):
    csv = df.to_csv(index=False)
    save_text(path, csv, content_type="text/csv")

def save_plt_png(path: str, dpi=150):
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    save_bytes(path, buf.getvalue(), content_type="image/png")

# -----------------------------
# Paths (auto cloud/local)
# -----------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET", "dsa5208-mllib-weather-predict")
ON_DATAPROC = bool(os.getenv("DATAPROC_BATCH_ID"))  # set on Dataproc Serverless
USE_GCS = ON_DATAPROC or os.getenv("USE_GCS", "1") == "1"  # default to GCS

if USE_GCS:
    ROOT = f"gs://{GCS_BUCKET}"
else:
    # local repo root
    here = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(here)

data_dir    = f"{ROOT}/data"
results_dir = f"{ROOT}/results"

train_path = f"{data_dir}/train_2024.parquet"
test_path  = f"{data_dir}/test_2024.parquet"

print("🔗 Using ROOT:", ROOT)
print("📦 Train:", train_path)
print("📦 Test :", test_path)
print("📂 Results dir:", results_dir)

# -----------------------------
# Spark session & load
# -----------------------------
spark = SparkSession.builder.appName("Baseline_LinearRegression").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE", "200"))

train_df = spark.read.parquet(train_path)
test_df  = spark.read.parquet(test_path)

print(f"✅ Loaded: {train_df.count():,} train rows, {len(train_df.columns)} columns")
print(f"Train rows: {train_df.count():,}")
print(f"Test rows : {test_df.count():,}")
train_df.printSchema()

# -----------------------------
# Features & model
# -----------------------------
feature_cols = [
    "DEW_C","SLP_hPa","WND_SPD_MS",
    "VIS_DIST_M","CIG_HEIGHT_M",
    "LATITUDE","LONGITUDE","ELEVATION"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_vec = assembler.transform(train_df).select("features", "TMP_C")
test_vec  = assembler.transform(test_df).select("features", "TMP_C")

lr = LinearRegression(featuresCol="features", labelCol="TMP_C", predictionCol="prediction", maxIter=100)
lr_model = lr.fit(train_vec)
predictions = lr_model.transform(test_vec)

# -----------------------------
# Metrics
# -----------------------------
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

# -----------------------------
# Save metrics (JSON + CSV) to GCS/local
# -----------------------------
save_text(f"{results_dir}/base_linear_metrics.json", json.dumps(results, indent=4), content_type="application/json")
save_csv(f"{results_dir}/base_linear_metrics.csv", pd.DataFrame([metrics]))
save_csv(f"{results_dir}/base_linear_coefficients.csv", pd.DataFrame(list(coefs.items()), columns=["feature","coefficient"]))

# -----------------------------
# Plots (sample → PNGs)
# -----------------------------
pdf = predictions.select("TMP_C","prediction").sample(False, 0.002, seed=42).toPandas()
pdf["residual"] = pdf["TMP_C"] - pdf["prediction"]

# Actual vs Predicted
plt.figure(figsize=(5,5))
plt.scatter(pdf["TMP_C"], pdf["prediction"], alpha=0.4, s=10)
plt.plot([-80, 60], [-80, 60], "r--")
plt.xlabel("Actual TMP_C"); plt.ylabel("Predicted TMP_C")
plt.title("Actual vs Predicted Temperature")
save_plt_png(f"{results_dir}/base_linear_reg_actual_vs_pred.png")

# Residual histogram
plt.figure(figsize=(6,3))
plt.hist(pdf["residual"], bins=40, edgecolor="black")
plt.xlabel("Residual (Actual - Predicted)"); plt.ylabel("Count")
plt.title("Residual Distribution")
save_plt_png(f"{results_dir}/base_linear_reg_residual_dist.png")

# QQ plot (optional)
try:
    import scipy.stats as stats
    plt.figure(figsize=(5,5))
    stats.probplot(pdf["residual"], dist="norm", plot=plt)
    plt.title("Residual QQ Plot")
    save_plt_png(f"{results_dir}/base_linear_reg_residual_qq.png")
except Exception as e:
    print("Skipping QQ plot:", e)

print("📝 Saved results under:", results_dir)