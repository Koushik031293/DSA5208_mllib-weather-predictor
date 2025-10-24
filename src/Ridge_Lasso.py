#!/usr/bin/env python3
# src/Ridge_Lasso.py
# Cloud-friendly version (GCS write-safe)

import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# ==========================================================
# === I/O helpers (local or GCS) ===========================
# ==========================================================
def is_gcs_path(p: str) -> bool:
    return str(p).startswith("gs://")

def split_gs(uri: str):
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
        b, k = split_gs(path)
        gcs_client().bucket(b).blob(k).upload_from_string(text, content_type=content_type)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f: f.write(text)

def save_bytes(path: str, data: bytes, content_type="application/octet-stream"):
    if is_gcs_path(path):
        b, k = split_gs(path)
        gcs_client().bucket(b).blob(k).upload_from_string(data, content_type=content_type)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: f.write(data)

def save_csv(path: str, df: pd.DataFrame):
    csv = df.to_csv(index=False)
    save_text(path, csv, content_type="text/csv")

def save_plot(path: str, dpi=150):
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    save_bytes(path, buf.getvalue(), content_type="image/png")

# ==========================================================
# === Paths (auto cloud/local) =============================
# ==========================================================
GCS_BUCKET = os.getenv("GCS_BUCKET", "dsa5208-mllib-weather-predict")
ON_DATAPROC = bool(os.getenv("DATAPROC_BATCH_ID"))
USE_GCS = ON_DATAPROC or os.getenv("USE_GCS", "1") == "1"

if USE_GCS:
    ROOT = f"gs://{GCS_BUCKET}"
else:
    here = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(here)

DATA_DIR    = f"{ROOT}/data"
RESULTS_DIR = f"{ROOT}/results"
TRAIN_PATH  = f"{DATA_DIR}/train_2024.parquet"
TEST_PATH   = f"{DATA_DIR}/test_2024.parquet"

print("🔗 Using ROOT:", ROOT)
print("📦 Train:", TRAIN_PATH)
print("📦 Test :", TEST_PATH)
print("📂 Results:", RESULTS_DIR)

# ==========================================================
# === Spark session ========================================
# ==========================================================
spark = SparkSession.builder.appName("Ridge_Lasso_ElasticNet").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "200")

train_df = spark.read.parquet(TRAIN_PATH)
test_df  = spark.read.parquet(TEST_PATH)
print(f"Train rows: {train_df.count():,}")
print(f"Test rows : {test_df.count():,}")

# ==========================================================
# === Features / evaluators ================================
# ==========================================================
feature_cols = [
    "DEW_C","SLP_hPa","WND_SPD_MS",
    "VIS_DIST_M","CIG_HEIGHT_M",
    "LATITUDE","LONGITUDE","ELEVATION"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                           withStd=True, withMean=True)

rmse_eval = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction", metricName="rmse")
mae_eval  = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction", metricName="mae")
r2_eval   = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction", metricName="r2")

def alpha_stub(alpha: float) -> str:
    s = f"{alpha:.2f}".rstrip("0").rstrip(".")
    return "a" + s.replace(".", "p")

# ==========================================================
# === Core routine =========================================
# ==========================================================
def fit_regularized(name: str, alpha: float, reg_grid: list):
    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol="TMP_C",
        predictionCol="prediction",
        elasticNetParam=float(alpha),
        maxIter=200,
        standardization=False
    )
    pipe = Pipeline(stages=[assembler, scaler, lr])
    grid = (ParamGridBuilder().addGrid(lr.regParam, reg_grid).build())

    tvs = TrainValidationSplit(estimator=pipe, estimatorParamMaps=grid,
                               evaluator=rmse_eval, trainRatio=0.8, seed=42)
    tvs_model = tvs.fit(train_df)
    best_pipe = tvs_model.bestModel
    best_lr: LinearRegressionModel = best_pipe.stages[-1]

    preds = best_pipe.transform(test_df).select("TMP_C","prediction")
    rmse = rmse_eval.evaluate(preds)
    mae  = mae_eval.evaluate(preds)
    r2   = r2_eval.evaluate(preds)

    pdf = preds.sample(False, 0.002, seed=42).toPandas()
    pdf["residual"] = pdf["TMP_C"] - pdf["prediction"]

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base  = f"{name}_{alpha_stub(alpha)}"

    try:
        best_lambda = float(best_lr._java_obj.getRegParam())
    except Exception:
        best_lambda = None

    metrics = {
        "timestamp": stamp,
        "model": f"{name.capitalize()} (alpha={alpha})",
        "elasticNetParam": alpha,
        "best_regParam": best_lambda,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "intercept": float(best_lr.intercept)
    }

    coefs_arr = best_lr.coefficients.toArray()
    coef_map  = {f: float(v) for f, v in zip(feature_cols, coefs_arr)}
    coef_diag = {
        "nonzero": int(np.count_nonzero(coefs_arr)),
        "l1_norm": float(np.abs(coefs_arr).sum()),
        "l2_norm": float(np.linalg.norm(coefs_arr, 2))
    }

    # --- Save metrics ---
    save_text(f"{RESULTS_DIR}/{base}_metrics.json",
              json.dumps({"metrics": metrics, "coefficients": coef_map,
                          "coef_diag": coef_diag}, indent=4),
              content_type="application/json")
    save_csv(f"{RESULTS_DIR}/{base}_metrics.csv", pd.DataFrame([metrics]))
    save_csv(f"{RESULTS_DIR}/{base}_coefficients.csv",
             pd.DataFrame(list(coef_map.items()), columns=["feature","coefficient"]))
    save_csv(f"{RESULTS_DIR}/{base}_coef_diagnostics.csv", pd.DataFrame([coef_diag]))

    # --- Plots ---
    plt.figure(figsize=(5,5))
    plt.scatter(pdf["TMP_C"], pdf["prediction"], alpha=0.4, s=10)
    plt.plot([-80,60],[-80,60],"r--")
    plt.xlabel("Actual TMP_C"); plt.ylabel("Predicted TMP_C")
    plt.title(f"Actual vs Predicted ({name.capitalize()}, α={alpha})")
    save_plot(f"{RESULTS_DIR}/{base}_actual_vs_pred.png")

    plt.figure(figsize=(6,3))
    plt.hist(pdf["residual"], bins=40, edgecolor="black")
    plt.xlabel("Residual (Actual - Predicted)"); plt.ylabel("Count")
    plt.title(f"Residual Distribution ({name.capitalize()}, α={alpha})")
    save_plot(f"{RESULTS_DIR}/{base}_residual_dist.png")

    try:
        import scipy.stats as stats
        plt.figure(figsize=(5,5))
        stats.probplot(pdf["residual"], dist="norm", plot=plt)
        plt.title(f"Residual QQ Plot ({name.capitalize()}, α={alpha})")
        save_plot(f"{RESULTS_DIR}/{base}_residual_qq.png")
    except Exception as e:
        print(f"Skipping QQ plot for {base}: {e}")

    print(f"[{name.upper()} α={alpha}] λ={best_lambda} | RMSE={metrics['rmse']:.3f} "
          f"MAE={metrics['mae']:.3f} R²={metrics['r2']:.3f} nz={coef_diag['nonzero']}")
    return metrics

# ==========================================================
# === Grid setup & run =====================================
# ==========================================================
RIDGE_GRID = [1e-6,1e-5,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0]
LASSO_GRID = [1e-5,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1]
ALPHAS = [0.0,0.25,0.5,0.75,1.0]

summary = []
for a in ALPHAS:
    if a == 0.0:
        m = fit_regularized("ridge", a, RIDGE_GRID)
    elif a == 1.0:
        m = fit_regularized("lasso", a, LASSO_GRID)
    else:
        m = fit_regularized("enet", a, LASSO_GRID)
    summary.append(m)

save_csv(f"{RESULTS_DIR}/regularized_summary.csv",
         pd.DataFrame(summary)[["model","elasticNetParam","best_regParam","rmse","mae","r2","intercept"]])
print("✅ Saved all artifacts to:", RESULTS_DIR)