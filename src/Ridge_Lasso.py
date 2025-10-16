# src/Ridge_Lasso.py
# -*- coding: utf-8 -*-

import os, json, datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no GUI popups
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# ---------------- Paths (project-relative) ----------------
ROOT = os.path.dirname(os.path.dirname(__file__))   # repo root
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_2024.parquet")  # parquet directories
TEST_PATH  = os.path.join(DATA_DIR, "test_2024.parquet")

# ---------------- Spark session ----------------
spark = SparkSession.builder.appName("Ridge_Lasso_ElasticNet").getOrCreate()
# (optional) be gentler on a laptop
spark.conf.set("spark.sql.shuffle.partitions", "200")

train_df = spark.read.parquet(TRAIN_PATH)
test_df  = spark.read.parquet(TEST_PATH)

print(f"Train rows: {train_df.count():,}")
print(f"Test rows : {test_df.count():,}")

# ---------------- Features ----------------
feature_cols = [
    "DEW_C","SLP_hPa","WND_SPD_MS",
    "VIS_DIST_M","CIG_HEIGHT_M",
    "LATITUDE","LONGITUDE","ELEVATION"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                           withStd=True, withMean=True)

# ---------------- Evaluators ----------------
rmse_eval = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction", metricName="rmse")
mae_eval  = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction", metricName="mae")
r2_eval   = RegressionEvaluator(labelCol="TMP_C", predictionCol="prediction", metricName="r2")

# ---------------- Helper: safe file stub for alpha ----------------
def alpha_stub(alpha: float) -> str:
    # e.g., 0.25 -> 'a0p25', 1.0 -> 'a1'
    s = f"{alpha:.2f}".rstrip("0").rstrip(".")  # '0.25' -> '0.25', '1.00'->'1'
    return "a" + s.replace(".", "p")

# ---------------- Core routine ----------------
def fit_regularized(name: str, alpha: float, reg_grid: list):
    """
    name: label base ('ridge','lasso','enet')
    alpha: elasticNetParam ∈ [0,1]   (0=ridge, 1=lasso)
    reg_grid: list of regParam (λ) to try
    """
    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol="TMP_C",
        predictionCol="prediction",
        elasticNetParam=float(alpha),
        maxIter=200,
        standardization=False  # already standardized by StandardScaler
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    param_grid = (ParamGridBuilder()
                  .addGrid(lr.regParam, reg_grid)
                  .build())

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=rmse_eval,   # minimize RMSE
        trainRatio=0.8,
        seed=42
    )

    tvs_model  = tvs.fit(train_df)
    best_pipe  = tvs_model.bestModel               # PipelineModel
    best_lr: LinearRegressionModel = best_pipe.stages[-1]  # last stage is the fitted LR model

    # Evaluate on held-out TEST
    preds = best_pipe.transform(test_df).select("TMP_C","prediction")
    rmse = rmse_eval.evaluate(preds)
    mae  = mae_eval.evaluate(preds)
    r2   = r2_eval.evaluate(preds)

    # Sample for plots
    sample_pdf = preds.sample(False, 0.002, seed=42).toPandas()
    sample_pdf["residual"] = sample_pdf["TMP_C"] - sample_pdf["prediction"]

    # Metrics & Coefs
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base  = f"{name}_{alpha_stub(alpha)}"  # e.g., ridge_a0, lasso_a1, enet_a0p50

    # regParam value from model
    try:
        best_lambda = float(best_lr._java_obj.getRegParam())
    except Exception:
        best_lambda = None

    metrics = {
        "timestamp": stamp,
        "model": f"{name.capitalize()} (alpha={alpha})",
        "elasticNetParam": float(alpha),
        "best_regParam": best_lambda,
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "r2":   round(r2, 4),
        "intercept": float(best_lr.intercept)
    }

    coefs_arr = best_lr.coefficients.toArray()
    coef_map  = {feat: float(val) for feat, val in zip(feature_cols, coefs_arr)}

    # coefficient diagnostics (on standardized scale)
    nz = int(np.count_nonzero(coefs_arr))
    l1 = float(np.abs(coefs_arr).sum())
    l2 = float(np.linalg.norm(coefs_arr, 2))
    coef_diag = {"nonzero": nz, "l1_norm": l1, "l2_norm": l2}

    # --- Save artifacts ---
    with open(os.path.join(RESULTS_DIR, f"{base}_metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "coefficients": coef_map, "coef_diag": coef_diag}, f, indent=4)

    pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, f"{base}_metrics.csv"), index=False)
    pd.DataFrame(list(coef_map.items()), columns=["feature","coefficient"]).to_csv(
        os.path.join(RESULTS_DIR, f"{base}_coefficients.csv"), index=False)
    pd.DataFrame([coef_diag]).to_csv(os.path.join(RESULTS_DIR, f"{base}_coef_diagnostics.csv"), index=False)

    # Plots
    # Actual vs Pred
    plt.figure(figsize=(5,5))
    plt.scatter(sample_pdf["TMP_C"], sample_pdf["prediction"], alpha=0.4, s=10)
    plt.plot([-80, 60], [-80, 60], "r--")
    plt.xlabel("Actual TMP_C"); plt.ylabel("Predicted TMP_C")
    plt.title(f"Actual vs Predicted ({name.capitalize()}, α={alpha})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{base}_actual_vs_pred.png"), dpi=150); plt.close()

    # Residual histogram
    plt.figure(figsize=(6,3))
    plt.hist(sample_pdf["residual"], bins=40, color="skyblue", edgecolor="black")
    plt.xlabel("Residual (Actual - Predicted)"); plt.ylabel("Count")
    plt.title(f"Residual Distribution ({name.capitalize()}, α={alpha})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{base}_residual_dist.png"), dpi=150); plt.close()

    # QQ
    try:
        import scipy.stats as stats
        plt.figure(figsize=(5,5))
        stats.probplot(sample_pdf["residual"], dist="norm", plot=plt)
        plt.title(f"Residual QQ Plot ({name.capitalize()}, α={alpha})")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{base}_residual_qq.png"), dpi=150); plt.close()
    except Exception as e:
        print(f"Skipping QQ plot for {base}: {e}")

    print(f"[{name.upper()} α={alpha}] best λ={best_lambda} | RMSE={metrics['rmse']:.3f}  "
          f"MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}  nonzero={nz}")

    return metrics

# ---------------- Grids ----------------
# Wider lambda (regParam) grid to expose shrinkage effects
RIDGE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
LASSO_GRID = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# Elastic Net alphas to try (includes Ridge/Lasso extremes for comparison)
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------------- Run all models ----------------
summary_rows = []
for alpha in ALPHAS:
    if alpha == 0.0:
        m = fit_regularized("ridge", alpha=alpha, reg_grid=RIDGE_GRID)
    elif alpha == 1.0:
        m = fit_regularized("lasso", alpha=alpha, reg_grid=LASSO_GRID)
    else:
        m = fit_regularized("enet",  alpha=alpha, reg_grid=LASSO_GRID)  # use the lasso grid; ENet will blend L1/L2
    summary_rows.append(m)

# Combined summary for the report
pd.DataFrame(summary_rows)[
    ["model","elasticNetParam","best_regParam","rmse","mae","r2","intercept"]
].to_csv(os.path.join(RESULTS_DIR, "regularized_summary.csv"), index=False)

print("Saved all Phase 2 artifacts to:", os.path.abspath(RESULTS_DIR))