#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# rf_train_eval_all_cloud.py — Random Forest (GCS + Spark Dataproc compatible, results under /results/rf)

import os, sys, argparse, subprocess
from datetime import datetime
from glob import glob

# --- Cloud/Local path auto-selection ---
GCS_BUCKET = os.getenv("GCS_BUCKET", "dsa5208-mllib-weather-predict")
ON_DATAPROC = bool(os.getenv("DATAPROC_BATCH_ID"))
USE_GCS = ON_DATAPROC or os.getenv("USE_GCS", "1") == "1"

if USE_GCS:
    ROOT = f"gs://{GCS_BUCKET}"
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Paths ---
DEFAULT_RESULTS = f"{ROOT}/results/rf"
DEFAULT_MODELS  = f"{ROOT}/models_rf"

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate Random Forest (GCS compatible).")
    p.add_argument("--train", required=True, help="Path to TRAIN parquet (gs:// or local)")
    p.add_argument("--test",  required=True, help="Path to TEST parquet (gs:// or local)")
    p.add_argument("--outdir", default=DEFAULT_RESULTS, help="Output directory root (default=results/rf)")
    p.add_argument("--models", default=DEFAULT_MODELS, help="Directory to save models")
    p.add_argument("--numTrees", type=int, default=120)
    p.add_argument("--maxDepth", type=int, default=12)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--partitions", type=int, default=300)
    p.add_argument("--driver_mem", default="6g")
    p.add_argument("--make_figs", action="store_true", help="Generate and store PNG plots under results/rf/plots/")
    return p.parse_args()

# ---------------------------
# Spark setup
# ---------------------------
def init_spark(app_name: str, driver_mem: str, partitions: int):
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.default.parallelism", str(partitions))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", 134217728)
        .config("spark.broadcast.blockSize", "4m")
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.retainedJobs", "50")
        .getOrCreate()
    )

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path: str):
    if not path.startswith("gs://"):
        os.makedirs(path, exist_ok=True)

def write_single_csv(df, folder: str):
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(folder)

def _gcs_upload(local_path: str, gcs_path: str):
    if not gcs_path.startswith("gs://"):
        return
    if gcs_path.endswith("/"):
        fname = os.path.basename(local_path)
        gcs_path = gcs_path + fname
    subprocess.run(["gsutil", "cp", local_path, gcs_path], check=False)
    print(f"[UPLOAD] {local_path} → {gcs_path}")

def save_plot(fig, outdir_plots: str, filename: str):
    import matplotlib.pyplot as plt
    fname = filename if filename.endswith(".png") else f"{filename}.png"
    if outdir_plots.startswith("gs://"):
        tmp_path = f"/tmp/{fname}"
        fig.savefig(tmp_path, dpi=200, bbox_inches="tight")
        _gcs_upload(tmp_path, outdir_plots + "/" if not outdir_plots.endswith("/") else outdir_plots)
    else:
        os.makedirs(outdir_plots, exist_ok=True)
        local_path = os.path.join(outdir_plots, fname)
        fig.savefig(local_path, dpi=200, bbox_inches="tight")
        print(f"[PLOT] {local_path}")
    plt.close(fig)

# ---------------------------
# Plot generator
# ---------------------------
def generate_plots(pred_sdf, fi_sdf, outdir_plots, label_col="TMP_C", sample_size=5000):
    import matplotlib.pyplot as plt
    from pyspark.sql import functions as F

    # --- Feature Importance ---
    fi_pdf = fi_sdf.orderBy(F.desc("importance")).toPandas()
    fig = plt.figure(figsize=(8,5))
    plt.barh(fi_pdf["feature"], fi_pdf["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    save_plot(fig, outdir_plots, "rf_feature_importance")

    # --- Residuals ---
    pred_sdf = pred_sdf.withColumn("residual", F.col("prediction") - F.col(label_col))

    # --- Predicted vs Actual ---
    pdf = pred_sdf.select(label_col, "prediction").orderBy(F.rand()).limit(sample_size).toPandas()
    fig = plt.figure(figsize=(5,5))
    plt.scatter(pdf[label_col], pdf["prediction"], alpha=0.3)
    lo, hi = pdf[label_col].min(), pdf[label_col].max()
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel("Actual (°C)")
    plt.ylabel("Predicted (°C)")
    plt.title("Predicted vs Actual")
    save_plot(fig, outdir_plots, "rf_pred_vs_actual")

    # --- Residual Histogram ---
    pdf_res = pred_sdf.select("residual").orderBy(F.rand()).limit(sample_size).toPandas()
    fig = plt.figure(figsize=(6,4))
    plt.hist(pdf_res["residual"], bins=40)
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    save_plot(fig, outdir_plots, "rf_residual_hist")

    # --- RMSE by Hour ---
    rmse_hour = (
        pred_sdf.withColumn("sq_err", F.pow(F.col("residual"), 2))
        .groupBy("hour")
        .agg(F.sqrt(F.avg("sq_err")).alias("rmse"))
        .orderBy("hour")
        .toPandas()
    )
    fig = plt.figure(figsize=(6,4))
    plt.bar(rmse_hour["hour"], rmse_hour["rmse"])
    plt.xlabel("Hour of Day")
    plt.ylabel("RMSE (°C)")
    plt.title("RMSE by Hour")
    save_plot(fig, outdir_plots, "rf_rmse_by_hour")

    # --- RMSE by Month ---
    rmse_month = (
        pred_sdf.withColumn("sq_err", F.pow(F.col("residual"), 2))
        .groupBy("month")
        .agg(F.sqrt(F.avg("sq_err")).alias("rmse"))
        .orderBy("month")
        .toPandas()
    )
    fig = plt.figure(figsize=(6,4))
    plt.bar(rmse_month["month"], rmse_month["rmse"])
    plt.xlabel("Month")
    plt.ylabel("RMSE (°C)")
    plt.title("RMSE by Month")
    save_plot(fig, outdir_plots, "rf_rmse_by_month")

    print("[PLOT] All plots saved to:", outdir_plots)

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    print(f"📦 Train = {args.train}")
    print(f"📦 Test  = {args.test}")
    print(f"📂 Results root = {args.outdir}")

    ensure_dir(args.outdir)
    ensure_dir(args.models)

    from pyspark.sql import functions as F
    spark = init_spark("RF_Weather_GCS", args.driver_mem, args.partitions)

    label_col = "TMP_C"
    feature_cols = [
        "LATITUDE","LONGITUDE","ELEVATION",
        "WND_DIR_DEG","WND_SPD_MS",
        "CIG_HEIGHT_M","VIS_DIST_M",
        "DEW_C","SLP_hPa",
        "year","month","day","hour"
    ]
    imp_cols = [c + "_imp" for c in feature_cols]

    # --- Load data ---
    train_df = spark.read.parquet(args.train).select(label_col, *feature_cols).repartition(args.partitions).persist()
    test_df  = spark.read.parquet(args.test).select("DATE", label_col, *feature_cols).repartition(args.partitions)

    _ = train_df.count()

    # --- Impute + assemble ---
    from pyspark.ml.feature import Imputer, VectorAssembler
    imputer = Imputer(strategy="median", inputCols=feature_cols, outputCols=imp_cols)
    imputerModel = imputer.fit(train_df)

    assembler = VectorAssembler(inputCols=imp_cols, outputCol="features")
    train_feat = assembler.transform(imputerModel.transform(train_df)).select(label_col, "features").persist()
    test_feat  = assembler.transform(imputerModel.transform(test_df)).select("DATE","year","month","day","hour",label_col,"features")

    # --- Train model ---
    from pyspark.ml.regression import RandomForestRegressor
    rf = RandomForestRegressor(featuresCol="features", labelCol=label_col,
                               numTrees=args.numTrees, maxDepth=args.maxDepth,
                               subsamplingRate=args.subsample, seed=42)
    rf_model = rf.fit(train_feat)

    # --- Predict ---
    preds = rf_model.transform(test_feat).persist()
    _ = preds.count()

    # --- Evaluate ---
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")
    rmse = float(evaluator.setMetricName("rmse").evaluate(preds))
    mae  = float(evaluator.setMetricName("mae").evaluate(preds))
    r2   = float(evaluator.setMetricName("r2").evaluate(preds))
    print(f"🌡️ RF Results → RMSE={rmse:.3f} | MAE={mae:.3f} | R²={r2:.3f}")

    # --- Save Models ---
    rf_model.write().overwrite().save(os.path.join(args.models, "rf_model"))
    imputerModel.write().overwrite().save(os.path.join(args.models, "rf_imputerModel"))

    # --- Create subfolders ---
    pred_dir  = os.path.join(args.outdir, "predictions_csv")
    metric_dir = os.path.join(args.outdir, "metrics_csv")
    fi_dir = os.path.join(args.outdir, "feature_importance_csv")
    plots_dir = os.path.join(args.outdir, "plots")

    # --- Write outputs ---
    write_single_csv(preds.select("DATE","year","month","day","hour",label_col,"prediction"), pred_dir)
    write_single_csv(spark.createDataFrame([(rmse, mae, r2)], ["rmse","mae","r2"]), metric_dir)

    imp = rf_model.featureImportances
    fi_rows = [(feature_cols[i], float(imp[i])) for i in range(len(feature_cols))]
    fi_sdf = spark.createDataFrame(fi_rows, ["feature","importance"]).orderBy(F.desc("importance"))
    write_single_csv(fi_sdf, fi_dir)

    # --- Generate plots ---
    if args.make_figs:
        print(f"[PLOT] Generating plots under {plots_dir}")
        generate_plots(preds, fi_sdf, plots_dir, label_col)

    print("✅ All results stored under:", args.outdir)
    spark.stop()

if __name__ == "__main__":
    sys.exit(main())