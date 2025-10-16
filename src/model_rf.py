#!/usr/bin/env python3
# rf_train_eval_all.py
# End-to-end Random Forest training, evaluation, artifacts, plots, and summary.

import os
import sys
import argparse
from glob import glob
from datetime import datetime
# --- First cell: force Java 11 inside the notebook process ---
import os, subprocess, sys

# Point JAVA_HOME to JDK 11 (Temurin 11) on macOS
os.environ["JAVA_HOME"] = subprocess.check_output(
    ["/usr/libexec/java_home", "-v", "11"], text=True
).strip()

# sanity check
print("JAVA_HOME =", os.environ["JAVA_HOME"])
print("java -version:")
print(subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True))

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate RF on weather data with Spark.")
    p.add_argument("--train", required=True, help="Path to TRAIN parquet")
    p.add_argument("--test",  required=True, help="Path to TEST parquet")
    p.add_argument("--outdir", default="results", help="Output directory for results (CSV/figures/MD)")
    p.add_argument("--models", default="models", help="Directory to save models")
    p.add_argument("--numTrees", type=int, default=80)
    p.add_argument("--maxDepth", type=int, default=10)
    p.add_argument("--subsample", type=float, default=0.7)
    p.add_argument("--partitions", type=int, default=500, help="Repartition count for local run")
    p.add_argument("--driver_mem", default="14g", help="Spark driver memory (local)")
    p.add_argument("--make_figs", action="store_true", help="Also generate PNG figures")
    p.add_argument("--make_summary", action="store_true", help="Also generate summary_rf.md")
    return p.parse_args()

# ---------------------------
# Spark setup
# ---------------------------
def init_spark(app_name: str, driver_mem: str):
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.4")
        .config("spark.sql.shuffle.partitions", "500")
        .config("spark.default.parallelism", "500")
        .getOrCreate()
    )
    return spark




# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def serialize_storage_level():
    # StorageLevel(useDisk, useMemory, useOffHeap, deserialized, replication)
    from pyspark import StorageLevel
    return StorageLevel(True, True, False, False, 1)

def write_single_csv(df, folder: str):
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(folder)

def find_part_csv(folder: str) -> str:
    cands = sorted([p for p in glob(os.path.join(folder, "part-*.csv")) if not os.path.basename(p).startswith("._")])
    if cands:
        return cands[0]
    # fallback: sometimes Spark writes without .csv extension
    cands = sorted([p for p in glob(os.path.join(folder, "part-*")) if not os.path.basename(p).startswith("._")])
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No part file found in {folder}")

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    args = parse_args()
    spark = init_spark("ISD_2024_RF_local", args.driver_mem)
    sc = spark.sparkContext
    print(f"[INFO] Working dir: {os.path.abspath('.')}")
    print(f"[INFO] Spark UI: {sc.uiWebUrl}")

    # Columns
    label_col = "TMP_C"
    feature_cols = [
        "LATITUDE","LONGITUDE","ELEVATION",
        "WND_DIR_DEG","WND_SPD_MS",
        "CIG_HEIGHT_M","VIS_DIST_M",
        "DEW_C","SLP_hPa",
        "year","month","day","hour"
    ]
    imp_cols = [c + "_imp" for c in feature_cols]

    # I/O dirs
    RESULTS_DIR = args.outdir
    MODELS_DIR  = args.models
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
    ensure_dir(RESULTS_DIR); ensure_dir(MODELS_DIR); ensure_dir(FIG_DIR)

    # Load data
    train_df = spark.read.parquet(args.train)
    test_df  = spark.read.parquet(args.test)

    # Keep DATE only in test for reporting
    train_df = train_df.select(label_col, *feature_cols).repartition(args.partitions)
    req_cols = ["DATE", label_col] + feature_cols
    missing = [c for c in req_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing columns in test set: {missing}")
    test_df  = test_df.select(*req_cols)

    # Fit imputer once; assemble
    from pyspark.sql import functions as F
    from pyspark.ml.feature import Imputer, VectorAssembler
    imputer = Imputer(strategy="median", inputCols=feature_cols, outputCols=imp_cols)
    imputerModel = imputer.fit(train_df)
    assembler = VectorAssembler(inputCols=imp_cols, outputCol="features")

    SER = serialize_storage_level()

    # Transform train/test
    train_feat = assembler.transform(imputerModel.transform(train_df)) \
                          .select(label_col, "features") \
                          .persist(SER)
    _ = train_feat.count()  # materialize

    test_feat = assembler.transform(imputerModel.transform(test_df)) \
                         .select("DATE", label_col, "features")

    # RF model
    from pyspark.ml.regression import RandomForestRegressor
    rf = RandomForestRegressor(
        featuresCol="features", labelCol=label_col,
        numTrees=args.numTrees, maxDepth=args.maxDepth,
        subsamplingRate=args.subsample, featureSubsetStrategy="sqrt",
        seed=42
    )
    rf_model = rf.fit(train_feat)

    # Predict
    predictions = rf_model.transform(test_feat) \
        .select("DATE", label_col, F.col("prediction").alias("prediction")) \
        .persist(SER)
    _ = predictions.count()

    # Evaluate
    from pyspark.ml.evaluation import RegressionEvaluator
    e_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    e_mae  = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
    e_r2   = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    rmse = float(e_rmse.evaluate(predictions))
    mae  = float(e_mae.evaluate(predictions))
    r2   = float(e_r2.evaluate(predictions))
    print(f"[RF] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    # Save models
    IMPUTER_PATH = os.path.join(MODELS_DIR, "rf_imputerModel")
    MODEL_PATH   = os.path.join(MODELS_DIR, "rf_weather_model")
    imputerModel.write().overwrite().save(IMPUTER_PATH)
    rf_model.write().overwrite().save(MODEL_PATH)

    # Save predictions (parquet + csv)
    PRED_PARQ = os.path.join(RESULTS_DIR, "rf_predictions.parquet")
    PRED_CSV_DIR = os.path.join(RESULTS_DIR, "rf_predictions_csv")
    predictions.write.mode("overwrite").parquet(PRED_PARQ)
    write_single_csv(predictions, PRED_CSV_DIR)

    # Save metrics
    MET_CSV_DIR = os.path.join(RESULTS_DIR, "rf_metrics_csv")
    spark.createDataFrame([(rmse, mae, r2)], ["rmse","mae","r2"]) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(MET_CSV_DIR)

    # Feature importances
    imp_vec = rf_model.featureImportances
    feat_imps = [(feature_cols[i], float(imp_vec[i])) for i in range(len(feature_cols))]
    IMP_CSV_DIR = os.path.join(RESULTS_DIR, "rf_feature_importance_csv")
    spark.createDataFrame(feat_imps, ["feature","importance"]) \
        .orderBy(F.desc("importance")) \
        .coalesce(1).write.mode("overwrite").option("header", True).csv(IMP_CSV_DIR)

    # Diagnostics: RMSE by hour/month + calibration
    eval_df = (predictions
        .withColumn("residual", F.col(label_col) - F.col("prediction"))
        .withColumn("sq_err", F.pow(F.col("residual"), 2))
        .withColumn("hour", F.hour("DATE"))
        .withColumn("month", F.month("DATE"))
    )

    RMSE_HOUR_DIR = os.path.join(RESULTS_DIR, "rf_rmse_by_hour_csv")
    RMSE_MONTH_DIR = os.path.join(RESULTS_DIR, "rf_rmse_by_month_csv")
    CAL_DIR = os.path.join(RESULTS_DIR, "rf_calibration_csv")

    rmse_by_hour = eval_df.groupBy("hour") \
        .agg(F.sqrt(F.avg("sq_err")).alias("rmse"), F.avg("residual").alias("bias")) \
        .orderBy("hour")
    write_single_csv(rmse_by_hour, RMSE_HOUR_DIR)

    rmse_by_month = eval_df.groupBy("month") \
        .agg(F.sqrt(F.avg("sq_err")).alias("rmse"), F.avg("residual").alias("bias")) \
        .orderBy("month")
    write_single_csv(rmse_by_month, RMSE_MONTH_DIR)

    cal = (predictions
        .withColumn("bin", F.floor(F.col("prediction")/1.0))
        .groupBy("bin")
        .agg(F.avg(label_col).alias("avg_true"),
             F.avg("prediction").alias("avg_pred"),
             F.count("*").alias("n"))
        .orderBy("bin"))
    write_single_csv(cal, CAL_DIR)

    # Optional: figures
    if args.make_figs or args.make_summary:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            pred_csv = find_part_csv(PRED_CSV_DIR)
            imp_csv  = find_part_csv(IMP_CSV_DIR)
            rmh_csv  = find_part_csv(RMSE_HOUR_DIR)
            rmm_csv  = find_part_csv(RMSE_MONTH_DIR)

            df = pd.read_csv(pred_csv)
            # 1) Pred vs Actual
            plt.figure(figsize=(6,6))
            plt.scatter(df["TMP_C"], df["prediction"], s=5, alpha=0.5)
            lims = [min(df["TMP_C"].min(), df["prediction"].min()),
                    max(df["TMP_C"].max(), df["prediction"].max())]
            plt.plot(lims, lims, linestyle="--")
            plt.xlabel("Actual TMP_C (°C)"); plt.ylabel("Predicted TMP_C (°C)")
            plt.title("RF – Predicted vs Actual")
            plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "rf_pred_vs_actual.png"), dpi=300); plt.close()

            # 2) Residual hist
            df["residual"] = df["TMP_C"] - df["prediction"]
            plt.figure(figsize=(6,4))
            plt.hist(df["residual"], bins=50, edgecolor="k", alpha=0.7)
            plt.xlabel("Residual (°C)"); plt.ylabel("Count"); plt.title("RF – Residual Distribution")
            plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "rf_residual_hist.png"), dpi=300); plt.close()

            # 3) Feature importance barh
            imp_pd = pd.read_csv(imp_csv).sort_values("importance", ascending=False)
            plt.figure(figsize=(8,5))
            plt.barh(imp_pd["feature"], imp_pd["importance"])
            plt.gca().invert_yaxis()
            plt.xlabel("Importance"); plt.title("RF – Feature Importance")
            plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "rf_feature_importance.png"), dpi=300); plt.close()

            # 4) RMSE by hour/month
            rmh = pd.read_csv(rmh_csv)
            plt.figure(figsize=(7,4))
            plt.plot(rmh["hour"], rmh["rmse"], marker="o")
            plt.xlabel("Hour"); plt.ylabel("RMSE (°C)"); plt.title("RF – RMSE by Hour")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "rf_rmse_by_hour.png"), dpi=300); plt.close()

            rmm = pd.read_csv(rmm_csv)
            plt.figure(figsize=(7,4))
            plt.plot(rmm["month"], rmm["rmse"], marker="o")
            plt.xlabel("Month"); plt.ylabel("RMSE (°C)"); plt.title("RF – RMSE by Month")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "rf_rmse_by_month.png"), dpi=300); plt.close()

            print(f"[INFO] Figures saved to {FIG_DIR}/")
        except Exception as e:
            print(f"[WARN] Figure generation skipped: {e}")

    # Optional: summary markdown
    if args.make_summary:
        try:
            import pandas as pd
            MET_CSV = find_part_csv(MET_CSV_DIR)
            IMP_CSV = find_part_csv(IMP_CSV_DIR)
            metrics = pd.read_csv(MET_CSV)
            imp_pd  = pd.read_csv(IMP_CSV).sort_values("importance", ascending=False)

            rmse_v = metrics['rmse'].iloc[0]; mae_v = metrics['mae'].iloc[0]; r2_v = metrics['r2'].iloc[0]
            top5 = imp_pd.head(5)

            md = []
            md.append("# Random Forest Model – Results Summary\n")
            md.append("## Performance\n")
            md.append("| Metric | Value |\n|:-------|------:|\n")
            md.append(f"| RMSE (°C) | **{rmse_v:.3f}** |\n")
            md.append(f"| MAE (°C)  | **{mae_v:.3f}** |\n")
            md.append(f"| R²        | **{r2_v:.3f}** |\n\n")
            md.append("## Model Parameters\n")
            md.append("| Parameter | Value |\n|:-----------|:------|\n")
            md.append(f"| numTrees | {args.numTrees} |\n")
            md.append(f"| maxDepth | {args.maxDepth} |\n")
            md.append(f"| subsamplingRate | {args.subsample} |\n")
            md.append(f'| featureSubsetStrategy | "sqrt" |\n')
            md.append("| imputation | Median |\n| scaling | Not applied |\n| seed | 42 |\n\n")
            md.append("## Top 5 Feature Importances\n")
            md.append("| Rank | Feature | Importance |\n|:----:|:---------|-----------:|\n")
            for i, r in top5.reset_index(drop=True).iterrows():
                md.append(f"| {i+1} | {r['feature']} | {r['importance']:.4f} |\n")
            md.append("\n## Figures\n")
            md.append("| Figure | File Path |\n|:-------|:-----------|\n")
            md.append("| Predicted vs Actual | `results/figures/rf_pred_vs_actual.png` |\n")
            md.append("| Residual Histogram  | `results/figures/rf_residual_hist.png` |\n")
            md.append("| Feature Importance  | `results/figures/rf_feature_importance.png` |\n")
            md.append("| RMSE by Hour        | `results/figures/rf_rmse_by_hour.png` |\n")
            md.append("| RMSE by Month       | `results/figures/rf_rmse_by_month.png` |\n\n")
            md.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M}_\n")

            OUT_MD = os.path.join(RESULTS_DIR, "summary_rf.md")
            with open(OUT_MD, "w") as f:
                f.writelines(md)
            print(f"[INFO] Summary markdown saved to {OUT_MD}")
        except Exception as e:
            print(f"[WARN] Summary generation skipped: {e}")

    # Cleanup
    predictions.unpersist()
    train_feat.unpersist()
    spark.stop()
    print("[DONE] All steps completed successfully.")

if __name__ == "__main__":
    sys.exit(main())