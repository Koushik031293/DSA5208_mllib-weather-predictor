
# === Auto-create a folder for all figures ===
import os
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
print(f"✅ Figures will be saved in: {os.path.abspath(FIG_DIR)}")

# === Utility to auto-save all open figures ===
from matplotlib import pyplot as plt
import atexit

def _save_all_figures():
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        fig.savefig(os.path.join(FIG_DIR, f"figure_{i}.png"), bbox_inches="tight", dpi=300)
    if plt.get_fignums():
        print(f"✅ Saved {len(plt.get_fignums())} figures to '{FIG_DIR}'")

atexit.register(_save_all_figures)
# === End of figure auto-save setup ===


#!/usr/bin/env python
# coding: utf-8





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

# EDA: Setup & Load
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("ISD_2024_EDA").getOrCreate()

PARQUET_PATH = "data/cleansed/2024_cleansed.parquet"  
df = spark.read.parquet(PARQUET_PATH)

print("Loaded:", PARQUET_PATH)
df.printSchema()
print("Rows:", f"{df.count():,}")




num_cols = ["WND_DIR_DEG", "WND_SPD_MS", "CIG_HEIGHT_M", "VIS_DIST_M", "TMP_C","DEW_C","SLP_hPa"]
df.select(*num_cols).describe().show()




# Null counts per column (Spark)
null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().T
null_counts.columns = ["null_count"]
null_counts = null_counts.sort_values("null_count", ascending=False)
null_counts.head(20)





# Sample a tiny fraction for plotting to avoid memory issues
# Adjust fraction if you want more/less points in plots
SAMPLE_FRAC = 0.001   # 0.1% — change to 0.002 or 0.005 if you want denser plots
SAMPLE_SEED = 42

plot_cols = ["TMP_C","DEW_C","SLP_hPa","WND_SPD_MS","VIS_DIST_M","CIG_HEIGHT_M","hour","month"]
sample_pdf = (
    df.select(*[c for c in plot_cols if c in df.columns])
      .sample(False, SAMPLE_FRAC, seed=SAMPLE_SEED)
      .toPandas()
)

sample_pdf.head()



# Histograms of key numeric variables
import matplotlib.pyplot as plt

cols_for_hist = ["WND_SPD_MS", "CIG_HEIGHT_M", "VIS_DIST_M", "TMP_C","DEW_C","SLP_hPa"]
sample_pdf[cols_for_hist].hist(bins=30, figsize=(12,8))
plt.suptitle("Distributions (sample)", y=1.02)
plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

corr_cols = ["TMP_C","DEW_C","SLP_hPa","WND_SPD_MS","VIS_DIST_M","CIG_HEIGHT_M"]
corr = sample_pdf[corr_cols].corr()

print(corr)

# Heatmap
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(corr.values, interpolation="nearest")
ax.set_xticks(np.arange(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticks(np.arange(len(corr_cols))); ax.set_yticklabels(corr_cols)
ax.set_title("Correlation (sample)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
# plt.show()




# Average temperature by hour
avg_by_hour = (df.groupBy("hour").agg(F.avg("TMP_C").alias("avg_tmp"))
                 .orderBy("hour")).toPandas()

# Average temperature by month
avg_by_month = (df.groupBy("month").agg(F.avg("TMP_C").alias("avg_tmp"))
                  .orderBy("month")).toPandas()

avg_by_hour.head(), avg_by_month.head()



import matplotlib.pyplot as plt

# Plot hourly pattern
plt.figure(figsize=(6,3))
plt.plot(avg_by_hour["hour"], avg_by_hour["avg_tmp"], marker="o")
plt.xlabel("Hour"); plt.ylabel("Avg TMP_C"); plt.title("Avg Temperature by Hour")
plt.tight_layout(); 
# plt.show()

# Plot monthly pattern
plt.figure(figsize=(6,3))
plt.plot(avg_by_month["month"], avg_by_month["avg_tmp"], marker="o")
plt.xlabel("Month"); plt.ylabel("Avg TMP_C"); plt.title("Avg Temperature by Month")
plt.tight_layout(); 
# plt.show()




# Scatter of station locations if many stations present
geo_cols_present = all(c in df.columns for c in ["LATITUDE","LONGITUDE"])
if geo_cols_present:
    geo_sample = df.select("LATITUDE","LONGITUDE").dropna().sample(False, 0.0005, seed=42).toPandas()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    plt.scatter(geo_sample["LONGITUDE"], geo_sample["LATITUDE"], s=1)
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Station Locations (sample)")
    plt.tight_layout(); 
    # plt.show()
else:
    print("LATITUDE/LONGITUDE not available for scatter.")



# Save summary tables for report reproducibility
from pathlib import Path
out_dir = Path("artifacts/eda"); out_dir.mkdir(parents=True, exist_ok=True)

df.describe(["TMP_C","DEW_C","SLP_hPa","WND_SPD_MS","VIS_DIST_M","CIG_HEIGHT_M"]).toPandas() \
  .to_csv(out_dir / "describe_numeric.csv", index=False)

avg_by_hour.to_csv(out_dir / "avg_tmp_by_hour.csv", index=False)
avg_by_month.to_csv(out_dir / "avg_tmp_by_month.csv", index=False)

print("EDA tables saved under:", out_dir.resolve())
