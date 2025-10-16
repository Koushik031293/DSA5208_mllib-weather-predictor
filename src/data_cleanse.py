
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

# Now import Spark *after* JAVA_HOME is set
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("WeatherData2024_Load")
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    .getOrCreate()
)

print("Spark session started")



data_path = "data/compacted/2024.parquet"
df = spark.read.parquet(data_path)
print("Data loaded successfully")
df.printSchema()

# Count total rows
print(f"Row count: {df.count():,}")

# Quick null check per column
for c in df.columns:
    nulls = df.filter(df[c].isNull()).count()
    print(f"{c:<10}: {nulls:,} missing")



#convert date column to timestamp and add year, month, day, hour columns
from pyspark.sql import functions as F

#Parse DATE col
df = df.withColumn("DATE", F.to_timestamp("DATE", "yyyy-MM-dd'T'HH:mm:ss"))

# Add new time cols
df = (
    df.withColumn("year",  F.year("DATE"))
      .withColumn("month", F.month("DATE"))
      .withColumn("day",   F.dayofmonth("DATE"))
      .withColumn("hour",  F.hour("DATE"))
)

# Reorder columns 
orig = df.columns
new_order = []
for c in orig:
    if c == "DATE":
        new_order += ["DATE", "year", "month", "day", "hour"]
    elif c not in {"year","month","day","hour"}:  # avoid duplicates
        new_order.append(c)

df = df.select(*new_order)

# sanity check
df.select("DATE", "year", "month", "day", "hour").show(5, truncate=False)
df.printSchema()




df.show(20, truncate=False)




from pyspark.sql import functions as F

def split_and_replace_safe(df, col_name: str, new_cols: list[str]):
    """
    Split a comma-separated column (e.g., 'WND') into multiple columns and
    replace it in-place without duplication or unresolved-column issues.
    """
    if col_name not in df.columns:
        return df  # no-op if column missing

    # Remember current order and insertion index
    cols_before = df.columns
    insert_idx = cols_before.index(col_name)

    # Ensure string and create a temp tokens array
    tmp_tokens = f"__{col_name}_tokens"
    df = df.withColumn(col_name, F.col(col_name).cast("string"))
    df = df.withColumn(tmp_tokens, F.split(F.col(col_name), ","))

    # Add new columns from the tokens 
    for i, nc in enumerate(new_cols):
        df = df.withColumn(nc, F.col(tmp_tokens).getItem(i))

    # Drop the original and the temp array
    df = df.drop(col_name, tmp_tokens)

    # Reorder columns so new cols sit exactly where the original was
    base_cols = [c for c in cols_before if c != col_name]
    final_order = base_cols[:insert_idx] + new_cols + base_cols[insert_idx:]
    df = df.select(*final_order)

    return df




df = split_and_replace_safe(df, "WND",
    ["WND_DIR_DEG","WND_DIR_QUAL","WND_TYPE","WND_SPD_RAW","WND_SPD_QUAL"])

df.select("STATION","DATE","WND_DIR_DEG","WND_DIR_QUAL","WND_TYPE","WND_SPD_RAW","WND_SPD_QUAL").show(10, truncate=False)
#df.printSchema()




df.show(10, truncate=False)



df = split_and_replace_safe(df, "CIG",
    ["CIG_HEIGHT_M","CIG_QUAL","CIG_DET_CODE","CIG_CAVOK"])
df.show(10, truncate=False)




df = split_and_replace_safe(df, "VIS",
    ["VIS_DIST_M","VIS_QUAL","VIS_VAR_CODE","VIS_VAR_QUAL"])
df.show(10, truncate=False)




df = split_and_replace_safe(df, "TMP", ["TMP_RAW", "TMP_QUAL"])
df.show(10, truncate=False)



# Split DEW ("-0130,1") → [dew_point_raw, quality_code]
df = split_and_replace_safe(df, "DEW", ["DEW_RAW", "DEW_QUAL"])
df.show(10, truncate=False)




# Split SLP ("10208,1") → [pressure_raw, quality_code]
df = split_and_replace_safe(df, "SLP", ["SLP_RAW", "SLP_QUAL"])
df.show(10, truncate=False)



#count total rows and columns
print(f"Row count before cleansing: {df.count():,}, Column count: {len(df.columns)}")



# remove rows with missing or invalid data based on data document
from pyspark.sql import functions as F

# WIND special cases:
# 999 direction = missing unless variable
missing_wind_dir = (F.col("WND_DIR_DEG") == "999") & (F.col("WND_TYPE") != "V")

# type '9' = missing unless speed == 0000
missing_wind_type = (F.col("WND_TYPE") == "9") & (F.col("WND_SPD_RAW") != "0000")

# define overall missing conditions for all weather elements
missing_conditions = (
    missing_wind_dir |
    missing_wind_type |
    (F.col("WND_SPD_RAW") == "9999") |
    (F.col("CIG_HEIGHT_M") == "99999") |
    #(F.col("CIG_DET_CODE") == "9") | #not removing because i don't think detemination code interfers with model
    #(F.col("CIG_CAVOK") == "9") | #not removing because i don't think CAVOK code interfers with model
    (F.col("VIS_DIST_M") == "999999") |
    (F.col("TMP_RAW").isin("+9999", "9999")) |
    (F.col("DEW_RAW").isin("+9999", "9999")) |
    (F.col("SLP_RAW") == "99999")
)

# Apply filter
df = df.filter(~missing_conditions)
print(f"Remaining rows after removing missing or invalid data: {df.count():,}")




df.show(10, truncate=False)



# remove rows with suspect or erroneous quality codes
from pyspark.sql import functions as F

# accepted quality codes per variable
accepted_qc = {
    "WND_DIR_QUAL": ["0","1","4","5","9"],
    "WND_SPD_QUAL": ["0","1","4","5","9"],
    "CIG_QUAL":     ["0","1","4","5","9"],
    "VIS_QUAL":     ["0","1","4","5","9"],
    "VIS_VAR_QUAL": ["0","1","4","5","9"],
    "SLP_QUAL":     ["0","1","4","5","9"],
    "TMP_QUAL":     ["0","1","4","5","9","A","C","I","M","P","R","U"],
    "DEW_QUAL":     ["0","1","4","5","9","A","C","I","M","P","R","U"]
}

# Build condition: rows are valid only if each QC col is within allowed set
valid_condition = F.lit(True)
for col, good_vals in accepted_qc.items():
    valid_condition = valid_condition & (F.col(col).isin(good_vals))

# Filter
df = df.filter(valid_condition)
print(f"Remaining rows after removing suspect/erroneous quality codes: {df.count():,}")




df.show(10, truncate=False)



#drop quality code columns as they are no longer needed
qc_cols = [
    "WND_DIR_QUAL","WND_SPD_QUAL",
    "CIG_QUAL", "CIG_DET_CODE", "CIG_CAVOK",
    "VIS_QUAL","VIS_VAR_QUAL", "VIS_VAR_CODE",
    "TMP_QUAL","DEW_QUAL",
    "SLP_QUAL"
]

df = df.drop(*qc_cols)

print(f"Dropped {len(qc_cols)} QC columns. New column count: {len(df.columns)}")




df.show(20, truncate=False)



from pyspark.sql import functions as F

def scale_and_replace(df, col_name, expr):
    """Apply transformation and replace column at same position."""
    cols = df.columns
    idx = cols.index(col_name)
    df = df.withColumn(col_name, expr)
    # Reorder to keep column position consistent
    reordered = cols[:idx] + [col_name] + cols[idx+1:]
    return df.select(*reordered)

# WIND direction (1x)
df = scale_and_replace(df, "WND_DIR_DEG", F.col("WND_DIR_DEG").cast("double"))

# WIND speed (÷10)
df = scale_and_replace(df, "WND_SPD_RAW", (F.col("WND_SPD_RAW").cast("double") / 10.0))

# CEILING height (1x)
df = scale_and_replace(df, "CIG_HEIGHT_M", F.col("CIG_HEIGHT_M").cast("double"))

# VISIBILITY distance (1x)
df = scale_and_replace(df, "VIS_DIST_M", F.col("VIS_DIST_M").cast("double"))

# AIR temperature (÷10)
df = scale_and_replace(df, "TMP_RAW", (F.col("TMP_RAW").cast("double") / 10.0))

# DEW point (÷10)
df = scale_and_replace(df, "DEW_RAW", (F.col("DEW_RAW").cast("double") / 10.0))

# ATMOSPHERIC-PRESSURE pressure (÷10)
df = scale_and_replace(df, "SLP_RAW", (F.col("SLP_RAW").cast("double") / 10.0))

print("Scaled all numeric fields successfully!")
df.printSchema()



df.show(10, truncate=False)




df = df.withColumnRenamed("WND_SPD_RAW", "WND_SPD_MS") \
       .withColumnRenamed("TMP_RAW", "TMP_C") \
       .withColumnRenamed("DEW_RAW", "DEW_C") \
       .withColumnRenamed("SLP_RAW", "SLP_hPa")




df.show(10, truncate=False)




from pyspark.sql import functions as F

def replace_in_place(df, col_name, expr):
    cols = df.columns
    idx = cols.index(col_name)
    df = df.withColumn(col_name, expr)
    return df.select(*(cols[:idx] + [col_name] + cols[idx+1:]))

if "LATITUDE" in df.columns:
    df = replace_in_place(df, "LATITUDE",  F.col("LATITUDE").cast("double"))
if "LONGITUDE" in df.columns:
    df = replace_in_place(df, "LONGITUDE", F.col("LONGITUDE").cast("double"))
if "ELEVATION" in df.columns:
    df = replace_in_place(df, "ELEVATION", F.col("ELEVATION").cast("double"))
if "STATION" in df.columns:
    df = replace_in_place(df, "STATION", F.col("STATION").cast("integer"))
df.printSchema()


#output cleansed data to new parquet file
output_path = "data/cleansed/2024_cleansed_test.parquet"

# Coalesce(1) ensures a single output file
df.coalesce(1).write.mode("overwrite").parquet(output_path)

print(f"Cleansed dataset written to: {output_path}")



#split cleansed data into 70% training and 30% data parquet
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

train_output = "data/train_2024.parquet_test"
test_output  = "data/test_2024.parquet_test"

(train_df
    .coalesce(1)  # merge all partitions into one file
    .write.mode("overwrite")
    .parquet(train_output))

print(f"Cleansed dataset written to: {train_output}")
(test_df
    .coalesce(1)
    .write.mode("overwrite")
    .parquet(test_output))
print(f"Cleansed dataset written to: {test_output}")
