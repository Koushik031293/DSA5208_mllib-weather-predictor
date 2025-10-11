**NOAA Weather Data Conversion & Cleaning**

**📦 Overview**

This project converts the NOAA Global Hourly CSV dataset (~51 GB) into optimized partitioned Parquet files, and performs a Spark-based cleaning and feature preparation step to produce a dataset ready for MLlib modeling.

🧰 Environment Setup

# Create & activate your environment
conda create -n weather-ml python=3.10 -y
conda activate weather-ml

# Install all dependencies
pip install -r requirements.txt

Optional (for DuckDB CLI inspection)
# macOS / Linux
brew install duckdb       # macOS
# or
conda install -c conda-forge duckdb-cli

**🚀 Step 1 — Convert NOAA CSV to Parquet (DuckDB)**

python src/local_csv_to_parquet.py \
  "data/noaa_raw/2024/*.csv" \
  "data/noaa_parquet/2024"

**Output**

[INFO] Files: 13,345
[INFO] DuckDB reported total rows across CSVs: 130,222,106
[OK] Done. Wrote Parquet partitions under: data/noaa_parquet/2024

**🩺 Step 2 — Read the parquet and do the cleaning**


python src/load_parquet.py

**Schema:**

root
 |-- STATION: string (nullable = true)
 |-- DATE: timestamp_ntz (nullable = true)
 |-- LATITUDE: double (nullable = true)
 |-- LONGITUDE: double (nullable = true)
 |-- ELEVATION: double (nullable = true)
 |-- TMP_C: double (nullable = true)
 |-- DEW_C: double (nullable = true)
 |-- SLP_hPa: double (nullable = true)
 |-- VIS_m: integer (nullable = true)
 |-- RH_pct: double (nullable = true)
 |-- year: integer (nullable = true)
 |-- month: integer (nullable = true)
