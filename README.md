**NOAA Weather Data Conversion & Cleaning**

**📦 Overview**

This project converts the NOAA Global Hourly CSV dataset (~51 GB) into optimized partitioned Parquet files, and performs a Spark-based cleaning and feature preparation step to produce a dataset ready for MLlib modeling.

🧰 Environment Setup

# Create & activate your environment
conda create -n weather-ml python=3.10 -y

conda activate weather-ml

# Install all dependencies
pip install -r requirement.txt

Optional (for DuckDB CLI inspection)
# macOS / Linux
brew install duckdb       # macOS
# or
conda install -c conda-forge duckdb-cli

**🚀 Step 1 — Convert NOAA CSV to Parquet (DuckDB)**

python src/local_csv_to_parquet_new.py \
  "data/raw/*.csv" \
  "data/compacted/2024.parquet"


**Schema:**
root
 |-- STATION: integer (nullable = true)
 
 |-- DATE: timestamp (nullable = true)
 
 |-- year: integer (nullable = true)
 
 |-- month: integer (nullable = true)
 
 |-- day: integer (nullable = true)
 
 |-- hour: integer (nullable = true)
 
 |-- LATITUDE: double (nullable = true)
 
 |-- LONGITUDE: double (nullable = true)
 
 |-- ELEVATION: double (nullable = true)
 
 |-- WND_DIR_DEG: double (nullable = true)
 
 |-- WND_TYPE: string (nullable = true)
 
 |-- WND_SPD_MS: double (nullable = true)
 
 |-- CIG_HEIGHT_M: double (nullable = true)
 
 |-- VIS_DIST_M: double (nullable = true)
 
 |-- TMP_C: double (nullable = true)
 
 |-- DEW_C: double (nullable = true)
 
 |-- SLP_hPa: double (nullable = true)
 
