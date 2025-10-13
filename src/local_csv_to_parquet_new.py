#!/usr/bin/env python3
import duckdb, os, sys, glob, shutil, uuid, pathlib 
from pathlib import Path

IN_GLOB  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/*.csv"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/compacted/2024.parquet"
KEEP = ["STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","WND","CIG","VIS","TMP","DEW","SLP"]

# Ensure output dir exists
pathlib.Path(os.path.dirname(OUT_FILE)).mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute(f"PRAGMA threads={os.cpu_count()}")
con.execute("PRAGMA memory_limit='8GB'")

# Keep only the 11 columns as VARCHAR to preserve raw strings like '0182,1'
select_list = ", ".join([f"CAST({c} AS VARCHAR) AS {c}" for c in KEEP])

# NOTE: all_varchar=1 ensures everything is read as text first; union_by_name handles column set differences across files
sql = f"""
SELECT {select_list}
FROM read_csv_auto(
  '{IN_GLOB}',
  HEADER=TRUE,
  SAMPLE_SIZE=200000,
  union_by_name=1,
  all_varchar=1
)
"""

print("[INFO] Writing single Parquet file:", OUT_FILE)
con.execute(f"COPY ({sql}) TO '{OUT_FILE}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');")
print("[OK] Done.")