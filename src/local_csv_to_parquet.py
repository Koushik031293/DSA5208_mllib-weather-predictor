#!/usr/bin/env python3
import duckdb
import os, sys, glob, shutil, uuid
from pathlib import Path

print("[DEBUG] script version: temp-swap v5 (force STATION as VARCHAR)")

IN_GLOB  = sys.argv[1] if len(sys.argv) > 1 else "data/noaa_raw/2024/*.csv"
OUT_ROOT = sys.argv[2] if len(sys.argv) > 2 else "data/noaa_parquet/2024"

final_root = Path(OUT_ROOT).resolve()
tmp_root   = final_root.parent / (final_root.name + "_tmpwrite")

files = glob.glob(IN_GLOB)
if not files:
    raise SystemExit(f"No CSV files match: {IN_GLOB}")
print(f"[INFO] Files:  {len(files):,}")
print(f"[INFO] Final:  {final_root}")
print(f"[INFO] Temp :  {tmp_root}")

con = duckdb.connect()
con.execute(f"PRAGMA threads={os.cpu_count()}")
con.execute("PRAGMA memory_limit='8GB'")

def Q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def present_column(csv_path: str, col: str) -> bool:
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_csv_auto(?, SAMPLE_SIZE=200000, HEADER=TRUE)",
        [csv_path]
    ).fetchall()
    cols = {r[0] for r in rows}
    return col in cols

def build_select_for_file(csv_path: str) -> str:
    has = {c: present_column(csv_path, c) for c in
           ["STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","TMP","DEW","SLP","VIS","RH"]}

    def col_or_null(c, cast=None):
        if has[c]:
            return f"CAST({Q(c)} AS {cast})" if cast else Q(c)
        return "NULL"

    # --- Force STATION to text regardless of inference ---
    station_expr = "CAST(" + Q("STATION") + " AS VARCHAR)" if has["STATION"] else "NULL"

    def tenths_expr(c):
        if not has.get(c, False):
            return "NULL::DOUBLE"
        return f"""
        CASE
          WHEN REGEXP_EXTRACT({Q(c)}, '^([^,]*)', 1) IS NULL THEN NULL
          WHEN REGEXP_EXTRACT({Q(c)}, '^([^,]*)', 1) = '' THEN NULL
          ELSE
            CASE
              WHEN TRY_CAST(REGEXP_REPLACE(REGEXP_EXTRACT({Q(c)}, '^([^,]*)', 1), '^\\+', '') AS INT) = -9999 THEN NULL
              ELSE TRY_CAST(REGEXP_REPLACE(REGEXP_EXTRACT({Q(c)}, '^([^,]*)', 1), '^\\+', '') AS INT) / 10.0
            END
        END
        """

    def vis_expr():
        if not has.get("VIS", False):
            return "NULL::INT"
        return f"""
        CASE
          WHEN REGEXP_EXTRACT({Q('VIS')}, '^([^,]*)', 1) IS NULL THEN NULL
          WHEN REGEXP_EXTRACT({Q('VIS')}, '^([^,]*)', 1) = '' THEN NULL
          ELSE
            CASE
              WHEN TRY_CAST(REGEXP_EXTRACT({Q('VIS')}, '^([^,]*)', 1) AS INT) = -999999 THEN NULL
              ELSE TRY_CAST(REGEXP_EXTRACT({Q('VIS')}, '^([^,]*)', 1) AS INT)
            END
        END
        """

    date_expr = f"CAST({Q('DATE')} AS TIMESTAMP)" if has["DATE"] else "NULL::TIMESTAMP"
    tmp_c     = tenths_expr("TMP")
    dew_c     = tenths_expr("DEW")
    slp_hpa   = tenths_expr("SLP")
    vis_m     = vis_expr()
    rh_given  = f"TRY_CAST({Q('RH')} AS DOUBLE)" if has["RH"] else "NULL::DOUBLE"

    rh_from_magnus = f"""
    100.0 * EXP(
      (17.625 * ({dew_c})) / (243.04 + ({dew_c})) -
      (17.625 * ({tmp_c})) / (243.04 + ({tmp_c}))
    )
    """

    rh_pct = f"""
    CASE
      WHEN {rh_given} IS NOT NULL THEN {rh_given}
      WHEN ({tmp_c}) IS NOT NULL AND ({dew_c}) IS NOT NULL THEN
        GREATEST(0, LEAST(100, {rh_from_magnus}))
      ELSE NULL
    END
    """

    year_expr  = f"EXTRACT(YEAR  FROM {date_expr})"
    month_expr = f"EXTRACT(MONTH FROM {date_expr})"

    return f"""
    SELECT
      {station_expr}                          AS STATION,
      {date_expr}                              AS DATE,
      {col_or_null('LATITUDE','DOUBLE')}       AS LATITUDE,
      {col_or_null('LONGITUDE','DOUBLE')}      AS LONGITUDE,
      {col_or_null('ELEVATION','DOUBLE')}      AS ELEVATION,
      {tmp_c}                                  AS TMP_C,
      {dew_c}                                  AS DEW_C,
      {slp_hpa}                                AS SLP_hPa,
      {vis_m}                                  AS VIS_m,
      {rh_pct}                                 AS RH_pct,
      {year_expr}                               AS year,
      {month_expr}                              AS month
    FROM read_csv_auto('{csv_path}', SAMPLE_SIZE=200000, HEADER=TRUE)
    """

# --- clean temp root ---
if tmp_root.exists():
    print(f"[WARN] Removing previous temp dir: {tmp_root}")
    shutil.rmtree(tmp_root)
tmp_root.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Writing to temp dir: {tmp_root}")

# --- process: count rows -> write -> recursive merge with unique filenames ---
total_rows = 0
for idx, csv in enumerate(files, 1):
    sql = build_select_for_file(csv)
    n = con.execute(f"SELECT COUNT(*) FROM ({sql}) t").fetchone()[0]
    total_rows += n

    staging_dir = tmp_root / f"stage_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    con.execute(f"""
    COPY ({sql})
    TO '{staging_dir.as_posix()}'
    (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION 'SNAPPY');
    """)

    for year_dir in staging_dir.iterdir():
        if not year_dir.is_dir():
            continue
        dest_year = (tmp_root / year_dir.name)
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            dest_month = (dest_year / month_dir.name)
            dest_month.mkdir(parents=True, exist_ok=True)
            for f in month_dir.glob("*.parquet"):
                suffix = f"_{idx:06d}_{uuid.uuid4().hex[:6]}"
                f.rename(dest_month / (f.stem + suffix + f.suffix))

    shutil.rmtree(staging_dir)

    if idx % 100 == 0:
        print(f"[INFO] Processed {idx} files... running total rows: {total_rows:,}")

print(f"[INFO] DuckDB reported total rows across CSVs: {total_rows:,}")

if not any(tmp_root.rglob("*.parquet")):
    raise RuntimeError(f"No parquet files written into temp dir: {tmp_root}")

if final_root.exists():
    print(f"[WARN] Removing existing final dir: {final_root}")
    shutil.rmtree(final_root)
tmp_root.rename(final_root)

con.close()
print(f"[OK] Done. Wrote Parquet partitions under: {final_root}")