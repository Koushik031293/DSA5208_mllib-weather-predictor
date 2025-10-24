
#!/usr/bin/env python3
# (Patched) main.py — safer input mapping, model selector
# - Fix: never delete the source when src == dst
# - Better handling for directory-style Parquet
# - Keeps --models/--skip/--list-models features

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src" 
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EXPECTED = DATA_DIR / "train_2024.parquet"
TEST_EXPECTED  = DATA_DIR / "test_2024.parquet"

MODEL_ALIASES = {
    "linear": {"linear", "lin", "lr", "base"},
    "ridge":  {"ridge", "lasso", "enet", "regularized", "reg"},
    "rf":     {"rf", "randomforest", "random_forest", "random-forest"},
    "xgb":    {"xgb", "xgboost"},
}
MODEL_ORDER = ["linear", "ridge", "rf", "xgb"]

def _script(name: str) -> Path:
    """Helper to get full path to a script inside src/."""
    return SRC_DIR / name

def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)

def _is_parquet_dir(p: Path) -> bool:
    return p.is_dir()

def _copy_or_link(src: Path, dst: Path):
    """Place src at dst without destroying src."""
    if _same_path(src, dst):
        print(f"[INFO] {src} is already at expected location; skipping copy/link.")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if _is_parquet_dir(src):
        # Directory-style parquet: safe copytree with dirs_exist_ok
        if dst.exists() and dst.is_file():
            dst.unlink()
        # Copy to temp then swap for atomicity
        tmp = dst.parent / (dst.name + ".tmp_copy")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp, dirs_exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
    else:
        # Single parquet file
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst, ignore_errors=True)
            else:
                dst.unlink()
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def _run(cmd: list, cwd: Path = ROOT, env: dict | None = None, name: str = "job"):
    print(f"\n[RUN] {name}: {' '.join(map(str, cmd))}\n")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] {name} failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
        raise SystemExit(proc.returncode)
    else:
        print(proc.stdout[-4000:])
        if proc.stderr.strip():
            print("\n[WARN/INFO STDERR TAIL]\n" + proc.stderr[-2000:])

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark selected models on the same dataset.")
    p.add_argument("--train", required=True, help="Path to TRAIN parquet (folder or file)")
    p.add_argument("--test",  required=True, help="Path to TEST parquet (folder or file)")
    p.add_argument("--ycol", required=True, help="Target column name for training (e.g., TMP_C)")
    p.add_argument("--driver_mem", default="14g", help="Spark driver memory for Spark-based models")
    p.add_argument("--java11", action="store_true", help="On macOS, export JAVA_HOME to JDK 11 for Spark")
    p.add_argument("--skip", default="", help="Comma list to skip: linear,ridge,rf,xgb")
    p.add_argument("--models", default="", help="Comma list to run (overrides default all). e.g., linear,rf")
    p.add_argument("--list-models", action="store_true", help="Print canonical model names and exit")
    return p.parse_args()

def ensure_inputs(train_path: Path, test_path: Path):
    train_path = Path(train_path)
    test_path  = Path(test_path)
    if not train_path.exists(): raise FileNotFoundError(f"Train parquet not found: {train_path}")
    if not test_path.exists():  raise FileNotFoundError(f"Test parquet not found:  {test_path}")
    _copy_or_link(train_path, TRAIN_EXPECTED)
    _copy_or_link(test_path,  TEST_EXPECTED)
    print(f"[INFO] Using TRAIN={TRAIN_EXPECTED}  TEST={TEST_EXPECTED}")

def maybe_java11_env(use_java11: bool) -> dict:
    env = os.environ.copy()
    if use_java11 and sys.platform == "darwin":
        try:
            jhome = subprocess.check_output(["/usr/libexec/java_home", "-v", "11"], text=True).strip()
            env["JAVA_HOME"] = jhome
            print(f"[INFO] JAVA_HOME -> {jhome}")
        except Exception as e:
            print(f"[WARN] Could not set JAVA_HOME to 11 automatically: {e}")
    env.setdefault("PYSPARK_SUBMIT_ARGS", f"--driver-memory {args.driver_mem} pyspark-shell")
    return env

def _normalize_model_token(tok: str) -> str | None:
    t = tok.strip().lower()
    for canon, aliases in MODEL_ALIASES.items():
        if t == canon or t in aliases:
            return canon
    return None

def _resolve_selection(models_arg: str, skip_arg: str):
    selected = set(MODEL_ORDER)
    if models_arg.strip():
        requested = set()
        for tok in models_arg.split(","):
            if tok.strip():
                canon = _normalize_model_token(tok)
                if not canon:
                    raise SystemExit(f"[ERR] Unknown model token: {tok}. Use --list-models to see options.")
                requested.add(canon)
        selected = requested
    for tok in (t for t in skip_arg.split(",") if t.strip()):
        canon = _normalize_model_token(tok)
        if canon:
            selected.discard(canon)
        else:
            print(f"[WARN] Unknown skip token ignored: {tok}")
    return [m for m in MODEL_ORDER if m in selected]

def run_linear(env):
    _run([sys.executable, str(_script("base_linearReg.py"))],
         cwd=ROOT, name="LinearRegression (Spark)", env=env)

def run_ridge_lasso(env):
    _run([sys.executable, str(_script("Ridge_Lasso.py"))],
         cwd=ROOT, name="Ridge/Lasso/ENet (Spark)", env=env)


def run_rf(train, test, env):
    outdir = RESULTS_DIR / "rf"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_script("model_rf.py")),
        "--train", str(train),
        "--test",  str(test),
        "--outdir", str(outdir),
        "--driver_mem", args.driver_mem,
        "--ycol", args.ycol,               # ← added
        "--make_figs", "--make_summary"
    ]
    _run(cmd, cwd=ROOT, name="RandomForest (Spark)", env=env)


def run_xgb(train, test, env):
    outdir = RESULTS_DIR / "xgb"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_script("XGBoost.py")),
        "--train", str(train),
        "--test",  str(test),
        "--outdir", str(outdir),
        "--ycol", args.ycol                # ← added
    ]
    _run(cmd, cwd=ROOT, name="XGBoost (sklearn)", env=env)

def collect_and_merge_metrics() -> pd.DataFrame:
    rows = []
    lin_csv = RESULTS_DIR / "base_linear_metrics.csv"
    if lin_csv.exists():
        df = pd.read_csv(lin_csv)
        row = df.iloc[0].to_dict()
        row.setdefault("model", "LinearRegression")
        rows.append(row)

    reg_csv = RESULTS_DIR / "regularized_summary.csv"
    if reg_csv.exists():
        rdf = pd.read_csv(reg_csv)
        best = rdf.sort_values("rmse").groupby("model", as_index=False).first()
        for _, r in best.iterrows():
            rows.append({
                "model": f"{r['model']} (best)" if 'model' in r else "Regularized (best)",
                "rmse": r.get("rmse"),
                "mae":  r.get("mae"),
                "r2":   r.get("r2")
            })

    rf_csv = RESULTS_DIR / "rf" / "rf_metrics.csv"
    if rf_csv.exists():
        rdf = pd.read_csv(rf_csv)
        rows.append({"model": "RandomForest", **{k: rdf[k].iloc[0] for k in rdf.columns if k in ["rmse","mae","r2"]}})

    xgb_csv = RESULTS_DIR / "xgb" / "metrics_xgb_base_and_opt.csv"
    if xgb_csv.exists():
        xdf = pd.read_csv(xgb_csv)
        for _, r in xdf.iterrows():
            rows.append({
                "model": r.get("model", "XGBoost"),
                "rmse": float(r.get("RMSE")) if "RMSE" in r else r.get("rmse"),
                "mae":  float(r.get("MAE"))  if "MAE" in r  else r.get("mae"),
                "r2":   float(r.get("R2"))   if "R2" in r   else r.get("r2"),
            })

    if not rows:
        print("[WARN] No metrics found to aggregate.")
        return pd.DataFrame(columns=["model","rmse","mae","r2"])

    bench = pd.DataFrame(rows)
    bench = bench.groupby("model", as_index=False).agg({"rmse":"min","mae":"min","r2":"max"})
    bench = bench.sort_values(["rmse","mae"], ascending=[True, True]).reset_index(drop=True)

    out_csv = RESULTS_DIR / "benchmark_summary.csv"
    bench.to_csv(out_csv, index=False)

    out_md = RESULTS_DIR / "benchmark_summary.md"
    with open(out_md, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write("| Model | RMSE | MAE | R² |\n|:--|--:|--:|--:|\n")
        for _, r in bench.iterrows():
            f.write(f"| {r['model']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |\n")
    print(f"[INFO] Wrote {out_csv} and {out_md}")
    return bench

def main():
    global args
    args = parse_args()

    if args.list_models:
        print("Canonical model names: " + ", ".join(MODEL_ORDER))
        return 0

    ensure_inputs(Path(args.train), Path(args.test))
    env = maybe_java11_env(args.java11)

    selection = _resolve_selection(args.models, args.skip)
    if not selection:
        print("[ERR] No models selected after applying --models/--skip.")
        return 2

    print(f"[INFO] Models to run -> {', '.join(selection)}")

    for m in selection:
        if m == "linear":
            run_linear(env)
        elif m == "ridge":
            run_ridge_lasso(env)
        elif m == "rf":
            run_rf(TRAIN_EXPECTED, TEST_EXPECTED, env)
        elif m == "xgb":
            run_xgb(TRAIN_EXPECTED, TEST_EXPECTED, env)

    bench = collect_and_merge_metrics()
    if not bench.empty:
        print("\n=== Benchmark Summary ===")
        try:
            from tabulate import tabulate
            print(tabulate(bench, headers="keys", tablefmt="github", floatfmt=".4f"))
        except Exception:
            print(bench.to_string(index=False))

    return 0

if __name__ == "__main__":
    sys.exit(main())
