#!/usr/bin/env python3
# model_report_to_pdf.py — unified reporting for RF / XGB / Linear-ElasticNet families

import os
import glob
import argparse
import pandas as pd
from datetime import datetime
import pypandoc
import tempfile
import shutil


# ---------- basic utils ----------

def _read_single_csv(glob_pattern: str):
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    matches = sorted(matches)
    return pd.read_csv(matches[0])


def _read_all_csv(glob_pattern: str):
    matches = glob.glob(glob_pattern)
    matches = sorted(matches)
    dfs = []
    for path in matches:
        try:
            df = pd.read_csv(path)
            df["__source_file"] = os.path.basename(path)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def df_to_markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    df_disp = df.copy()
    if len(df_disp) > max_rows:
        df_disp = df_disp.head(max_rows)
    # requires tabulate in env: pip install tabulate
    return df_disp.to_markdown(index=False)


def debug_show_metrics_df(df: pd.DataFrame, label: str):
    print(f"=== DEBUG: {label} preview ===")
    if df is None:
        print("None")
    else:
        print("columns:", list(df.columns))
        print(df.head().to_string(index=False))
    print("=================================")


def safe_find_plot_any(plots_dir: str, keywords):
    """
    Return first plot file in plots_dir whose name contains any of keywords.
    """
    if not os.path.isdir(plots_dir):
        return None
    if isinstance(keywords, str):
        keywords = [keywords]
    for f in sorted(os.listdir(plots_dir)):
        lower = f.lower()
        if lower.endswith((".png", ".jpg", ".jpeg")):
            if any(kw.lower() in lower for kw in keywords):
                return os.path.join(plots_dir, f)
    return None


def safe_find_variant_plots(plots_dir: str, variant_prefix: str):
    """
    For linear / enet style:
    try to grab variant-specific plots: actual_vs_pred, residual_dist, residual_qq.
    Returns dict of available images for that variant.
    """
    found = {}
    if not os.path.isdir(plots_dir):
        return found

    targets = {
        "actual_vs_pred": ["actual_vs_pred", "vs_pred"],
        "residual_dist": ["residual_dist", "residual_hist", "residual_dist"],
        "residual_qq": ["residual_qq", "qq"]
    }

    files = sorted(os.listdir(plots_dir))
    for plot_type, kws in targets.items():
        for f in files:
            low = f.lower()
            if low.endswith((".png", ".jpg", ".jpeg")):
                if variant_prefix.lower() in low and any(kw in low for kw in kws):
                    found[plot_type] = os.path.join(plots_dir, f)
                    break
    return found


# ---------- layout detection ----------

def detect_layout(results_dir: str):
    """
    We try three layouts:
    1. RF-style
       results_dir/metrics_csv/*.csv (single-row rmse,mae,r2)
       results_dir/plots/*.png
    2. XGB-style
       results_dir/metrics*.csv (ONE csv with multiple rows, columns model,RMSE,MAE,R2)
       results_dir/*.png
    3. Linear/ElasticNet-style
       results_dir/*_metrics.csv (MULTIPLE one-row files, each with columns timestamp,model,rmse,mae,r2,...)
       results_dir/*.png
    Return dict with:
      layout_type: "rf" | "xgb" | "linear"
      metrics_df: DataFrame (combined/normalized)
      raw_metric_dfs: optional breakdown per variant for linear
      plots_dir
    """

    # --- 1. RF layout ---
    rf_metrics = _read_single_csv(os.path.join(results_dir, "metrics_csv", "*.csv"))
    if rf_metrics is not None:
        return {
            "layout_type": "rf",
            "metrics_df": rf_metrics,
            "raw_metric_dfs": None,
            "plots_dir": os.path.join(results_dir, "plots")
        }

    # --- 2. XGB layout ---
    # look for a single csv with multiple rows (model,RMSE,MAE,R2)
    xgb_metrics = _read_single_csv(os.path.join(results_dir, "metrics*.csv"))
    if xgb_metrics is not None and "model" in [c.lower() for c in xgb_metrics.columns]:
        # treat as xgb-style
        return {
            "layout_type": "xgb",
            "metrics_df": xgb_metrics,
            "raw_metric_dfs": None,
            "plots_dir": results_dir
        }

    # --- 3. Linear/ElasticNet layout ---
    # gather ALL *_metrics.csv, stack them
    lin_metrics_all = _read_all_csv(os.path.join(results_dir, "*_metrics.csv"))
    if lin_metrics_all is not None and not lin_metrics_all.empty:
        # we want to normalize column names to look like xgb comparison:
        # final table columns: model, RMSE, MAE, R2, (optional intercept)
        norm_rows = []
        for _, row in lin_metrics_all.iterrows():
            # expect columns: model, rmse, mae, r2, maybe intercept, maybe timestamp
            model_name = None
            rmse = None
            mae = None
            r2  = None
            intercept = None

            for col in row.index:
                col_low = col.lower()
                val = row[col]
                if col_low == "model":
                    model_name = val
                elif col_low == "rmse":
                    rmse = val
                elif col_low == "mae":
                    mae = val
                elif col_low == "r2":
                    r2 = val
                elif col_low == "intercept":
                    intercept = val

            norm_rows.append({
                "model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "intercept": intercept
            })

        norm_df = pd.DataFrame(norm_rows)

        return {
            "layout_type": "linear",
            "metrics_df": norm_df,
            "raw_metric_dfs": lin_metrics_all,
            "plots_dir": results_dir
        }

    # If nothing matched, fall back to "unknown"
    return {
        "layout_type": "unknown",
        "metrics_df": None,
        "raw_metric_dfs": None,
        "plots_dir": results_dir
    }


# ---------- summary text + comparison logic ----------

def build_summary_from_metrics(metrics_df: pd.DataFrame, layout_type: str):
    """
    Returns:
      summary_block (str): bullet lines for exec summary
      best_notes (str): highlight best candidate (multi-model cases)
      metrics_table_md (str): table view
    """

    if metrics_df is None or metrics_df.empty:
        return "No metrics found.\n", "", "_no data_\n"

    # table for the report
    metrics_table_md = df_to_markdown_table(metrics_df)

    cols_lower_map = {c.lower(): c for c in metrics_df.columns}
    has_model_col = "model" in cols_lower_map

    # MULTI model comparison (xgb style, linear style normalized)
    if has_model_col and len(metrics_df) > 1:
        summary_lines = ["This report compares multiple trained configurations.\n"]

        # try auto-pick best
        best_notes = ""
        try:
            if "RMSE" in metrics_df.columns:
                best_rmse_idx = metrics_df["RMSE"].idxmin()
                best_rmse_row = metrics_df.iloc[best_rmse_idx]
                best_notes += f"- ✅ Lowest RMSE: **{best_rmse_row['model']}** ({best_rmse_row['RMSE']})\n"

            if "R2" in metrics_df.columns:
                best_r2_idx = metrics_df["R2"].idxmax()
                best_r2_row = metrics_df.iloc[best_r2_idx]
                best_notes += f"- ✅ Highest R²: **{best_r2_row['model']}** ({best_r2_row['R2']})\n"
        except Exception:
            pass

        return (
            "\n".join(summary_lines),
            best_notes,
            metrics_table_md
        )

    # SINGLE model case (RF style or just one-row CSV)
    rmse = metrics_df.iloc[0][cols_lower_map.get("rmse")] if "rmse" in cols_lower_map else None
    mae  = metrics_df.iloc[0][cols_lower_map.get("mae")]  if "mae"  in cols_lower_map else None
    r2   = metrics_df.iloc[0][cols_lower_map.get("r2")]   if "r2"   in cols_lower_map else None

    summary_block = (
        f"- RMSE: {rmse}\n"
        f"- MAE: {mae}\n"
        f"- R²: {r2}\n"
    )
    return summary_block, "", metrics_table_md


# ---------- markdown assembly ----------

def build_markdown_report(model_name: str,
                          target_col: str,
                          layout_type: str,
                          metrics_df: pd.DataFrame,
                          raw_metric_dfs: pd.DataFrame,
                          plots_dir: str):
    """
    layout_type in {"rf","xgb","linear","unknown"}
    We will pick plots to show based on that.
    """

    summary_block, best_notes, metrics_table_md = build_summary_from_metrics(metrics_df, layout_type)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# {model_name} — Model Evaluation Report

**Generated:** {timestamp}  
**Target variable:** `{target_col}`  

---

## 1. Executive Summary

{summary_block}
"""

    if best_notes:
        md += best_notes + "\n"

    md += """Lower RMSE / MAE means the model is closer to the true values.  
An R² near 1.0 means the model explains most of the variation in the target.

---

## 2. Metrics Summary

Below is the table of performance metrics:

""" + metrics_table_md + "\n"

    # ---------- Plots section varies by layout ----------

    # RF layout plots
    if layout_type == "rf":
        img_feature_imp    = safe_find_plot_any(plots_dir, ["feature_importance"])
        img_pred_vs_actual = safe_find_plot_any(plots_dir, ["pred_vs_actual", "actual_vs_pred"])
        img_residual_hist  = safe_find_plot_any(plots_dir, ["residual_hist", "residual_dist"])
        img_rmse_hour      = safe_find_plot_any(plots_dir, ["rmse_by_hour"])
        img_rmse_month     = safe_find_plot_any(plots_dir, ["rmse_by_month"])

        md += """
---

## 3. Model Diagnostics (Random Forest)

### Feature Importance
"""
        if img_feature_imp:
            md += f"![Feature Importance]({os.path.relpath(img_feature_imp)})\n"

        md += "\n### Predictions vs Actual\n"
        if img_pred_vs_actual:
            md += f"![Predicted vs Actual]({os.path.relpath(img_pred_vs_actual)})\n"

        md += "\n### Residual Distribution\n"
        if img_residual_hist:
            md += f"![Residual Distribution]({os.path.relpath(img_residual_hist)})\n"

        md += "\n### RMSE by Hour / Month\n"
        if img_rmse_hour:
            md += f"![RMSE by Hour]({os.path.relpath(img_rmse_hour)})\n"
        if img_rmse_month:
            md += f"![RMSE by Month]({os.path.relpath(img_rmse_month)})\n"

    # XGB layout plots
    elif layout_type == "xgb":
        img_feature_imp    = safe_find_plot_any(plots_dir, ["feature_importance"])
        img_pred_vs_actual = safe_find_plot_any(plots_dir, ["actual_vs_pred", "pred_vs_actual"])
        img_learning_curve = safe_find_plot_any(plots_dir, ["learning_curve"])
        img_metric_bar     = safe_find_plot_any(plots_dir, ["metric_bar"])

        md += """
---

## 3. Model Diagnostics (XGBoost)

### Feature Importance
"""
        if img_feature_imp:
            md += f"![Feature Importance]({os.path.relpath(img_feature_imp)})\n"

        md += "\n### Predictions vs Actual\n"
        if img_pred_vs_actual:
            md += f"![Predicted vs Actual]({os.path.relpath(img_pred_vs_actual)})\n"

        md += "\n### Training Curve / Metric Bar\n"
        if img_learning_curve:
            md += f"![Learning Curve]({os.path.relpath(img_learning_curve)})\n"
        if img_metric_bar:
            md += f"![Metric Bar Comparison]({os.path.relpath(img_metric_bar)})\n"

    # Linear / Elastic Net layout plots
    elif layout_type == "linear":
        # We can loop variants using raw_metric_dfs["model"]
        md += """
---

## 3. Model Diagnostics (Linear / Elastic Net)
"""
        if raw_metric_dfs is not None and "model" in [c.lower() for c in raw_metric_dfs.columns]:
            model_col_actual = [c for c in raw_metric_dfs.columns if c.lower() == "model"][0]
            seen = set()
            for _, row in raw_metric_dfs.iterrows():
                variant_name = str(row[model_col_actual])
                # prevent duplicates if same model appears multiple metric files
                if variant_name in seen:
                    continue
                seen.add(variant_name)

                md += f"\n### {variant_name}\n"

                # guess variant keyword from file naming if available
                # try to pull likely prefix from __source_file if present
                prefix_guess = None
                if "__source_file" in row.index:
                    # e.g. 'enet_a0p25_metrics.csv' -> 'enet_a0p25'
                    base = row["__source_file"]
                    prefix_guess = base.replace("_metrics.csv", "")

                plots_for_variant = safe_find_variant_plots(plots_dir, prefix_guess or variant_name)

                if "actual_vs_pred" in plots_for_variant:
                    md += f"![Actual vs Pred]({os.path.relpath(plots_for_variant['actual_vs_pred'])})\n"
                if "residual_dist" in plots_for_variant:
                    md += f"![Residual Dist]({os.path.relpath(plots_for_variant['residual_dist'])})\n"
                if "residual_qq" in plots_for_variant:
                    md += f"![Residual QQ Plot]({os.path.relpath(plots_for_variant['residual_qq'])})\n"

    else:
        md += """
---

## 3. Model Diagnostics

No standard plots found for this layout.
"""

    # wrap up
    md += f"""
---

## 4. Key Takeaways

- RMSE and MAE quantify average prediction error (lower = better).
- R² shows how much variance in `{target_col}` is explained (closer to 1 = better).
- Comparing runs (baseline vs tuned / different alphas / etc.) shows how optimisation changes accuracy.
- Plots (feature importance, residuals, actual vs predicted, learning curve) help explain *why* each model behaves the way it does.

---

_Report auto-generated for `{model_name}`._
"""
    return md


# ---------- markdown->pdf ----------

def markdown_to_pdf(md_text: str, out_pdf: str):
    """
    Convert Markdown -> PDF using pandoc + xelatex.
    Requires:
      brew install pandoc basictex
      sudo tlmgr update --self
      sudo tlmgr install xelatex
      pip install pypandoc tabulate pandas
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_md:
        tmp_md.write(md_text.encode("utf-8"))
        tmp_md.flush()
        tmp_md_path = tmp_md.name

    pypandoc.convert_text(
        md_text,
        to="pdf",
        format="md",
        outputfile=out_pdf,
        extra_args=[
            "--standalone",
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "mainfont=Helvetica"
        ]
    )

    try:
        os.unlink(tmp_md_path)
    except Exception:
        pass


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF/Markdown model report for RF, XGB, or Linear/ElasticNet runs."
    )
    parser.add_argument("--results_dir", required=True,
                        help="Folder containing this model's results.")
    parser.add_argument("--model_name", required=True,
                        help="Human-readable model label (goes in PDF heading and report folder name).")
    parser.add_argument("--target_col", required=True,
                        help="Name of the target variable, e.g. TMP_C.")
    args = parser.parse_args()

    # Detect layout + load data
    layout_info = detect_layout(args.results_dir)
    layout_type   = layout_info["layout_type"]
    metrics_df    = layout_info["metrics_df"]
    raw_metric_dfs= layout_info["raw_metric_dfs"]
    plots_dir     = layout_info["plots_dir"]

    debug_show_metrics_df(metrics_df, "metrics_df (normalized)")
    debug_show_metrics_df(raw_metric_dfs, "raw_metric_dfs (linear only)")

    # Build the markdown text
    md_text = build_markdown_report(
        model_name=args.model_name,
        target_col=args.target_col,
        layout_type=layout_type,
        metrics_df=metrics_df,
        raw_metric_dfs=raw_metric_dfs,
        plots_dir=plots_dir
    )

    # Prepare output folder named by model_name
    safe_name = (
        args.model_name.replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("/", "_")
    )
    report_dir = os.path.join("reports", safe_name)
    os.makedirs(report_dir, exist_ok=True)

    # Copy plots into the report folder for convenience
    dest_plots_dir = os.path.join(report_dir, "plots")
    os.makedirs(dest_plots_dir, exist_ok=True)
    if os.path.isdir(plots_dir):
        for img in glob.glob(os.path.join(plots_dir, "*.png")):
            shutil.copy(img, dest_plots_dir)

    # Write markdown + pdf
    md_path = os.path.join(report_dir, "report.md")
    pdf_path = os.path.join(report_dir, "report.pdf")

    with open(md_path, "w") as f:
        f.write(md_text)

    markdown_to_pdf(md_text, pdf_path)

    print("✅ Report generated:")
    print(f"  Markdown : {md_path}")
    print(f"  PDF      : {pdf_path}")
    print(f"  Plots    : {dest_plots_dir}")


if __name__ == "__main__":
    main()