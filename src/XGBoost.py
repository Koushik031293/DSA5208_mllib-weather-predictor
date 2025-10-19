#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, math, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ===================== Utils ===================== #
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_dict(name: str, y_true, y_pred) -> dict:
    return {"model": name, "RMSE": rmse(y_true, y_pred),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2":  float(r2_score(y_true, y_pred))}

def get_ohe() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)

def get_feature_names(preprocessor: ColumnTransformer, numeric_cols, cat_cols):
    names = list(numeric_cols)
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            try:
                ohe_names = ohe.get_feature_names_out(cols)
            except Exception:
                ohe_names = []
                for i, c in enumerate(cols):
                    cats = ohe.categories_[i]
                    ohe_names.extend([f"{c}={k}" for k in cats])
            names.extend(list(ohe_names))
    return np.array(names)

def sanitize_xgb_params(p: dict) -> dict:
    if p is None:
        return {}
    drop = {
        "objective",
        "eval_metric",
        "n_jobs", "nthread",
        "tree_method",
        "random_state", "seed",
        "verbosity", "silent",
        # also drop these if present from model dumps
        "device", "gpu_id", "predictor"
    }
    return {k: v for k, v in p.items() if k not in drop}

def params_from_model(model: XGBRegressor) -> dict:
    p = model.get_xgb_params()
    p["n_estimators"] = model.get_params().get("n_estimators", p.get("n_estimators", 100))
    return sanitize_xgb_params(p)


# ===================== Plotting ===================== #
def plot_actual_vs_pred_single(y_true, y_pred, title, out_png):
    plt.figure(figsize=(7, 6))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([mn, mx], [mn, mx], linewidth=2)
    plt.title(title)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_feature_importance(xgb_model: XGBRegressor, feat_names, title, out_png, top_n=20):
    booster = xgb_model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    pairs = []
    for k, v in gain_map.items():
        m = re.match(r"f(\d+)$", k)
        if m:
            idx = int(m.group(1))
            if idx < len(feat_names):
                pairs.append((feat_names[idx], v))
    if not pairs:
        arr = getattr(xgb_model, "feature_importances_", None)
        if arr is None or len(arr) != len(feat_names):
            return
        pairs = list(zip(feat_names, arr))
    pairs.sort(key=lambda x: x[1], reverse=True)
    labels, scores = zip(*pairs[:top_n])
    plt.figure(figsize=(10, 6))
    y = np.arange(len(labels))
    plt.barh(y, scores); plt.yticks(y, labels); plt.gca().invert_yaxis()
    plt.title(title); plt.xlabel("Importance (gain)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_learning_curve_manual(params: dict, X, y, title, out_png, cv=3, random_state=42):
    params = sanitize_xgb_params(params)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    fracs = np.linspace(0.1, 1.0, 6)

    train_rmse, val_rmse, sizes = [], [], []
    rng = np.random.RandomState(random_state)

    for f in fracs:
        n = max(50, int(len(X) * f))
        idx = rng.choice(len(X), size=n, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        tr_scores, va_scores = [], []
        for tr_idx, va_idx in kf.split(X_sub):
            est = XGBRegressor(
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="rmse",
                **params
            )
            # Early stopping per fold using a small validation split from the training fold
            Xtr_in, Xva_in, ytr_in, yva_in = train_test_split(
                X_sub[tr_idx], y_sub[tr_idx], test_size=0.15, random_state=random_state
            )
            est.fit(Xtr_in, ytr_in, eval_set=[(Xva_in, yva_in)],
                    early_stopping_rounds=50, verbose=False)

            y_tr_pred = est.predict(X_sub[tr_idx])
            y_va_pred = est.predict(X_sub[va_idx])
            tr_scores.append(rmse(y_sub[tr_idx], y_tr_pred))
            va_scores.append(rmse(y_sub[va_idx], y_va_pred))

        sizes.append(n)
        train_rmse.append(float(np.mean(tr_scores)))
        val_rmse.append(float(np.mean(va_scores)))

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(sizes, val_rmse, marker="o", label="Validation RMSE")
    plt.title(title)
    plt.xlabel("Training Samples"); plt.ylabel("RMSE")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_metric_bar_single(metrics: dict, title, out_png):
    plt.figure(figsize=(6, 5))
    bars = ["RMSE", "MAE"]
    vals = [metrics["RMSE"], metrics["MAE"]]
    x = np.arange(len(bars))
    plt.bar(x, vals, width=0.55)
    plt.xticks(x, bars)
    plt.title(f"{title}\n(R2 = {metrics['R2']:.3f})")
    plt.ylabel("Score")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


# ===================== Main ===================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to TRAIN parquet")
    ap.add_argument("--test",  required=True, help="Path to TEST parquet")
    ap.add_argument("--ycol",  required=True, help="Target column name")
    ap.add_argument("--outdir", default="results_xgb")
    ap.add_argument("--n_iter", type=int, default=20, help="Manual random search iterations")
    ap.add_argument("--cv", type=int, default=3, help="Folds for learning curve")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0, help="Downsample train rows for speed (0=all)")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # ---- Load ----
    df_tr = pd.read_parquet(args.train)
    df_te = pd.read_parquet(args.test)

    if args.max_rows > 0:
        df_tr = df_tr.sample(n=min(args.max_rows, len(df_tr)), random_state=args.random_state)

    if args.ycol not in df_tr.columns or args.ycol not in df_te.columns:
        raise ValueError(f"Target '{args.ycol}' not found in both train/test.\n"
                         f"Train cols: {list(df_tr.columns)}\n"
                         f"Test  cols: {list(df_te.columns)}")

    # Optional: drop obvious timestamp/identifier if present
    for c in ["DATE"]:
        if c in df_tr.columns and c in df_te.columns:
            df_tr = df_tr.drop(columns=[c]); df_te = df_te.drop(columns=[c])

    X_tr = df_tr.drop(columns=[args.ycol]);  y_tr = df_tr[args.ycol].astype(float)
    X_te = df_te.drop(columns=[args.ycol]);  y_te = df_te[args.ycol].astype(float)

    # ---- Column typing ----
    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_tr.columns if c not in numeric_cols]

    # ---- Preprocessing to matrices (XGBRegressor works with numpy arrays) ----
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", get_ohe())]), cat_cols),
    ])

    Xtr_mat = pre.fit_transform(X_tr)
    Xte_mat = pre.transform(X_te)
    feat_names = get_feature_names(pre, numeric_cols, cat_cols)

    # ===================== BASELINE ===================== #
    xgb_base = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse"
    )
    xgb_base.fit(Xtr_mat, y_tr)
    yhat_base = xgb_base.predict(Xte_mat)
    base_metrics = metrics_dict("XGBoost_Base", y_te, yhat_base)

    # Separate plots for BASE
    plot_actual_vs_pred_single(
        y_te.values, yhat_base,
        "Actual vs Predicted — XGBoost Base",
        os.path.join(args.outdir, "xgb_base_actual_vs_pred.png")
    )
    plot_feature_importance(
        xgb_base, feat_names,
        "Feature Importance — XGBoost Base (Top 20)",
        os.path.join(args.outdir, "xgb_base_feature_importance_top20.png"),
        top_n=20
    )
    plot_learning_curve_manual(
        params_from_model(xgb_base), Xtr_mat, y_tr.values,
        "Learning Curve (RMSE) — XGBoost Base",
        os.path.join(args.outdir, "xgb_base_learning_curve.png"),
        cv=args.cv, random_state=args.random_state
    )
    plot_metric_bar_single(
        base_metrics,
        "Metric Bar — XGBoost Base",
        os.path.join(args.outdir, "xgb_base_metric_bar.png")
    )

    # ===================== OPTIMISED MODEL ===================== #
    rng = np.random.RandomState(args.random_state)
    X_tr_in, X_val, y_tr_in, y_val = train_test_split(
        Xtr_mat, y_tr, test_size=0.15, random_state=args.random_state
    )

    def sample_params():
        return {
            "n_estimators":     int(rng.randint(200, 1201) // 100 * 100),
            "learning_rate":    float(rng.uniform(0.01, 0.3)),
            "max_depth":        int(rng.randint(3, 13)),
            "min_child_weight": int(rng.randint(1, 10)),
            "subsample":        float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_alpha":        float(rng.uniform(0.0, 1.0)),
            "reg_lambda":       float(rng.uniform(0.0, 2.0)),
            "gamma":            float(rng.uniform(0.0, 2.0))
        }

    best_score = math.inf
    best_params = None

    for _ in range(args.n_iter):
        params = sample_params()
        est = XGBRegressor(
            objective="reg:squarederror",
            random_state=args.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
            **sanitize_xgb_params(params)
        )
        est.fit(X_tr_in, y_tr_in, eval_set=[(X_val, y_val)],
                early_stopping_rounds=50, verbose=False)
        score = rmse(y_val, est.predict(X_val))
        if score < best_score:
            best_score, best_params = score, params

    # Train final best model (with small ES to set best_iteration_)
    best_params = sanitize_xgb_params(best_params or {})
    xgb_best = XGBRegressor(
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
        **best_params
    )
    X_tr_in, X_val, y_tr_in, y_val = train_test_split(
        Xtr_mat, y_tr, test_size=0.15, random_state=args.random_state
    )
    xgb_best.fit(X_tr_in, y_tr_in, eval_set=[(X_val, y_val)],
                 early_stopping_rounds=50, verbose=False)

    yhat_best = xgb_best.predict(Xte_mat)
    best_metrics = metrics_dict("XGBoost_Optimised", y_te, yhat_best)

    # Separate plots for OPTIMISED
    plot_actual_vs_pred_single(
        y_te.values, yhat_best,
        "Actual vs Predicted — XGBoost Optimised",
        os.path.join(args.outdir, "xgb_opt_actual_vs_pred.png")
    )
    plot_feature_importance(
        xgb_best, feat_names,
        "Feature Importance — XGBoost Optimised (Top 20)",
        os.path.join(args.outdir, "xgb_opt_feature_importance_top20.png"),
        top_n=20
    )
    plot_learning_curve_manual(
        best_params, Xtr_mat, y_tr.values,
        "Learning Curve (RMSE) — XGBoost Optimised",
        os.path.join(args.outdir, "xgb_opt_learning_curve.png"),
        cv=args.cv, random_state=args.random_state
    )
    plot_metric_bar_single(
        best_metrics,
        "Metric Bar — XGBoost Optimised",
        os.path.join(args.outdir, "xgb_opt_metric_bar.png")
    )

    # ===================== Save summaries ===================== #
    results_df = pd.DataFrame([base_metrics, best_metrics])
    results_df.to_csv(os.path.join(args.outdir, "metrics_xgb_base_and_opt.csv"), index=False)
    pd.DataFrame([best_params]).to_csv(os.path.join(args.outdir, "xgb_best_params.csv"), index=False)

    print("\n=== XGBoost (Separate) Results ===")
    print(results_df.to_string(index=False))
    print("\nBest XGB Params (Optimised):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

