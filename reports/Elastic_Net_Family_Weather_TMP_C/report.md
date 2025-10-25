# Elastic Net Family (Weather TMP_C) — Model Evaluation Report

**Generated:** 2025-10-25 12:27  
**Target variable:** `TMP_C`  

---

## 1. Executive Summary

This report compares multiple trained configurations.

- ✅ Lowest RMSE: **Baseline Linear Regression** (5.3225)
- ✅ Highest R²: **Baseline Linear Regression** (0.8206)

Lower RMSE / MAE means the model is closer to the true values.  
An R² near 1.0 means the model explains most of the variation in the target.

---

## 2. Metrics Summary

Below is the table of performance metrics:

| model                      |   RMSE |    MAE |     R2 |   intercept |
|:---------------------------|-------:|-------:|-------:|------------:|
| Baseline Linear Regression | 5.3225 | 3.7743 | 0.8206 |     48.6964 |
| Enet (alpha=0.25)          | 5.3225 | 3.7744 | 0.8206 |     13.1766 |
| Enet (alpha=0.5)           | 5.3225 | 3.7745 | 0.8206 |     13.1766 |
| Enet (alpha=0.75)          | 5.3225 | 3.7743 | 0.8206 |     13.1766 |
| Lasso (alpha=1.0)          | 5.3225 | 3.7744 | 0.8206 |     13.1766 |
| Ridge (alpha=0.0)          | 5.3225 | 3.7748 | 0.8206 |     13.1766 |

---

## 3. Model Diagnostics (Linear / Elastic Net)

### Baseline Linear Regression
![Actual vs Pred](results_cloud/linear/lasso/base_linear_reg_actual_vs_pred.png)
![Residual Dist](results_cloud/linear/lasso/base_linear_reg_residual_dist.png)
![Residual QQ Plot](results_cloud/linear/lasso/base_linear_reg_residual_qq.png)

### Enet (alpha=0.25)
![Actual vs Pred](results_cloud/linear/lasso/enet_a0p25_actual_vs_pred.png)
![Residual Dist](results_cloud/linear/lasso/enet_a0p25_residual_dist.png)
![Residual QQ Plot](results_cloud/linear/lasso/enet_a0p25_residual_qq.png)

### Enet (alpha=0.5)
![Actual vs Pred](results_cloud/linear/lasso/enet_a0p5_actual_vs_pred.png)
![Residual Dist](results_cloud/linear/lasso/enet_a0p5_residual_dist.png)
![Residual QQ Plot](results_cloud/linear/lasso/enet_a0p5_residual_qq.png)

### Enet (alpha=0.75)
![Actual vs Pred](results_cloud/linear/lasso/enet_a0p75_actual_vs_pred.png)
![Residual Dist](results_cloud/linear/lasso/enet_a0p75_residual_dist.png)
![Residual QQ Plot](results_cloud/linear/lasso/enet_a0p75_residual_qq.png)

### Lasso (alpha=1.0)
![Actual vs Pred](results_cloud/linear/lasso/lasso_a1_actual_vs_pred.png)
![Residual Dist](results_cloud/linear/lasso/lasso_a1_residual_dist.png)
![Residual QQ Plot](results_cloud/linear/lasso/lasso_a1_residual_qq.png)

### Ridge (alpha=0.0)
![Actual vs Pred](results_cloud/linear/lasso/ridge_a0_actual_vs_pred.png)
![Residual Dist](results_cloud/linear/lasso/ridge_a0_residual_dist.png)
![Residual QQ Plot](results_cloud/linear/lasso/ridge_a0_residual_qq.png)

---

## 4. Key Takeaways

- RMSE and MAE quantify average prediction error (lower = better).
- R² shows how much variance in `TMP_C` is explained (closer to 1 = better).
- Comparing runs (baseline vs tuned / different alphas / etc.) shows how optimisation changes accuracy.
- Plots (feature importance, residuals, actual vs predicted, learning curve) help explain *why* each model behaves the way it does.

---

_Report auto-generated for `Elastic Net Family (Weather TMP_C)`._
