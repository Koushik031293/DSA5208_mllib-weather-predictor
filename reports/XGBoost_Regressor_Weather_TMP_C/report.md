# XGBoost Regressor (Weather TMP_C) — Model Evaluation Report

**Generated:** 2025-10-25 12:20  
**Target variable:** `TMP_C`  

---

## 1. Executive Summary
This report compares multiple trained configurations of the model.

This report compares multiple trained configurations.

- ✅ Lowest RMSE: **XGBoost_Optimised** (1.527428197843241)
- ✅ Highest R²: **XGBoost_Optimised** (0.985226035118103)

Lower RMSE / MAE means the model is closer to the true values.  
An R² near 1.0 means the model explains most of the variation in the target.

---

## 2. Metrics Summary

Below is the table of performance metrics:

| model             |    RMSE |     MAE |       R2 |
|:------------------|--------:|--------:|---------:|
| XGBoost_Base      | 2.15807 | 1.56732 | 0.970508 |
| XGBoost_Optimised | 1.52743 | 1.10803 | 0.985226 |

---

## 3. Feature Importance
![Feature Importance](results_cloud/xgb_results/xgb_base_feature_importance.png)

---

## 4. Predictions vs Actual
![Predicted vs Actual](results_cloud/xgb_results/xgb_base_actual_vs_pred.png)

---

## 5. Error Diagnostics / Training Behaviour
![Learning Curve](results_cloud/xgb_results/xgb_base_learning_curve.png)

![Metric Bar Comparison](results_cloud/xgb_results/xgb_base_metric_bar.png)


---

## 6. Key Takeaways
- RMSE and MAE quantify average prediction error (lower is better).
- R² shows how much variance in `TMP_C` is explained.
- Feature importance highlights which inputs drive predictions.
- Visual diagnostics (residuals, RMSE by hour/month, learning curve, metric bars) tell us *where* the model struggles or improves.

---

_Report auto-generated for `XGBoost Regressor (Weather TMP_C)`._
