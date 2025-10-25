# Random Forest Regressor (Weather TMP_C) — Model Evaluation Report

**Generated:** 2025-10-25 12:20  
**Target variable:** `TMP_C`  

---

## 1. Executive Summary
This report documents the performance of **Random Forest Regressor (Weather TMP_C)** trained to predict `TMP_C`.

Key performance indicators on the evaluation split:

- RMSE: 3.2329279001831592
- MAE: 2.333279119117583
- R²: 0.9338139117834048

Lower RMSE / MAE means the model is closer to the true values.  
An R² near 1.0 means the model explains most of the variation in the target.

---

## 2. Metrics Summary

Below is the table of performance metrics:

|    rmse |     mae |       r2 |
|--------:|--------:|---------:|
| 3.23293 | 2.33328 | 0.933814 |

---

## 3. Feature Importance
| feature      |   importance |
|:-------------|-------------:|
| DEW_C        |   0.58643    |
| LATITUDE     |   0.174782   |
| month        |   0.0951031  |
| CIG_HEIGHT_M |   0.0451922  |
| SLP_hPa      |   0.0339863  |
| hour         |   0.0219859  |
| LONGITUDE    |   0.0213332  |
| VIS_DIST_M   |   0.00935961 |
| ELEVATION    |   0.00865258 |
| WND_SPD_MS   |   0.00142607 |

![Feature Importance](results_cloud/rf_results/plots/rf_feature_importance.png)

---

## 4. Predictions vs Actual
![Predicted vs Actual](results_cloud/rf_results/plots/rf_pred_vs_actual.png)

Sample predictions:

| DATE                     |   year |   month |   day |   hour |   TMP_C |   prediction |
|:-------------------------|-------:|--------:|------:|-------:|--------:|-------------:|
| 2024-09-09T12:00:00.000Z |   2024 |       9 |     9 |     12 |    28   |     24.4123  |
| 2024-08-25T22:00:00.000Z |   2024 |       8 |    25 |     22 |    22.5 |     26.2063  |
| 2024-11-23T05:51:00.000Z |   2024 |      11 |    23 |      5 |    21.7 |     20.7739  |
| 2024-03-25T06:53:00.000Z |   2024 |       3 |    25 |      6 |     8.3 |      8.85306 |
| 2024-03-05T12:00:00.000Z |   2024 |       3 |     5 |     12 |    -7   |     -5.44982 |
| 2024-05-20T07:52:00.000Z |   2024 |       5 |    20 |      7 |    16.7 |     20.9249  |
| 2024-12-21T00:00:00.000Z |   2024 |      12 |    21 |      0 |    24.8 |     27.2128  |
| 2024-06-25T14:56:00.000Z |   2024 |       6 |    25 |     14 |    25.6 |     26.8469  |
| 2024-12-13T14:54:00.000Z |   2024 |      12 |    13 |     14 |    -3.9 |     -3.31615 |
| 2024-08-27T19:51:00.000Z |   2024 |       8 |    27 |     19 |    30.6 |     29.4327  |
| 2024-08-10T15:00:00.000Z |   2024 |       8 |    10 |     15 |    20.7 |     19.2885  |
| 2024-04-11T00:51:00.000Z |   2024 |       4 |    11 |      0 |     6.1 |      9.92051 |
| 2024-08-09T06:00:00.000Z |   2024 |       8 |     9 |      6 |    32.8 |     27.7461  |
| 2024-11-22T08:00:00.000Z |   2024 |      11 |    22 |      8 |     0.2 |      1.38283 |
| 2024-09-03T15:00:00.000Z |   2024 |       9 |     3 |     15 |    29.9 |     28.1804  |

---

## 5. Error Diagnostics / Training Behaviour
![Residual Distribution](results_cloud/rf_results/plots/rf_residual_hist.png)

![RMSE by Hour of Day](results_cloud/rf_results/plots/rf_rmse_by_hour.png)

![RMSE by Month](results_cloud/rf_results/plots/rf_rmse_by_month.png)


---

## 6. Key Takeaways
- RMSE and MAE quantify average prediction error (lower is better).
- R² shows how much variance in `TMP_C` is explained.
- Feature importance highlights which inputs drive predictions.
- Visual diagnostics (residuals, RMSE by hour/month, learning curve, metric bars) tell us *where* the model struggles or improves.

---

_Report auto-generated for `Random Forest Regressor (Weather TMP_C)`._
