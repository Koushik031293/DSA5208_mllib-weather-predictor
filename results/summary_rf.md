# 🌡️ Random Forest Model – Results Summary

## **Performance Metrics**
| Metric | Value |
|:-------|------:|
| RMSE (°C) | **3.722** |
| MAE (°C)  | **2.689** |
| R²        | **0.912** |

## **Model Parameters**
| Parameter | Value |
|:-----------|:------|
| numTrees | 80 |
| maxDepth | 10 |
| subsamplingRate | 0.7 |
| featureSubsetStrategy | "sqrt" |
| imputation | Median |
| scaling | Not applied |
| seed | 42 |

## **Top 5 Feature Importances**
| Rank | Feature | Importance |
|:----:|:---------|-----------:|
| 1 | DEW_C | 0.5805 |
| 2 | LATITUDE | 0.1700 |
| 3 | month | 0.0970 |
| 4 | CIG_HEIGHT_M | 0.0436 |
| 5 | SLP_hPa | 0.0432 |

## **Visual Results**

**Predicted vs Actual**

![Predicted vs Actual](results/figures/rf_pred_vs_actual.png)

**Residual Distribution**

![Residual Histogram](results/figures/rf_residual_hist.png)

**Feature Importance**

![Feature Importance](results/figures/rf_feature_importance.png)

**RMSE by Hour**

![RMSE by Hour](results/figures/rf_rmse_by_hour.png)

**RMSE by Month**

![RMSE by Month](results/figures/rf_rmse_by_month.png)

_Generated: 2025-10-16 14:12_
