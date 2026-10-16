# Customer 6521-YYTYI

* Predicted churn probability (XGBoost (tuned), uncalibrated): **87.8%**
* Risk rank in the test set: 3 of 1407
* Actual outcome in the data: churned

## Main reasons for the risk (SHAP, log-odds)

| Feature | Value | Contribution |
|---|---|---|
| tenure | 1 | +0.842 |
| Contract | Month-to-month | +0.609 |
| TotalCharges | 93.30 | +0.290 |

## Factors lowering the risk

| Feature | Value | Contribution |
|---|---|---|
| Dependents | Yes | -0.132 |
| SeniorCitizen | No | -0.026 |

Baseline (average model output): -1.016 log-odds. Contributions are model associations, not causal effects.
