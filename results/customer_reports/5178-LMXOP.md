# Customer 5178-LMXOP

* Predicted churn probability (XGBoost (tuned), uncalibrated): **90.7%**
* Risk rank in the test set: 1 of 1407
* Actual outcome in the data: churned

## Main reasons for the risk (SHAP, log-odds)

| Feature | Value | Contribution |
|---|---|---|
| tenure | 1 | +0.816 |
| Contract | Month-to-month | +0.599 |
| TotalCharges | 95.10 | +0.287 |

## Factors lowering the risk

| Feature | Value | Contribution |
|---|---|---|
| gender | Male | -0.018 |
| PhoneService | Yes | -0.008 |

Baseline (average model output): -1.016 log-odds. Contributions are model associations, not causal effects.
