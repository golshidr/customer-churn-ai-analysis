# Customer 9057-SIHCH

* Predicted churn probability (XGBoost (tuned), uncalibrated): **87.0%**
* Risk rank in the test set: 5 of 1407
* Actual outcome in the data: churned

## Main reasons for the risk (SHAP, log-odds)

| Feature | Value | Contribution |
|---|---|---|
| Contract | Month-to-month | +0.581 |
| tenure | 3 | +0.470 |
| MonthlyCharges | 96.60 | +0.367 |

## Factors lowering the risk

| Feature | Value | Contribution |
|---|---|---|
| SeniorCitizen | No | -0.028 |
| PhoneService | Yes | -0.008 |

Baseline (average model output): -1.016 log-odds. Contributions are model associations, not causal effects.
