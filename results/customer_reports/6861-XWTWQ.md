# Customer 6861-XWTWQ

* Predicted churn probability (XGBoost (tuned), uncalibrated): **87.2%**
* Risk rank in the test set: 4 of 1407
* Actual outcome in the data: churned

## Main reasons for the risk (SHAP, log-odds)

| Feature | Value | Contribution |
|---|---|---|
| Contract | Month-to-month | +0.567 |
| MonthlyCharges | 99.25 | +0.440 |
| tenure | 7 | +0.344 |

## Factors lowering the risk

| Feature | Value | Contribution |
|---|---|---|
| gender | Male | -0.017 |
| PhoneService | Yes | -0.009 |

Baseline (average model output): -1.016 log-odds. Contributions are model associations, not causal effects.
