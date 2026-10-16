# Customer 1069-XAIEM

* Predicted churn probability (XGBoost (tuned), uncalibrated): **89.9%**
* Risk rank in the test set: 2 of 1407
* Actual outcome in the data: churned

## Main reasons for the risk (SHAP, log-odds)

| Feature | Value | Contribution |
|---|---|---|
| tenure | 1 | +0.853 |
| Contract | Month-to-month | +0.634 |
| TotalCharges | 85.05 | +0.303 |

## Factors lowering the risk

| Feature | Value | Contribution |
|---|---|---|
| StreamingTV | No | -0.057 |
| PhoneService | Yes | -0.008 |

Baseline (average model output): -1.016 log-odds. Contributions are model associations, not causal effects.
