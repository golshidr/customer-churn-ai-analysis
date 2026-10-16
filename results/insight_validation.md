| Claim | Observed churn rates (all customers) | SHAP evidence (test set) | Model agrees |
|---|---|---|---|
| 1. Short tenure -> more churn | tenure <= 12 m: 47.7% vs > 12 m: 17.1% | Spearman(tenure, SHAP) = -0.97 | yes |
| 2. Higher monthly charges -> more churn | top quartile (> 90): 32.9% vs bottom (< 36): 11.3% | Spearman(MonthlyCharges, SHAP) = +0.95 | yes |
| 3. Long-term contracts -> less churn | two-year: 2.8%, one-year: 11.3%, month-to-month: 42.7% | mean SHAP two-year -1.33 vs month-to-month +0.51 | yes |
| 4. Fiber optic -> more churn | fiber: 41.9% vs DSL: 19.0% | mean SHAP fiber +0.25 vs DSL -0.26 | yes |
| 5. TechSupport -> less churn (internet customers) | with: 15.2% vs without: 41.6% | mean SHAP with -0.21 vs without +0.14 | yes |
| 5. OnlineSecurity -> less churn (internet customers) | with: 14.6% vs without: 41.8% | mean SHAP with -0.26 vs without +0.20 | yes |
