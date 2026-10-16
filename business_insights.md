# Business insights from the customer churn analysis

> **Updated 2026-10-15.** The sections below add evidence (churn rates, SHAP directions,
> survival-model hazard ratios) to the five original insights. The original text is
> kept at the end of this file.

Evidence for each statement comes from the pipeline outputs in [`results/`](results/)
(IBM Telco sample data, 7,032 customers, 26.6% churned in the last month). Churn rates are
observed in the data; "model" refers to SHAP values of the tuned XGBoost model on the
held-out test set. All of these are associations. They show where churn concentrates, not
what an intervention would change; testing an offer on a random subset of customers is the
way to measure that.

## The five original insights, checked

| # | Insight | Observed churn rate | Model agrees | Survival model (hazard ratio, adjusted) |
|---|---|---|---|---|
| 1 | Short tenure → more churn | 47.7% in the first 12 months vs 17.1% after | yes | — (tenure is the time axis) |
| 2 | Higher monthly charges → more churn | 32.9% in the top quartile (> 90) vs 11.3% in the bottom (< 36) | yes | not in the model (determined by services) |
| 3 | Long-term contracts → less churn | 2.8% two-year, 11.3% one-year, 42.7% month-to-month | yes | two-year 0.08 [0.06, 0.10], one-year 0.22 [0.19, 0.27] |
| 4 | Fiber optic → more churn | 41.9% fiber vs 19.0% DSL | yes | 1.50 [1.29, 1.74] vs DSL |
| 5 | Tech support and online security → less churn | 15.2% / 14.6% with vs 41.6% / 41.8% without (internet customers) | yes | 0.65 [0.57, 0.75] / 0.54 [0.47, 0.62] |

Sources: [`insight_validation.md`](results/insight_validation.md),
[`cox_summary.csv`](results/cox_summary.csv).

### 1. The first year is where customers are lost
Almost half of customers in their first 12 months churn, against one in six afterwards,
and the model's strongest single interaction is tenure × contract: short tenure matters
most for month-to-month customers. Onboarding and the first renewal points are the
natural place for retention effort.

### 2. Higher bills go with higher churn, but price is tangled with service
Monthly charges are largely set by the services a customer has (fiber costs more), and
in the random forest, permutation importance gives `MonthlyCharges` no value once the
other features are known. The data cannot separate "price is too high" from "fiber customers are different".

### 3. Contract length is the strongest signal
SHAP and permutation importance both rank contract type first, and it has the largest
hazard ratios in the Cox model.
Customers on one- and two-year contracts are both less likely to leave and, by contract
design, less able to; this is not evidence that moving someone to a longer contract would
keep them.

### 4. Fiber customers churn more even after adjustment
The fiber effect survives adjustment for contract, payment and add-ons in the Cox model.
Possible reasons (price, service quality, competition for fiber customers) are not in the
data and would need to be investigated directly.

### 5. Add-on services go with lower churn
Tech support and online security have the largest associations among add-ons. Customers
who choose them may be more engaged to begin with, so an offer of free tech support should
be tested before it is rolled out.

## Additional findings

* **Electronic check payment** has the highest raw churn rate of any payment method (45%),
  but its hazard ratio after adjustment is 1.10 [0.95, 1.28] relative to mailed check:
  most of the raw difference is explained by the contracts and services of those
  customers. Automatic payments (bank transfer, credit card) remain associated with lower
  churn after adjustment.
* **Gender has no predictive value.** The first version listed `gender_Male` among the top
  ten features; that was an artefact of impurity-based importance. Permutation importance
  is slightly negative and removing gender and senior-citizen status changes test ROC-AUC
  by −0.001.
* **Senior citizens** churn more (41% vs 24% in the test set) and are predicted less
  accurately (ROC-AUC 0.79 vs 0.85). A risk-based campaign would contact most of them;
  check that the offer is appropriate for that group.

## What to do with the model

* **Target by expected value, not a 0.5 cut-off.** With the illustrative economics in
  `configs/default.yaml`, contacting customers whose expected offer value is positive
  (about 43% of customers, 80% of churners) earns 6,252 more per 1,000 customers than
  contacting those with a predicted probability above 0.5 (paired 95% CI [2,381, 10,164]).
* **Budget around 30–50% of the base** for an offer like the assumed one; beyond that,
  offers go mostly to customers who would have stayed and the simulated savings fall
  ([`campaign_simulation.md`](results/campaign_simulation.md)).
* **The offer's acceptance rate matters more than the model.** Under the assumptions, the
  expected profit of the expected-value rule ranges from about 1,100 to 90,600 per 1,000
  customers across acceptance rates of 10–50% and incentives of 0.5–3 months of charges
  ([sensitivity grid](figures/campaign_sensitivity.png)). Measuring acceptance in a small randomised pilot is worth more than further model tuning.

## Original insights (first version)

The text below is the first version of this file. The wording is unchanged; headings
moved down one level. The tables above check each statement against the data.

This analysis uses machine learning and explainable AI (SHAP) to identify the key factors influencing customer churn in a telecommunications company.

### Key Business Insights

#### 1. Customer tenure is a strong predictor of churn
Customers who have been with the company for a shorter period are significantly more likely to leave.
This suggests that the early stages of the customer lifecycle are critical.

#### 2. Higher monthly charges increase churn probability
Customers paying higher monthly fees tend to have a higher probability of leaving the service.

#### 3. Long-term contracts reduce churn risk
Customers with two-year contracts show a much lower churn probability compared to month-to-month plans.

#### 4. Fiber optic internet users exhibit higher churn risk
Customers subscribed to fiber optic internet services appear more likely to churn.

#### 5. Additional support services reduce churn probability
Customers who subscribe to services such as technical support or online security generally show lower churn rates.
