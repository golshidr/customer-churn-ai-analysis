# Customer churn prediction with explainable AI

[![CI](https://github.com/golshidr/customer-churn-ai-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/golshidr/customer-churn-ai-analysis/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A reproducible analysis of the IBM Telco Customer Churn data (7,043 customers) that predicts
which customers are likely to leave, explains why, and decides who is worth a retention
offer. It is built for data scientists and analysts who need churn models judged by
uncertainty, calibration and profit rather than accuracy alone, and it ships as a tested
Python package with a scoring API.

<p align="center">
  <img src="figures/profit_curve.png" width="760"
       alt="Profit curve on the test set: expected profit per 1,000 customers against the share of customers contacted by predicted risk. Tuned XGBoost beats the original random forest and random targeting, and the expected-value rule sits near the peak, above the 0.5 threshold.">
</p>

## Key results

Test-set figures are for 1,407 held-out customers; brackets are 95% confidence intervals.
Full tables are in [docs/RESULTS.md](docs/RESULTS.md).

- **Prediction.** The selected model, tuned XGBoost, reaches a test ROC-AUC of 0.841
  [0.820, 0.862] and PR-AUC of 0.657 [0.608, 0.707], against 0.813 for the original
  random-forest settings. XGBoost, LightGBM and CatBoost are indistinguishable.
- **Targeting.** Contacting customers by expected value earns 34,957 per 1,000 customers
  [28,844, 41,234] under the illustrative offer economics, 6,252 more than a 0.5 threshold
  (paired bootstrap CI [2,381, 10,164]).
- **Calibration.** Tuned XGBoost is well calibrated as trained (expected calibration error
  0.024), so its probabilities feed the decision rule directly.
- **Drivers.** Contract type dominates (mean |SHAP| 0.71 log-odds), followed by tenure
  (0.40). In the Cox model a two-year contract has a hazard ratio of 0.08 [0.06, 0.10].
- **Subgroups.** Calibration holds across gender, senior-citizen, partner and dependents
  groups (within 2.4 points), but discrimination is lower for senior citizens
  (ROC-AUC 0.79 vs 0.85).

## What's inside

- **Model comparison:** logistic regression, random forest, XGBoost, LightGBM and CatBoost,
  tuned with Optuna under 5-fold cross-validation and compared with bootstrap intervals.
- **Profit-based targeting:** calibration checks, an expected-value contact rule, a Monte
  Carlo campaign simulation and a sensitivity grid over the offer's economics.
- **Survival analysis:** Kaplan-Meier curves, log-rank tests and a penalised Cox model of
  time to churn.
- **SHAP reports:** global importance, interactions, a comparison of importance methods and
  per-customer reason codes.
- **Fairness audit:** calibration, discrimination and error rates by subgroup, plus a model
  trained without gender and senior-citizen status.
- **FastAPI service:** returns a churn probability, a recommendation and the top reasons
  for each customer.
- **Docker:** a Dockerfile that packages the scoring API.
- **Tests and CI:** a pytest suite on synthetic data and ruff, run by GitHub Actions on
  Python 3.11, 3.12 and 3.13.

## Quickstart

Requires Python 3.11 or later (tested on Linux x86_64).

```bash
git clone https://github.com/golshidr/customer-churn-ai-analysis.git
cd customer-churn-ai-analysis
python -m venv .venv && source .venv/bin/activate
make setup        # installs requirements.txt and requirements-dev.txt, then the package
```

Then download the data and run every stage (about 30 minutes on 8 CPU threads and one GPU,
most of it model tuning):

```bash
make reproduce    # same as: make data all
```

## Reproducing the results

All committed results and figures come from `make reproduce` with `configs/default.yaml`
(seed 42). Each stage can also be run on its own:

| Target | Stage |
|---|---|
| `make original` | re-run the original notebook pipeline |
| `make eda` | churn rates by segment |
| `make benchmark` | Optuna tuning and model comparison with bootstrap intervals |
| `make decide` | calibration and cost-sensitive targeting |
| `make simulate` | Monte Carlo campaign simulation and sensitivity grid |
| `make survival` | Kaplan-Meier, log-rank and Cox models |
| `make explain` | SHAP explanations and per-customer reports |
| `make fairness` | subgroup audit |
| `make export`, `make serve` | package the model and start the API at http://127.0.0.1:8000/docs |

`make smoke` runs every stage on synthetic data in about a minute, and `make test` and
`make lint` run the checks CI runs. `make help` lists all targets. XGBoost uses a GPU when
one is visible; set `compute.device: cpu` in the config to force CPU. Outputs per stage
and an example API request are in [docs/RESULTS.md](docs/RESULTS.md#reproducing-the-results).

## Documentation

- [docs/RESULTS.md](docs/RESULTS.md): full results, tables, figures, limitations and references
- [CHANGELOG.md](CHANGELOG.md): release history, including the fixes to the original notebook
- [business_insights.md](business_insights.md): the business insights, checked against the data
- [notebooks/](notebooks/churn_analysis.ipynb): the original exploratory notebook, fixed and re-executed

## Project structure

```
.
├── src/churn_analysis/   # package: data, models, tuning, decisions, survival, SHAP, fairness, API
├── configs/              # default.yaml (full run), smoke.yaml (synthetic-data run)
├── tests/                # pytest suite (synthetic data, CPU only)
├── results/              # CSV, JSON and Markdown outputs of the full run
├── figures/              # figures of the full run
├── notebooks/            # original analysis notebook
├── docs/                 # detailed results
├── scripts/              # data download
├── Dockerfile, Makefile, pyproject.toml
└── requirements.txt, requirements-dev.txt
```

## Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data on Kaggle; 7,043 customers,
21 columns), downloaded by `make data` with a fallback to IBM's public GitHub copy and verified by checksum.
Kaggle lists the license as "Data files © Original Authors"; the data is not redistributed here.

## License

The code is released under the [MIT License](LICENSE).
