# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-16

First stable release. The original exploratory notebook (random forest and SHAP) is now an
installable, tested package that reproduces every result and figure with `make reproduce`.

### Added

- Data download from Kaggle with a fallback to IBM's public GitHub copy, verified by a
  SHA-256 checksum of the normalised file, plus schema checks and cleaning.
- Model comparison of logistic regression, random forest, XGBoost, LightGBM and CatBoost,
  each tuned with Optuna (60 trials, 5-fold stratified cross-validation) and evaluated on a
  held-out test set with bootstrap and paired-bootstrap confidence intervals.
- Calibration checks and cost-sensitive targeting: an expected-value contact rule, a
  profit-optimal threshold and a comparison of contact policies.
- Monte Carlo retention-campaign simulation and a sensitivity grid over the offer's
  acceptance rate and incentive.
- Survival analysis: Kaplan-Meier curves, log-rank tests and a penalised Cox model with
  proportional-hazards tests.
- SHAP explanations: global importance, pairwise interactions, a comparison with impurity
  and permutation importance, per-customer reason-code reports, and a check of each claim
  in `business_insights.md` against the data.
- Subgroup audit across gender, senior-citizen, partner and dependents groups, with an
  ablation that removes gender and senior-citizen status from the inputs.
- FastAPI scoring service that returns a churn probability, a recommendation and the top
  reasons for each customer, and a Dockerfile for it.
- `churn <stage>` command-line interface, a Makefile target for every stage and YAML
  configuration (`configs/default.yaml`, `configs/smoke.yaml`).
- pytest suite on synthetic data, ruff linting and GitHub Actions CI on Python 3.11, 3.12
  and 3.13.
- Pinned dependencies in `requirements.txt` and `requirements-dev.txt`.
- `docs/RESULTS.md` with the full results, limitations and references.

### Changed

- Models are compared on ROC-AUC, PR-AUC and Brier score with confidence intervals instead
  of accuracy at a 0.5 threshold (a model that predicts nobody churns already reaches 73%
  accuracy).
- The README is a short overview; the detailed results moved to `docs/RESULTS.md`.
- `business_insights.md` now gives churn rates, SHAP directions and hazard ratios for each
  of the original five insights.

### Fixed

- The SHAP summary figure was wrong. With SHAP >= 0.45, `TreeExplainer.shap_values()` on a
  scikit-learn classifier returns one array of shape (customers, features, classes), which
  `summary_plot` reads as interaction values, so the figure showed only two features. The
  notebook now keeps the positive class.
- The notebook did not run top to bottom. Its cells had been executed out of order; run in
  order, one-hot encoding renamed `Churn` and the next cell raised `KeyError`. The label is
  now mapped before encoding and the notebook is re-executed; the random-forest numbers are
  unchanged.
- The hard-coded Colab data path is replaced by the download script, and dependencies are
  pinned.
- The notebook cells that ran `pip install` or overwrote `README.md`, `requirements.txt`
  and `business_insights.md` are kept as non-executing cells.

[1.0.0]: https://github.com/golshidr/customer-churn-ai-analysis/releases/tag/v1.0.0
