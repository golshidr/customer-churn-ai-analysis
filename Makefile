# Reproduce everything with `make setup data all` (or `make reproduce`).
PYTHON ?= python
CONFIG ?= configs/default.yaml
CHURN  := $(PYTHON) -m churn_analysis --config $(CONFIG)

.PHONY: help setup data all reproduce original eda benchmark decide simulate survival \
        explain fairness export serve test lint format smoke notebook clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## install pinned dependencies and the package (editable)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

data:  ## download the Telco dataset to data/raw/ (Kaggle, IBM GitHub fallback; checksum verified)
	$(CHURN) download

all:  ## run every analysis stage (results/ and figures/)
	$(CHURN) all

reproduce: data all  ## download data and run everything

original:  ## re-run the original notebook pipeline
	$(CHURN) reproduce-original
eda:  ## descriptive churn-rate figure
	$(CHURN) eda
benchmark:  ## Optuna-tuned model comparison with CV and bootstrap CIs
	$(CHURN) benchmark
decide:  ## calibration and cost-sensitive targeting
	$(CHURN) decide
simulate:  ## Monte Carlo retention-campaign simulation
	$(CHURN) simulate
survival:  ## Kaplan-Meier and Cox models of time to churn
	$(CHURN) survival
explain:  ## SHAP global, interaction and per-customer explanations
	$(CHURN) explain
fairness:  ## subgroup audit
	$(CHURN) fairness
export:  ## package the selected model for the API
	$(CHURN) export

serve:  ## start the scoring API on http://127.0.0.1:8000 (docs at /docs)
	$(CHURN) serve

notebook:  ## re-execute the original notebook in place
	cd notebooks && $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace churn_analysis.ipynb

test:  ## run the test suite
	$(PYTHON) -m pytest

smoke:  ## fast end-to-end run on synthetic data (outputs in build/smoke/)
	$(PYTHON) -m churn_analysis --config configs/smoke.yaml all

lint:  ## ruff lint + format check
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:  ## apply ruff formatting and safe fixes
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

clean:  ## remove generated artifacts that are not committed
	rm -rf build artifacts models .pytest_cache .ruff_cache
