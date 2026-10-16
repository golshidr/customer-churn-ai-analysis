"""End-to-end run of every stage on synthetic data in a temporary directory."""

import json

import pytest

from churn_analysis.cli import STAGES, run_stage
from churn_analysis.config import REPO_ROOT, load_config


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return load_config(
        REPO_ROOT / "configs" / "smoke.yaml",
        overrides={
            "paths": {k: str(out / k) for k in ("results", "figures", "models", "artifacts")},
            "tuning": {"n_trials": 2, "models": ["logreg", "xgboost"]},
            "evaluation": {"n_bootstrap": 50},
            "campaign": {"n_simulations": 30, "budget_fractions": [0.1, 0.3, 1.0]},
            "calibration": {"methods": ["sigmoid"]},
        },
    )


def test_all_stages_run(cfg):
    from pathlib import Path

    for d in ("results", "figures", "models", "artifacts"):
        Path(cfg["paths"][d]).mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        run_stage(stage, cfg)
    res = Path(cfg["paths"]["results"])
    selected = json.loads((res / "selected_model.json").read_text())
    assert selected["selected_model"] in {"logreg", "xgboost", "rf_original"}
    for name in (
        "benchmark.csv",
        "decision_policies.csv",
        "campaign_simulation.csv",
        "survival_summary.json",
        "explain_summary.json",
        "fairness_subgroups.csv",
        "api_example.json",
        "original_baseline.json",
    ):
        assert (res / name).exists(), name
    figs = list(Path(cfg["paths"]["figures"]).glob("*.png"))
    assert len(figs) >= 15
