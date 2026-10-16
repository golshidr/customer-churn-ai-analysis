import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn_analysis.evaluate import (
    bootstrap_ci,
    compute_metrics,
    expected_calibration_error,
    metrics_with_ci,
    paired_bootstrap_diff,
    reliability_table,
)

rng = np.random.default_rng(0)
Y = rng.binomial(1, 0.3, 2000)
P_GOOD = np.clip(Y * 0.6 + rng.uniform(0, 0.4, 2000), 0, 1)
P_RAND = rng.uniform(0, 1, 2000)


def test_metrics_on_perfect_scores():
    m = compute_metrics(Y, Y.astype(float))
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0 and m["brier"] == 0.0


def test_ece_is_small_for_calibrated_and_large_for_shifted():
    p = rng.uniform(0, 1, 20000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_bootstrap_ci_brackets_point_estimate():
    point, lo, hi = bootstrap_ci(Y, P_GOOD, roc_auc_score, n_boot=300, seed=1)
    assert lo <= point <= hi
    assert hi - lo < 0.05


def test_paired_difference_detects_better_model():
    d = paired_bootstrap_diff(Y, P_GOOD, P_RAND, roc_auc_score, n_boot=300, seed=1)
    assert d["diff"] > 0.3 and d["ci_low"] > 0 and d["p_value"] < 0.01
    same = paired_bootstrap_diff(Y, P_GOOD, P_GOOD, roc_auc_score, n_boot=100, seed=1)
    assert same["diff"] == pytest.approx(0.0)


def test_metrics_with_ci_has_interval_keys():
    m = metrics_with_ci(Y, P_GOOD, n_boot=100)
    for k in ("roc_auc", "pr_auc", "brier"):
        assert m[f"{k}_ci_low"] <= m[k] <= m[f"{k}_ci_high"]


def test_reliability_table_counts_all_rows():
    t = reliability_table(Y, P_RAND, n_bins=10)
    assert t.n.sum() == len(Y)
