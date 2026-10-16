"""Metrics, bootstrap confidence intervals and calibration diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

MetricFn = Callable[[np.ndarray, np.ndarray], float]

#: Threshold-free metrics on predicted probabilities.
PROBA_METRICS: dict[str, MetricFn] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
    "log_loss": lambda y, p: log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum over bins of |mean(p) - mean(y)| weighted by bin size."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    out = {name: float(fn(y, p)) for name, fn in PROBA_METRICS.items()}
    out["ece"] = expected_calibration_error(y, p)
    out.update(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
    )
    return out


def _bootstrap_indices(n: int, n_boot: int, seed: int, y: np.ndarray | None = None):
    """Yield bootstrap index arrays; resamples that contain a single class are skipped."""
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < n_boot:
        idx = rng.integers(0, n, n)
        if y is not None and len(np.unique(y[idx])) < 2:
            continue
        produced += 1
        yield idx


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI of ``metric(y, p)``."""
    y, p = np.asarray(y), np.asarray(p)
    stats = [metric(y[i], p[i]) for i in _bootstrap_indices(len(y), n_boot, seed, y)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the difference ``metric(a) - metric(b)`` on the same resampled rows.

    ``p_value`` is the two-sided bootstrap p-value for "no difference" (share of resamples
    on the other side of zero, doubled).
    """
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = np.array(
        [
            metric(y[i], p_a[i]) - metric(y[i], p_b[i])
            for i in _bootstrap_indices(len(y), n_boot, seed, y)
        ]
    )
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(metric(y, p_a) - metric(y, p_b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p_value)),
    }


def metrics_with_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0, threshold: float = 0.5
) -> dict[str, float]:
    """All point metrics plus bootstrap CIs for ROC-AUC, PR-AUC and Brier score."""
    out = compute_metrics(y, p, threshold)
    for name in ("roc_auc", "pr_auc", "brier"):
        _, lo, hi = bootstrap_ci(y, p, PROBA_METRICS[name], n_boot=n_boot, seed=seed)
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = lo, hi
    return out


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed churn rate per equal-width probability bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append(
                {
                    "bin": b,
                    "n": int(mask.sum()),
                    "mean_pred": p[mask].mean(),
                    "observed": y[mask].mean(),
                }
            )
    return pd.DataFrame(rows)
