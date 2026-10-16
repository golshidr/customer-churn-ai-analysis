"""Stage 6: subgroup performance audit.

For each attribute (gender, senior citizen, partner, dependents) and each group:
discrimination (ROC-AUC with bootstrap CI), calibration-in-the-large (observed vs mean
predicted churn), and the error rates of the recommended targeting policy (the
expected-value rule on calibrated probabilities): share contacted, true-positive rate
(churners reached) and false-positive rate (non-churners contacted).

An "unaware" ablation refits the selected model with the sensitive attributes set to a
constant, which removes them as inputs, and compares test ROC-AUC with a paired bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import plotting as P
from .data import split_xy
from .evaluate import PROBA_METRICS, paired_bootstrap_diff
from .models import make_pipeline, model_device
from .runtime import (
    artifact_path,
    best_params,
    figure_path,
    get_split,
    results_path,
    save_json,
    selected_model_name,
    to_markdown,
)


def group_metrics(y, p, target, groups, n_boot=1000, seed=0) -> pd.DataFrame:
    y, p, target, groups = map(np.asarray, (y, p, target, groups))
    rng = np.random.default_rng(seed)
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        yg, pg, tg = y[m], p[m], target[m]
        auc = roc_auc_score(yg, pg) if len(np.unique(yg)) == 2 else np.nan
        boots = []
        idx_g = np.flatnonzero(m)
        for _ in range(n_boot):
            b = rng.choice(idx_g, len(idx_g), replace=True)
            if len(np.unique(y[b])) == 2:
                boots.append(roc_auc_score(y[b], p[b]))
        lo, hi = np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan)
        rows.append(
            {
                "group": g,
                "n": int(m.sum()),
                "churn_rate": float(yg.mean()),
                "mean_predicted": float(pg.mean()),
                "roc_auc": float(auc),
                "roc_auc_ci_low": float(lo),
                "roc_auc_ci_high": float(hi),
                "contacted_share": float(tg.mean()),
                "tpr": float(tg[yg == 1].mean()) if (yg == 1).any() else np.nan,
                "fpr": float(tg[yg == 0].mean()) if (yg == 0).any() else np.nan,
                "precision": float(yg[tg].mean()) if tg.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def disparity(table: pd.DataFrame) -> dict:
    """Max-minus-min gaps across groups (and min/max ratio for the contacted share)."""
    return {
        "roc_auc_gap": float(table.roc_auc.max() - table.roc_auc.min()),
        "tpr_gap": float(table.tpr.max() - table.tpr.min()),
        "fpr_gap": float(table.fpr.max() - table.fpr.min()),
        "contacted_ratio": float(table.contacted_share.min() / table.contacted_share.max())
        if table.contacted_share.max() > 0
        else float("nan"),
        "calibration_gap_max": float((table.churn_rate - table.mean_predicted).abs().max()),
    }


def auc_gap_ci(y, p, groups, n_boot=1000, seed=0) -> tuple[float, float]:
    """Bootstrap CI for the AUC difference between the two groups of a binary attribute."""
    y, p, groups = map(np.asarray, (y, p, groups))
    levels = sorted(pd.unique(groups))
    if len(levels) != 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb, pb, gb = y[b], p[b], groups[b]
        a = [yb[gb == lv] for lv in levels]
        if any(len(np.unique(x)) < 2 for x in a):
            continue
        diffs.append(
            roc_auc_score(a[1], pb[gb == levels[1]]) - roc_auc_score(a[0], pb[gb == levels[0]])
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def run_fairness(cfg: dict) -> pd.DataFrame:
    fcfg, seed = cfg["fairness"], cfg["seed"]
    n_boot = min(cfg["evaluation"]["n_bootstrap"], 1000)
    train, test = get_split(cfg)
    cal = pd.read_csv(artifact_path(cfg, "test_calibrated.csv"))
    assert (cal.customerID.to_numpy() == test.customerID.to_numpy()).all()
    y, p, target = (
        cal.y.to_numpy(),
        cal.p_calibrated.to_numpy(),
        cal.target_ev_rule.to_numpy().astype(bool),
    )

    tables, gaps = [], []
    for attr in fcfg["attributes"]:
        t = group_metrics(y, p, target, test[attr], n_boot=n_boot, seed=seed)
        t.insert(0, "attribute", attr)
        tables.append(t)
        lo, hi = auc_gap_ci(y, p, test[attr], n_boot=n_boot, seed=seed)
        levels = sorted(test[attr].unique())
        gaps.append(
            {
                "attribute": attr,
                "comparison": f"{levels[1]} - {levels[0]}" if len(levels) == 2 else "",
                **disparity(t),
                "roc_auc_gap_ci_low": lo,
                "roc_auc_gap_ci_high": hi,
            }
        )
    sub = pd.concat(tables, ignore_index=True)
    gaps = pd.DataFrame(gaps)
    sub.to_csv(results_path(cfg, "fairness_subgroups.csv"), index=False)
    gaps.to_csv(results_path(cfg, "fairness_disparities.csv"), index=False)
    show = sub.assign(
        **{
            "ROC-AUC [95% CI]": [
                f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
                for a, lo, hi in zip(sub.roc_auc, sub.roc_auc_ci_low, sub.roc_auc_ci_high)
            ],
            "Churn rate": sub.churn_rate.map("{:.1%}".format),
            "Mean predicted": sub.mean_predicted.map("{:.1%}".format),
            "Contacted": sub.contacted_share.map("{:.1%}".format),
            "TPR": sub.tpr.map("{:.2f}".format),
            "FPR": sub.fpr.map("{:.2f}".format),
        }
    )
    results_path(cfg, "fairness_subgroups.md").write_text(
        to_markdown(
            show[
                [
                    "attribute",
                    "group",
                    "n",
                    "Churn rate",
                    "Mean predicted",
                    "ROC-AUC [95% CI]",
                    "Contacted",
                    "TPR",
                    "FPR",
                ]
            ]
        )
    )

    unaware = unaware_ablation(cfg, train, test, n_boot)
    save_json(unaware, results_path(cfg, "fairness_unaware.json"))
    _plot(sub, cfg)
    return sub


def unaware_ablation(cfg: dict, train: pd.DataFrame, test: pd.DataFrame, n_boot: int) -> dict:
    selected = selected_model_name(cfg)
    params = best_params(cfg)[selected]
    device = model_device(selected, cfg)
    X_tr, y_tr = split_xy(train)
    X_te, y_te = split_xy(test)
    sensitive = cfg["fairness"]["sensitive_features"]
    neutral_tr, neutral_te = X_tr.copy(), X_te.copy()
    for col in sensitive:
        constant = X_tr[col].mode().iloc[0]
        neutral_tr[col] = constant
        neutral_te[col] = constant
    full = make_pipeline(
        selected, params, device=device, n_jobs=cfg["compute"]["n_jobs"], seed=cfg["seed"]
    )
    blind = make_pipeline(
        selected, params, device=device, n_jobs=cfg["compute"]["n_jobs"], seed=cfg["seed"]
    )
    p_full = full.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    p_blind = blind.fit(neutral_tr, y_tr).predict_proba(neutral_te)[:, 1]
    diff = paired_bootstrap_diff(
        y_te, p_blind, p_full, PROBA_METRICS["roc_auc"], n_boot=n_boot, seed=cfg["seed"]
    )
    return {
        "model": selected,
        "removed": sensitive,
        "roc_auc_full": float(roc_auc_score(y_te, p_full)),
        "roc_auc_without_sensitive": float(roc_auc_score(y_te, p_blind)),
        "difference": diff,
    }


def _plot(sub: pd.DataFrame, cfg: dict) -> None:
    P.apply_style()
    labels = [f"{a}: {g}" for a, g in zip(sub.attribute, sub.group)]
    y = np.arange(len(sub))[::-1]
    fig, axes = P.plt.subplots(1, 3, figsize=(13, 0.42 * len(sub) + 1.6), sharey=True)
    ax = axes[0]
    ax.hlines(y, sub.roc_auc_ci_low, sub.roc_auc_ci_high, color=P.BLUE, lw=2)
    ax.scatter(sub.roc_auc, y, color=P.BLUE, s=36, zorder=3, edgecolor="white", linewidth=1.2)
    ax.set(xlabel="ROC-AUC (95% CI)", title="Discrimination")
    ax.set_yticks(y, labels)
    ax = axes[1]
    ax.scatter(
        sub.churn_rate,
        y + 0.12,
        color=P.INK_2,
        marker="o",
        s=34,
        label="Observed churn rate",
        zorder=3,
    )
    ax.scatter(
        sub.mean_predicted,
        y - 0.12,
        color=P.ORANGE,
        marker="D",
        s=30,
        label="Mean predicted",
        zorder=3,
    )
    ax.set(xlabel="Churn rate", title="Calibration in the large")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    ax = axes[2]
    ax.scatter(
        sub.tpr,
        y + 0.12,
        color=P.AQUA,
        marker="o",
        s=34,
        label="TPR (churners contacted)",
        zorder=3,
    )
    ax.scatter(
        sub.fpr,
        y - 0.12,
        color=P.MAGENTA,
        marker="s",
        s=30,
        label="FPR (stayers contacted)",
        zorder=3,
    )
    ax.set(xlabel="Rate", title="Targeting policy error rates")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    for a in axes:
        a.grid(axis="y", visible=False)
        for k in range(1, len(sub)):
            if sub.attribute.iloc[k] != sub.attribute.iloc[k - 1]:
                a.axhline(y[k] + 0.5, color=P.GRID, lw=1)
    for a in axes[1:]:
        a.xaxis.set_major_formatter(P.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle(
        "Subgroup audit of the selected model (test set)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    P.savefig(fig, figure_path(cfg, "fairness_subgroups.png"))
