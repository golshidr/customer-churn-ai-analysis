"""Stage 2: probability calibration and cost-sensitive targeting.

Economics of one retention offer (all amounts per customer; see ``business`` in the config):

* ``V`` value of keeping the customer = ``horizon_months * MonthlyCharges``
* ``d`` incentive = ``incentive_months * MonthlyCharges``, paid only if the offer is accepted
* ``f`` contact cost, paid for every contacted customer
* ``g`` probability that a would-be churner accepts the offer and stays

A contacted churner is worth ``g * (V - d) - f`` in expectation. A contacted non-churner
would have stayed anyway; we conservatively assume they take the incentive, so they cost
``d + f``. For a customer with churn probability ``p`` the expected value of contacting is::

    EV = p * g * (V - d) - (1 - p) * d - f

and the per-customer rule "contact if EV > 0" is only sensible when ``p`` is calibrated.
This is the instance-dependent version of the expected-profit view of churn models in
Verbraken, Verbeke & Baesens (2013), IEEE TKDE 25(5), doi:10.1109/TKDE.2012.50.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import plotting as P
from .data import split_xy
from .evaluate import compute_metrics, expected_calibration_error, reliability_table
from .models import DISPLAY_NAMES, make_pipeline, model_device
from .runtime import (
    artifact_path,
    best_params,
    figure_path,
    get_split,
    results_path,
    save_json,
    save_model,
    selected_model_name,
    to_markdown,
)

log = logging.getLogger("churn_analysis")

THRESHOLD_05 = "Threshold 0.5"
EV_RULE = "Expected-value rule"


# --------------------------------------------------------------------------- economics
def customer_economics(monthly_charges, business: dict) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(monthly_charges, dtype=float)
    return business["horizon_months"] * m, business["incentive_months"] * m


def expected_offer_value(p, value, incentive, contact_cost: float, acceptance: float) -> np.ndarray:
    """Expected incremental profit of contacting each customer (see module docstring)."""
    p = np.asarray(p, dtype=float)
    return p * acceptance * (value - incentive) - (1 - p) * incentive - contact_cost


def campaign_profit(y, target, value, incentive, contact_cost: float, acceptance: float) -> float:
    """Profit of contacting ``target`` customers given their true outcomes ``y``.

    The expectation is over offer acceptance only; ``y`` is observed.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(target, dtype=bool)
    per = y * acceptance * (value - incentive) - (1 - y) * incentive - contact_cost
    return float(per[t].sum())


def profit_curve(y, score, value, incentive, contact_cost: float, acceptance: float):
    """Cumulative profit when contacting customers in decreasing order of ``score``.

    Returns ``(fraction_contacted, profit)`` arrays of length ``n + 1`` (starting at 0).
    """
    y = np.asarray(y, dtype=float)
    order = np.argsort(-np.asarray(score), kind="stable")
    per = (y * acceptance * (value - incentive) - (1 - y) * incentive - contact_cost)[order]
    profit = np.concatenate([[0.0], np.cumsum(per)])
    frac = np.arange(len(y) + 1) / len(y)
    return frac, profit


def best_threshold(
    y, p, value, incentive, contact_cost: float, acceptance: float, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Global probability threshold that maximises campaign profit on (y, p)."""
    grid = np.linspace(0.0, 1.0, 201) if grid is None else grid
    profits = [
        campaign_profit(y, np.asarray(p) >= t, value, incentive, contact_cost, acceptance)
        for t in grid
    ]
    i = int(np.argmax(profits))
    return float(grid[i]), float(profits[i])


def paired_profit_diff(
    y, target_a, target_b, value, incentive, contact_cost, acceptance, n_boot=2000, seed=0
) -> dict:
    """Profit difference (a - b) per 1,000 customers, bootstrapping the same customers."""
    y, target_a, target_b, value, incentive = map(
        np.asarray, (y, target_a, target_b, value, incentive)
    )
    per = y * acceptance * (value - incentive) - (1 - y) * incentive - contact_cost
    diff = per * target_a - per * target_b
    n = len(y)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, n, n)].sum() * 1000 / n for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "diff": float(diff.sum() * 1000 / n),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))),
    }


def bootstrap_profit(y, target, value, incentive, contact_cost, acceptance, n_boot=2000, seed=0):
    """Profit per 1,000 customers with a percentile bootstrap 95% CI (resampling customers)."""
    y, target, value, incentive = map(np.asarray, (y, target, value, incentive))
    n = len(y)
    per = (y * acceptance * (value - incentive) - (1 - y) * incentive - contact_cost) * target
    rng = np.random.default_rng(seed)
    boots = [per[rng.integers(0, n, n)].sum() * 1000 / n for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return per.sum() * 1000 / n, float(lo), float(hi)


# --------------------------------------------------------------------------- stage
def run_decision(cfg: dict) -> pd.DataFrame:
    seed, n_jobs = cfg["seed"], cfg["compute"]["n_jobs"]
    biz = cfg["business"]
    selected = selected_model_name(cfg)
    device = model_device(selected, cfg)
    params = best_params(cfg)[selected]
    train, test = get_split(cfg)
    X_tr, y_tr = split_xy(train)
    X_te, y_te = split_xy(test)
    oof_all = pd.read_csv(artifact_path(cfg, "oof_train.csv"))
    test_all = pd.read_csv(artifact_path(cfg, "test_predictions.csv"))
    cv = StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=seed)

    # ---- calibration -------------------------------------------------------------
    variants = {"uncalibrated": (oof_all[selected].to_numpy(), test_all[selected].to_numpy())}
    fitted = {}
    for method in cfg["calibration"]["methods"]:

        def make(method=method):
            return CalibratedClassifierCV(
                make_pipeline(selected, params, device=device, n_jobs=n_jobs, seed=seed),
                method=method,
                cv=cv,
            )

        oof_cal = cross_val_predict(make(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        model = make().fit(X_tr, y_tr)
        variants[method] = (oof_cal, model.predict_proba(X_te)[:, 1])
        fitted[method] = model

    rows = []
    for name in [c for c in test_all.columns if c not in ("customerID", "y", "dummy")]:
        m = compute_metrics(y_te, test_all[name])
        rows.append(
            {
                "model": name,
                "calibration": "uncalibrated",
                "oof_brier": float(np.mean((oof_all[name] - oof_all.y) ** 2)),
                **m,
            }
        )
    for method in cfg["calibration"]["methods"]:
        oof_p, te_p = variants[method]
        m = compute_metrics(y_te, te_p)
        rows.append(
            {
                "model": selected,
                "calibration": method,
                "oof_brier": float(np.mean((oof_p - y_tr.to_numpy()) ** 2)),
                **m,
            }
        )
    calib = pd.DataFrame(rows)
    calib.to_csv(results_path(cfg, "calibration.csv"), index=False)

    # Choose the calibration variant on training OOF Brier score (never on test).
    sel_rows = calib[calib.model == selected].sort_values("oof_brier")
    chosen = sel_rows.iloc[0]["calibration"]
    oof_p, p_te = variants[chosen]
    p_uncal = variants["uncalibrated"][1]
    if chosen != "uncalibrated":
        save_model(fitted[chosen], cfg, f"{selected}_calibrated")
    log.info("calibration chosen on OOF Brier: %s", chosen)

    # ---- policies ----------------------------------------------------------------
    v_tr, d_tr = customer_economics(X_tr.MonthlyCharges, biz)
    v_te, d_te = customer_economics(X_te.MonthlyCharges, biz)
    f, g = biz["contact_cost"], biz["acceptance_rate"]
    thr, _ = best_threshold(y_tr, oof_p, v_tr, d_tr, f, g)
    ev_cal = expected_offer_value(p_te, v_te, d_te, f, g)
    ev_uncal = expected_offer_value(p_uncal, v_te, d_te, f, g)
    y = y_te.to_numpy()
    policies = {
        "No campaign": np.zeros(len(y), bool),
        "Contact everyone": np.ones(len(y), bool),
        THRESHOLD_05: p_uncal >= 0.5,
        f"Profit-optimal threshold {thr:.3f} (chosen on train OOF)": p_te >= thr,
        "Expected-value rule, uncalibrated scores": ev_uncal > 0,
        f"{EV_RULE} ({chosen} probabilities)": ev_cal > 0,
        "Oracle (knows who churns)": (y == 1) & (g * (v_te - d_te) - f > 0),
    }
    if chosen == "uncalibrated":  # identical to the chosen-probability row
        del policies["Expected-value rule, uncalibrated scores"]
    prow = []
    for name, target in policies.items():
        point, lo, hi = bootstrap_profit(
            y, target, v_te, d_te, f, g, n_boot=cfg["evaluation"]["n_bootstrap"], seed=seed
        )
        prow.append(
            {
                "policy": name,
                "contacted": int(target.sum()),
                "contacted_share": float(target.mean()),
                "churners_reached": float(y[target].sum() / max(y.sum(), 1)),
                "precision": float(y[target].mean()) if target.any() else float("nan"),
                "profit_test": campaign_profit(y, target, v_te, d_te, f, g),
                "profit_per_1000": point,
                "profit_per_1000_ci_low": lo,
                "profit_per_1000_ci_high": hi,
            }
        )
    pol = pd.DataFrame(prow)
    pol.to_csv(results_path(cfg, "decision_policies.csv"), index=False)
    show = pol.assign(
        **{
            "Contacted": [f"{c} ({s:.0%})" for c, s in zip(pol.contacted, pol.contacted_share)],
            "Churners reached": pol.churners_reached.map("{:.0%}".format),
            "Precision": pol.precision.map(lambda v: "" if np.isnan(v) else f"{v:.2f}"),
            "Profit per 1,000 customers [95% CI]": [
                f"{p:,.0f} [{lo:,.0f}, {hi:,.0f}]"
                for p, lo, hi in zip(
                    pol.profit_per_1000, pol.profit_per_1000_ci_low, pol.profit_per_1000_ci_high
                )
            ],
        }
    )
    results_path(cfg, "decision_policies.md").write_text(
        to_markdown(
            show[
                [
                    "policy",
                    "Contacted",
                    "Churners reached",
                    "Precision",
                    "Profit per 1,000 customers [95% CI]",
                ]
            ].rename(columns={"policy": "Policy"})
        )
    )
    pd.DataFrame(
        {
            "customerID": test.customerID,
            "y": y,
            "p_uncalibrated": p_uncal,
            "p_calibrated": p_te,
            "value": v_te,
            "incentive": d_te,
            "expected_value": ev_cal,
            "target_ev_rule": ev_cal > 0,
        }
    ).to_csv(artifact_path(cfg, "test_calibrated.csv"), index=False)
    ev_name = f"{EV_RULE} ({chosen} probabilities)"
    paired = paired_profit_diff(
        y,
        policies[ev_name],
        policies[THRESHOLD_05],
        v_te,
        d_te,
        f,
        g,
        n_boot=cfg["evaluation"]["n_bootstrap"],
        seed=seed,
    )
    summary = {
        "selected_model": selected,
        "ev_rule_vs_threshold_05_profit_per_1000": paired,
        "calibration_chosen": chosen,
        "profit_optimal_threshold": thr,
        "assumptions": biz,
        "test_customers": int(len(y)),
        "ece_uncalibrated": expected_calibration_error(y, p_uncal),
        "ece_chosen": expected_calibration_error(y, p_te),
    }
    save_json(summary, results_path(cfg, "decision_summary.json"))

    _plot_calibration(y, variants, selected, chosen, cfg)
    _plot_profit(y, v_te, d_te, f, g, p_te, test_all, pol, selected, chosen, cfg)
    return pol


# --------------------------------------------------------------------------- figures
def _plot_calibration(y, variants, selected, chosen, cfg) -> None:
    P.apply_style()
    fig, (ax1, ax2) = P.plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"uncalibrated": P.MUTED, "sigmoid": P.BLUE, "isotonic": P.ORANGE}
    ax1.plot([0, 1], [0, 1], color=P.AXIS, lw=1)
    for name, (_, p) in variants.items():
        t = reliability_table(y, p, cfg["calibration"]["n_bins"])
        t = t[t.n >= 10]
        ece = expected_calibration_error(y, p, cfg["calibration"]["n_bins"])
        ax1.plot(
            t.mean_pred,
            t.observed,
            "-o",
            ms=5,
            color=colors.get(name, P.GREEN),
            mec="white",
            mew=1.2,
            label=f"{name} (ECE {ece:.3f})",
        )
    ax1.set(
        xlabel="Mean predicted churn probability",
        ylabel="Observed churn rate",
        title=f"Reliability, {DISPLAY_NAMES[selected]} (test set)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax1.legend(loc="upper left")
    bins = np.linspace(0, 1, 26)
    chosen_p = variants[chosen][1]
    for label, mask, color in (("Stayed", y == 0, P.BLUE), ("Churned", y == 1, P.ORANGE)):
        ax2.hist(chosen_p[mask], bins=bins, color=color, histtype="step", lw=2, label=label)
    ax2.set(
        xlabel=f"Predicted churn probability ({chosen})",
        ylabel="Customers",
        title="Score distribution by actual outcome",
    )
    ax2.legend()
    P.savefig(fig, figure_path(cfg, "calibration_reliability.png"))


def _plot_profit(y, v, d, f, g, p_sel, test_all, pol, selected, chosen, cfg) -> None:
    P.apply_style()
    fig, ax = P.plt.subplots(figsize=(8.5, 4.8))
    n = len(y)
    curves = [(selected, p_sel, f"{DISPLAY_NAMES[selected]}, {chosen} probabilities")]
    if "rf_original" in test_all and selected != "rf_original":
        curves.append(
            ("rf_original", test_all["rf_original"].to_numpy(), DISPLAY_NAMES["rf_original"])
        )
    for name, score, label in curves:
        frac, prof = profit_curve(y, score, v, d, f, g)
        ls = "--" if name == "rf_original" else "-"
        ax.plot(frac * 100, prof * 1000 / n, ls, color=P.model_color(name), label=label)
    total = campaign_profit(y, np.ones(n, bool), v, d, f, g) * 1000 / n
    ax.plot([0, 100], [0, total], ":", color=P.INK_2, lw=1.5, label="Random targeting (expected)")
    ax.axhline(0, color=P.AXIS, lw=1)
    marks = {
        f"{EV_RULE} ({chosen} probabilities)": (EV_RULE, "o", P.INK, (8, 6)),
        THRESHOLD_05: (THRESHOLD_05, "s", P.INK_2, (8, -14)),
    }
    for name, (label, marker, color, offset) in marks.items():
        row = pol[pol.policy == name].iloc[0]
        ax.plot(
            row.contacted_share * 100,
            row.profit_per_1000,
            marker,
            ms=8,
            color=color,
            mec="white",
            mew=1.5,
        )
        ax.annotate(
            label,
            (row.contacted_share * 100, row.profit_per_1000),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=P.INK_2,
        )
    ax.set(
        xlabel="Customers contacted, ranked by predicted risk (%)",
        ylabel="Expected profit per 1,000 customers",
        title="Retention-offer profit curve (test set, illustrative economics)",
    )
    ax.yaxis.set_major_formatter(P.plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    ax.legend(loc="lower right")
    P.savefig(fig, figure_path(cfg, "profit_curve.png"))
