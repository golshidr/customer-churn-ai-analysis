"""Stage 4: survival analysis of time to churn.

``tenure`` (months) is the time a customer has been observed. Customers with
``Churn = 1`` left in the last month, so their tenure is an observed time-to-churn; for the
others it is right-censored (they were still customers when the snapshot was taken).

* Kaplan-Meier curves and log-rank tests by contract, internet service and payment method.
* A penalised Cox proportional-hazards model fitted on the training split and scored on
  the test split with Harrell's concordance index. ``TotalCharges`` is excluded because it
  is roughly ``tenure * MonthlyCharges`` and so leaks the duration; ``MonthlyCharges`` is
  excluded because it is almost fully determined by the service indicators.
* Caveat: the data is a one-month snapshot (``Churn`` = left within the last month), not a
  cohort followed from sign-up. Customers who left before that month are not in the data,
  so treating ``tenure`` as follow-up time is an approximation. Hazard ratios and group
  comparisons are informative; absolute retention curves should not be read as the
  retention of a real sign-up cohort.
* The proportional-hazards assumption is tested (Schoenfeld residuals, rank transform);
  a Cox model stratified by contract is reported as a robustness check.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time

from . import plotting as P
from .runtime import figure_path, get_split, load_data, results_path, save_json, to_markdown

log = logging.getLogger("churn_analysis")

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
BINARY = {
    "gender": "Male",
    "SeniorCitizen": "Yes",
    "Partner": "Yes",
    "Dependents": "Yes",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "MultipleLines": "Yes",
}
#: column -> (reference level, other levels encoded as indicators)
MULTI = {
    "InternetService": ("DSL", ["Fiber optic", "No"]),
    "Contract": ("Month-to-month", ["One year", "Two year"]),
    "PaymentMethod": (
        "Mailed check",
        ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
    ),
}


def survival_design(df: pd.DataFrame, stratify_contract: bool = False) -> pd.DataFrame:
    """Covariate matrix for Cox models plus ``duration`` and ``event`` columns.

    Add-on services are coded "has the service" (Yes vs anything else), so the
    "No internet service" levels do not duplicate ``InternetService = No``.
    """
    out = pd.DataFrame({"duration": df["tenure"].astype(float), "event": df["Churn"].astype(int)})
    for col, level in BINARY.items():
        out[f"{col}={level}"] = (df[col] == level).astype(float)
    for col in ADDONS:
        out[f"{col}=Yes"] = (df[col] == "Yes").astype(float)
    for col, (_, levels) in MULTI.items():
        if stratify_contract and col == "Contract":
            out["Contract"] = df["Contract"].to_numpy()
            continue
        for level in levels:
            out[f"{col}={level}"] = (df[col] == level).astype(float)
    return out


def km_table(df: pd.DataFrame, group: str, horizon: float) -> pd.DataFrame:
    rows = []
    for level, sub in df.groupby(group):
        kmf = KaplanMeierFitter().fit(sub.tenure, sub.Churn)
        rows.append(
            {
                "variable": group,
                "group": level,
                "n": len(sub),
                "events": int(sub.Churn.sum()),
                "survival_12m": float(kmf.survival_function_at_times(12).iloc[0]),
                "survival_24m": float(kmf.survival_function_at_times(24).iloc[0]),
                "survival_72m": float(kmf.survival_function_at_times(72).iloc[0]),
                "median_months": float(kmf.median_survival_time_),
                f"rmst_{int(horizon)}m": float(restricted_mean_survival_time(kmf, t=horizon)),
            }
        )
    return pd.DataFrame(rows)


def run_survival(cfg: dict) -> dict:
    scfg = cfg["survival"]
    df = load_data(cfg)
    train, test = get_split(cfg)

    # ---- Kaplan-Meier + log-rank on all customers ----------------------------------
    km = pd.concat(
        [km_table(df, g, scfg["rmst_horizon"]) for g in scfg["groups"]], ignore_index=True
    )
    km.to_csv(results_path(cfg, "survival_km_summary.csv"), index=False)
    lr_rows = []
    for g in scfg["groups"]:
        res = multivariate_logrank_test(df.tenure, df[g], df.Churn)
        lr_rows.append(
            {
                "variable": g,
                "test_statistic": float(res.test_statistic),
                "degrees_of_freedom": int(df[g].nunique() - 1),
                "p_value": float(res.p_value),
            }
        )
    logrank = pd.DataFrame(lr_rows)
    logrank.to_csv(results_path(cfg, "survival_logrank.csv"), index=False)

    # ---- Cox PH on train, C-index on test ------------------------------------------
    d_tr, d_te = survival_design(train), survival_design(test)
    cph = CoxPHFitter(penalizer=scfg["penalizer"]).fit(d_tr, "duration", "event")
    c_train = float(cph.concordance_index_)
    c_test = float(cph.score(d_te, scoring_method="concordance_index"))
    summ = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={
            "exp(coef)": "hazard_ratio",
            "exp(coef) lower 95%": "hr_ci_low",
            "exp(coef) upper 95%": "hr_ci_high",
            "p": "p_value",
        }
    )
    summ.index.name = "covariate"
    summ.to_csv(results_path(cfg, "cox_summary.csv"))

    ph = proportional_hazard_test(cph, d_tr, time_transform="rank").summary
    ph.index.name = "covariate"
    ph = ph.reset_index()[["covariate", "test_statistic", "p"]].rename(columns={"p": "p_value"})
    n_cov = len(ph)
    ph["violates_at_0.05_bonferroni"] = ph.p_value < 0.05 / n_cov
    ph.to_csv(results_path(cfg, "cox_ph_test.csv"), index=False)

    s_tr, s_te = survival_design(train, True), survival_design(test, True)
    cph_s = CoxPHFitter(penalizer=scfg["penalizer"]).fit(
        s_tr, "duration", "event", strata=["Contract"]
    )
    c_test_strat = float(cph_s.score(s_te, scoring_method="concordance_index"))

    summary = {
        "n_customers": int(len(df)),
        "n_events": int(df.Churn.sum()),
        "cox_c_index_train": c_train,
        "cox_c_index_test": c_test,
        "cox_stratified_by_contract_c_index_test": c_test_strat,
        "ph_violations_bonferroni": ph.loc[ph["violates_at_0.05_bonferroni"], "covariate"].tolist(),
        "penalizer": scfg["penalizer"],
    }
    save_json(summary, results_path(cfg, "survival_summary.json"))
    results_path(cfg, "survival_km_summary.md").write_text(
        to_markdown(
            km.assign(
                median_months=km.median_months.map(
                    lambda v: "not reached" if np.isinf(v) else f"{v:.0f}"
                )
            )
        )
    )
    log.info("Cox C-index train %.3f test %.3f (stratified %.3f)", c_train, c_test, c_test_strat)

    _plot_km(df, scfg["groups"], logrank, cfg)
    _plot_hazard_ratios(summ, cfg)
    _plot_profiles(cph, d_tr, cfg)
    return summary


# --------------------------------------------------------------------------- figures
def _plot_km(df, groups, logrank, cfg) -> None:
    P.apply_style()
    fig, axes = P.plt.subplots(1, len(groups), figsize=(5.0 * len(groups), 4.4), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), groups):
        levels = df.groupby(g).Churn.mean().sort_values().index  # lowest churn first
        for color, level in zip(P.PALETTE, levels):
            sub = df[df[g] == level]
            kmf = KaplanMeierFitter().fit(sub.tenure, sub.Churn, label=str(level))
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_
            ax.step(
                sf.index,
                sf.iloc[:, 0],
                where="post",
                color=color,
                label=f"{level} (n={len(sub):,})",
            )
            ax.fill_between(
                ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=color, alpha=0.15, lw=0
            )
        p = logrank.loc[logrank.variable == g, "p_value"].iloc[0]
        ax.set_title(f"{g}  (log-rank p {'< 1e-16' if p < 1e-16 else f'= {p:.1e}'})", fontsize=11.5)
        ax.set_xlabel("Tenure (months)")
        ax.set_xlim(0, 72)
        ax.legend(loc="lower left", fontsize=8.8)
    np.atleast_1d(axes)[0].set_ylabel("Share of customers still active")
    np.atleast_1d(axes)[0].set_ylim(0, 1.02)
    fig.suptitle(
        "Kaplan-Meier retention curves with 95% confidence bands",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    P.savefig(fig, figure_path(cfg, "survival_km_curves.png"))


def _plot_hazard_ratios(summ: pd.DataFrame, cfg) -> None:
    P.apply_style()
    s = summ.sort_values("hazard_ratio")
    fig, ax = P.plt.subplots(figsize=(7.5, 0.33 * len(s) + 1.3))
    y = np.arange(len(s))
    # red = significantly higher hazard, blue = significantly lower, grey = CI includes 1
    colors = [
        P.RED if lo > 1 else P.BLUE if hi < 1 else P.MUTED
        for lo, hi in zip(s.hr_ci_low, s.hr_ci_high)
    ]
    ax.hlines(y, s.hr_ci_low, s.hr_ci_high, colors=colors, lw=2)
    ax.scatter(s.hazard_ratio, y, color=colors, s=40, zorder=3, edgecolor="white", linewidth=1.2)
    ax.axvline(1, color=P.INK_2, lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y, s.index)
    ax.grid(axis="y", visible=False)
    ax.set(
        xlabel="Hazard ratio (log scale, 95% CI); >1 = churns sooner",
        title="Cox model: hazard ratios for churn (training split)",
    )
    ax.xaxis.set_major_formatter(P.plt.FuncFormatter(lambda v, _: f"{v:g}"))
    P.savefig(fig, figure_path(cfg, "survival_cox_hazard_ratios.png"))


def _plot_profiles(cph: CoxPHFitter, d_tr: pd.DataFrame, cfg) -> None:
    """Predicted retention for three illustrative customer profiles."""
    base = {c: 0.0 for c in d_tr.columns if c not in ("duration", "event")}
    base.update({"PhoneService=Yes": 1.0, "PaperlessBilling=Yes": 1.0})
    profiles = {
        "Month-to-month, fiber, electronic check, no add-ons": {
            "InternetService=Fiber optic": 1.0,
            "PaymentMethod=Electronic check": 1.0,
        },
        "One year, DSL, credit card, tech support + security": {
            "Contract=One year": 1.0,
            "PaymentMethod=Credit card (automatic)": 1.0,
            "TechSupport=Yes": 1.0,
            "OnlineSecurity=Yes": 1.0,
        },
        "Two year, DSL, bank transfer, tech support + security": {
            "Contract=Two year": 1.0,
            "PaymentMethod=Bank transfer (automatic)": 1.0,
            "TechSupport=Yes": 1.0,
            "OnlineSecurity=Yes": 1.0,
        },
    }
    X = pd.DataFrame([{**base, **v} for v in profiles.values()], index=list(profiles))
    sf = cph.predict_survival_function(X, times=np.arange(1, 73))
    sf.loc[0] = 1.0  # everyone is active at sign-up (no events at t = 0)
    sf = sf.sort_index()
    P.apply_style()
    fig, ax = P.plt.subplots(figsize=(8, 4.4))
    for color, name in zip(P.PALETTE, profiles):
        ax.plot(sf.index, sf[name], color=color, label=name)
        ax.text(
            72.5,
            sf[name].iloc[-1],
            f"{sf[name].iloc[-1]:.0%}",
            va="center",
            fontsize=9,
            color=P.INK_2,
        )
    ax.set(
        xlabel="Months since joining",
        ylabel="Predicted probability still active",
        ylim=(0, 1.02),
        xlim=(0, 76),
        title="Cox model: predicted retention for three example profiles",
    )
    ax.legend(loc="lower left", fontsize=9)
    P.savefig(fig, figure_path(cfg, "survival_profiles.png"))
