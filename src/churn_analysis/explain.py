"""Stage 5: SHAP explanations.

* Global: beeswarm of the best tree model, SHAP aggregated per original feature, and
  dependence plots.
* Interactions: SHAP interaction values aggregated per original feature pair.
* Method comparison: impurity importance vs permutation importance vs mean |SHAP| for the
  notebook's Random Forest configuration (impurity importance favours continuous and
  high-cardinality features, which is why ``gender`` looked important in the original).
* Per-customer: reason-code reports for the highest-risk test customers, and a chart
  contrasting a high-, a medium- and a low-risk customer.
* Evidence table for each statement in ``business_insights.md``.

SHAP values of gradient-boosted models are in log-odds units; those of the Random Forest
are in probability units. One-hot columns are summed back to their original feature,
which is exact because SHAP values are additive.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from . import plotting as P
from .data import FEATURE_COLUMNS, split_xy
from .features import aggregate_by_group, feature_groups
from .models import DISPLAY_NAMES
from .runtime import (
    figure_path,
    get_split,
    load_json,
    load_model,
    results_path,
    save_json,
    to_markdown,
)

log = logging.getLogger("churn_analysis")


# --------------------------------------------------------------------------- core
def positive_class(values) -> np.ndarray:
    """SHAP values for the positive class as an (n_samples, n_features) array.

    Handles the three layouts SHAP has used for binary classifiers: a list
    ``[negative, positive]``, a 3-D array ``(n, features, 2)`` (SHAP >= 0.45 with
    scikit-learn models) and a plain 2-D array (models with a single log-odds output).
    The original notebook passed the 3-D array straight to ``summary_plot``, which then
    treated it as interaction values.
    """
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    if values.ndim == 2:
        return values
    raise ValueError(f"unexpected SHAP value shape {values.shape}")


def positive_base_value(expected) -> float:
    arr = np.atleast_1d(np.asarray(expected, dtype=float))
    return float(arr[-1])


def tree_shap(pipe, X: pd.DataFrame, interactions: bool = False):
    """SHAP values of a fitted (prep -> tree model) pipeline on raw features ``X``."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    if model.__class__.__name__ == "XGBClassifier":
        model.set_params(device="cpu")  # explain on CPU; avoids device-mismatch warnings
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    explainer = shap.TreeExplainer(model)
    values = positive_class(explainer.shap_values(Xt))
    base = positive_base_value(explainer.expected_value)
    inter = None
    if interactions:
        inter = explainer.shap_interaction_values(Xt)
        inter = np.asarray(inter[-1] if isinstance(inter, list) else inter)
        if inter.ndim == 4:
            inter = inter[..., -1]
    return values, base, Xt, inter


def aggregate_interactions(inter: np.ndarray, groups: dict[str, list[int]]) -> pd.DataFrame:
    """Mean |interaction| between original features (one-hot blocks summed)."""
    names = list(groups)
    out = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        block_a = inter[:, groups[a], :].sum(axis=1)
        for j, b in enumerate(names):
            out[i, j] = np.abs(block_a[:, groups[b]].sum(axis=1)).mean()
    return pd.DataFrame(out, index=names, columns=names)


def reason_codes(contrib: pd.Series, raw: pd.Series, k: int = 3) -> list[dict]:
    """Top-``k`` features pushing churn risk up, with their values."""
    top = contrib.sort_values(ascending=False).head(k)
    return [
        {"feature": f, "value": _fmt_value(raw[f]), "contribution": float(c)}
        for f, c in top.items()
        if c > 0
    ]


def _fmt_value(v) -> str:
    if isinstance(v, float | np.floating):
        return f"{v:,.2f}"
    return str(v)


# --------------------------------------------------------------------------- stage
def run_explain(cfg: dict) -> dict:
    ecfg = cfg["explain"]
    sel = load_json(results_path(cfg, "selected_model.json"))
    name = sel["best_tree_model"] if ecfg["model"] == "auto_tree" else ecfg["model"]
    train, test = get_split(cfg)
    X_te, y_te = split_xy(test)
    pipe = load_model(cfg, name)
    groups = feature_groups(pipe.named_steps["prep"])
    values, base, Xt, _ = tree_shap(pipe, X_te)
    agg = aggregate_by_group(values, groups)
    agg.index = X_te.index
    proba = pipe.predict_proba(X_te)[:, 1]

    # additivity check: base + sum(SHAP) reproduces the model output
    margin = base + values.sum(axis=1)
    out_space = "probability" if name in ("random_forest", "rf_original") else "log-odds"
    model_out = proba if out_space == "probability" else np.log(proba / (1 - proba))
    max_err = float(np.max(np.abs(margin - model_out)))

    importance = agg.abs().mean().sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(
        results_path(cfg, "shap_importance.csv"), index_label="feature"
    )

    # interactions (fall back to XGBoost if the chosen model does not support them)
    inter_model = name
    n_int = min(ecfg["interaction_sample"], len(X_te))
    X_int = X_te.sample(n_int, random_state=cfg["seed"])
    try:
        _, _, _, inter = tree_shap(pipe, X_int, interactions=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("interaction values not available for %s (%s); using xgboost", name, exc)
        inter_model = "xgboost"
        _, _, _, inter = tree_shap(load_model(cfg, "xgboost"), X_int, interactions=True)
    inter_agg = aggregate_interactions(inter, groups)
    pairs = [
        {
            "feature_a": a,
            "feature_b": b,
            "mean_abs_interaction": float(inter_agg.loc[a, b] + inter_agg.loc[b, a]),
        }
        for i, a in enumerate(inter_agg.index)
        for b in inter_agg.index[i + 1 :]
    ]
    pairs = pd.DataFrame(pairs).sort_values("mean_abs_interaction", ascending=False)
    pairs.head(15).to_csv(results_path(cfg, "shap_top_interactions.csv"), index=False)

    comparison = importance_comparison(cfg, test)
    insights = insight_evidence(pd.concat([train, test]), test, agg)
    insights.to_csv(results_path(cfg, "insight_validation.csv"), index=False)
    results_path(cfg, "insight_validation.md").write_text(to_markdown(insights))

    reports = customer_reports(
        cfg, test, proba, agg, base, out_space, ecfg["n_customer_reports"], name
    )

    summary = {
        "explained_model": name,
        "output_space": out_space,
        "additivity_max_abs_error": max_err,
        "top_features": importance.head(10).round(4).to_dict(),
        "interaction_model": inter_model,
        "top_interactions": pairs.head(5).round(4).to_dict(orient="records"),
        "importance_rank_correlation": comparison,
        "customer_reports": reports,
    }
    save_json(summary, results_path(cfg, "explain_summary.json"))

    _plot_beeswarm(values, base, Xt, name, out_space, cfg)
    _plot_grouped_importance(importance, name, out_space, cfg)
    _plot_dependence(X_te, agg, cfg, out_space)
    _plot_interactions(inter_agg, importance, inter_model, out_space, cfg)
    return summary


def importance_comparison(cfg: dict, test: pd.DataFrame) -> dict:
    """Impurity vs permutation vs SHAP importance for the original RF configuration."""
    X_te, y_te = split_xy(test)
    rf = load_model(cfg, "rf_original")
    groups = feature_groups(rf.named_steps["prep"])
    imp = rf.named_steps["model"].feature_importances_
    impurity = pd.Series({g: imp[idx].sum() for g, idx in groups.items()})
    perm = permutation_importance(
        rf,
        X_te,
        y_te,
        scoring="roc_auc",
        n_repeats=cfg["explain"]["permutation_repeats"],
        random_state=cfg["seed"],
        n_jobs=1,
    )
    permutation = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS)
    values, _, _, _ = tree_shap(rf, X_te)
    shap_imp = aggregate_by_group(values, groups).abs().mean()
    table = pd.DataFrame(
        {
            "impurity": impurity,
            "permutation_auc_drop": permutation,
            "permutation_auc_drop_std": pd.Series(perm.importances_std, index=FEATURE_COLUMNS),
            "mean_abs_shap": shap_imp,
        }
    )
    table["impurity_rank"] = table.impurity.rank(ascending=False).astype(int)
    table["permutation_rank"] = table.permutation_auc_drop.rank(ascending=False).astype(int)
    table["shap_rank"] = table.mean_abs_shap.rank(ascending=False).astype(int)
    table.sort_values("mean_abs_shap", ascending=False).to_csv(
        results_path(cfg, "importance_comparison.csv"), index_label="feature"
    )
    _plot_importance_comparison(table, cfg)
    return {
        "spearman_impurity_vs_shap": float(
            spearmanr(table.impurity, table.mean_abs_shap).statistic
        ),
        "spearman_impurity_vs_permutation": float(
            spearmanr(table.impurity, table.permutation_auc_drop).statistic
        ),
        "spearman_permutation_vs_shap": float(
            spearmanr(table.permutation_auc_drop, table.mean_abs_shap).statistic
        ),
        "gender_ranks": table.loc["gender", ["impurity_rank", "permutation_rank", "shap_rank"]]
        .astype(int)
        .to_dict(),
    }


def insight_evidence(full: pd.DataFrame, test: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Raw churn-rate contrasts and SHAP direction for each original business insight."""
    rate = lambda m: float(full.loc[m, "Churn"].mean())  # noqa: E731
    internet = full.InternetService != "No"
    t_int = test.InternetService != "No"
    q1, q3 = full.MonthlyCharges.quantile([0.25, 0.75])

    def shap_mean(col, level):
        return float(agg.loc[(test[col] == level).to_numpy(), col].mean())

    rho = lambda col: float(spearmanr(test[col], agg[col]).statistic)  # noqa: E731
    rows = [
        (
            "1. Short tenure -> more churn",
            f"tenure <= 12 m: {rate(full.tenure <= 12):.1%} vs > 12 m: {rate(full.tenure > 12):.1%}",
            f"Spearman(tenure, SHAP) = {rho('tenure'):+.2f}",
            rho("tenure") < 0 and rate(full.tenure <= 12) > rate(full.tenure > 12),
        ),
        (
            "2. Higher monthly charges -> more churn",
            f"top quartile (> {q3:.0f}): {rate(full.MonthlyCharges > q3):.1%} vs bottom (< {q1:.0f}): {rate(full.MonthlyCharges < q1):.1%}",
            f"Spearman(MonthlyCharges, SHAP) = {rho('MonthlyCharges'):+.2f}",
            rho("MonthlyCharges") > 0
            and rate(full.MonthlyCharges > q3) > rate(full.MonthlyCharges < q1),
        ),
        (
            "3. Long-term contracts -> less churn",
            f"two-year: {rate(full.Contract == 'Two year'):.1%}, one-year: {rate(full.Contract == 'One year'):.1%}, month-to-month: {rate(full.Contract == 'Month-to-month'):.1%}",
            f"mean SHAP two-year {shap_mean('Contract', 'Two year'):+.2f} vs month-to-month {shap_mean('Contract', 'Month-to-month'):+.2f}",
            shap_mean("Contract", "Two year") < shap_mean("Contract", "Month-to-month"),
        ),
        (
            "4. Fiber optic -> more churn",
            f"fiber: {rate(full.InternetService == 'Fiber optic'):.1%} vs DSL: {rate(full.InternetService == 'DSL'):.1%}",
            f"mean SHAP fiber {shap_mean('InternetService', 'Fiber optic'):+.2f} vs DSL {shap_mean('InternetService', 'DSL'):+.2f}",
            shap_mean("InternetService", "Fiber optic") > shap_mean("InternetService", "DSL"),
        ),
    ]
    for col in ("TechSupport", "OnlineSecurity"):
        yes = float(agg.loc[(t_int & (test[col] == "Yes")).to_numpy(), col].mean())
        no = float(agg.loc[(t_int & (test[col] == "No")).to_numpy(), col].mean())
        rows.append(
            (
                f"5. {col} -> less churn (internet customers)",
                f"with: {rate(internet & (full[col] == 'Yes')):.1%} vs without: {rate(internet & (full[col] == 'No')):.1%}",
                f"mean SHAP with {yes:+.2f} vs without {no:+.2f}",
                yes < no,
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "Claim",
            "Observed churn rates (all customers)",
            "SHAP evidence (test set)",
            "Model agrees",
        ],
    )
    df["Model agrees"] = df["Model agrees"].map({True: "yes", False: "no"})
    return df


def customer_reports(cfg, test, proba, agg, base, out_space, k, model_name) -> list[dict]:
    """Reason-code reports for the riskiest customers; chart of a high/medium/low-risk trio."""
    folder = results_path(cfg, "customer_reports/.keep").parent
    order = np.argsort(-proba)
    raw = test[FEATURE_COLUMNS]
    reports = []
    unit = "probability points" if out_space == "probability" else "log-odds"
    for rank, i in enumerate(order[:k], start=1):
        cid = test.customerID.iloc[i]
        contrib = agg.iloc[i]
        reasons = reason_codes(contrib, raw.iloc[i], k=3)
        protective = contrib.sort_values().head(2)
        lines = [
            f"# Customer {cid}",
            "",
            f"* Predicted churn probability ({DISPLAY_NAMES[model_name]}, uncalibrated): **{proba[i]:.1%}**",
            f"* Risk rank in the test set: {rank} of {len(test)}",
            f"* Actual outcome in the data: {'churned' if test.Churn.iloc[i] == 1 else 'stayed'}",
            "",
            f"## Main reasons for the risk (SHAP, {unit})",
            "",
            "| Feature | Value | Contribution |",
            "|---|---|---|",
            *[f"| {r['feature']} | {r['value']} | {r['contribution']:+.3f} |" for r in reasons],
            "",
            "## Factors lowering the risk",
            "",
            "| Feature | Value | Contribution |",
            "|---|---|---|",
            *[f"| {f} | {_fmt_value(raw.iloc[i][f])} | {c:+.3f} |" for f, c in protective.items()],
            "",
            f"Baseline (average model output): {base:+.3f} {unit}. Contributions are model associations,"
            " not causal effects.",
            "",
        ]
        (folder / f"{cid}.md").write_text("\n".join(lines))
        reports.append({"customerID": cid, "probability": float(proba[i]), "reasons": reasons})
    # chart: the riskiest customer, the one closest to p = 0.5, and the least risky one
    picks = [order[0], int(np.argmin(np.abs(proba - 0.5))), order[-1]]
    chart = [(test.customerID.iloc[i], proba[i], agg.iloc[i], raw.iloc[i]) for i in picks]
    _plot_customers(chart, base, unit, cfg)
    return reports


# --------------------------------------------------------------------------- figures
def _plot_beeswarm(values, base, Xt, name, out_space, cfg) -> None:
    P.apply_style()
    exp = shap.Explanation(
        values=values,
        base_values=np.full(len(values), base),
        data=Xt.to_numpy(),
        feature_names=list(Xt.columns),
    )
    P.plt.figure(figsize=(8.5, 6.4))
    shap.plots.beeswarm(exp, max_display=15, color=P.FEATURE_VALUE, show=False, plot_size=None)
    fig = P.plt.gcf()
    ax = fig.axes[0]
    ax.set_title(f"SHAP values: {DISPLAY_NAMES[name]}, test set", loc="left")
    ax.set_xlabel(f"SHAP value ({out_space} of churn); right = higher risk")
    ax.grid(axis="x", color=P.GRID)
    P.savefig(fig, figure_path(cfg, "shap_beeswarm.png"))


def _plot_grouped_importance(importance: pd.Series, name, out_space, cfg) -> None:
    P.apply_style()
    s = importance.head(15)[::-1]
    fig, ax = P.plt.subplots(figsize=(7.5, 5.2))
    ax.barh(range(len(s)), s.values, height=0.6, color=P.BLUE)
    for i, v in enumerate(s.values):
        ax.text(v + s.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=P.INK_2)
    ax.set_yticks(range(len(s)), s.index)
    ax.grid(axis="y", visible=False)
    ax.set(
        xlabel=f"Mean |SHAP| ({out_space}), one-hot levels summed per feature",
        title=f"Global feature importance, {DISPLAY_NAMES[name]}",
    )
    P.savefig(fig, figure_path(cfg, "shap_importance_grouped.png"))


def _plot_importance_comparison(table: pd.DataFrame, cfg) -> None:
    P.apply_style()
    t = table.sort_values("mean_abs_shap")
    norm = t[["impurity", "permutation_auc_drop", "mean_abs_shap"]].clip(lower=0)
    norm = norm / norm.sum()
    fig, ax = P.plt.subplots(figsize=(8, 6.2))
    y = np.arange(len(t))
    specs = [
        ("impurity", "Impurity (Gini) importance", P.BLUE, "o", -0.18),
        ("permutation_auc_drop", "Permutation importance (ROC-AUC drop)", P.ORANGE, "s", 0.0),
        ("mean_abs_shap", "Mean |SHAP|", P.AQUA, "D", 0.18),
    ]
    for col, label, color, marker, dy in specs:
        ax.scatter(
            norm[col],
            y + dy,
            color=color,
            marker=marker,
            s=34,
            label=label,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.set_yticks(y, t.index)
    gi = list(t.index).index("gender")
    ax.axhspan(gi - 0.45, gi + 0.45, color=P.NEUTRAL, zorder=0)
    ax.set(
        xlabel="Share of total importance (negative permutation values shown as 0)",
        title="Random forest (original configuration): three importance measures",
    )
    ax.xaxis.set_major_formatter(P.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right")
    P.savefig(fig, figure_path(cfg, "importance_method_comparison.png"))


def _plot_dependence(X: pd.DataFrame, agg: pd.DataFrame, cfg, out_space: str) -> None:
    P.apply_style()
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    specs = [
        ("tenure", "Contract", "Tenure (months)"),
        ("MonthlyCharges", "InternetService", "Monthly charges"),
    ]
    for ax, (feat, color_by, xlabel) in zip(axes, specs):
        levels = X[color_by].value_counts().index[:3]
        for color, level in zip(P.PALETTE, levels):
            m = (X[color_by] == level).to_numpy()
            ax.scatter(
                X.loc[m, feat],
                agg.loc[m, feat],
                s=10,
                alpha=0.55,
                color=color,
                label=f"{color_by}: {level}",
                linewidth=0,
            )
        ax.axhline(0, color=P.AXIS, lw=1)
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper right", markerscale=2, fontsize=9)
    axes[0].set_ylabel(f"SHAP value of the feature ({out_space})")
    axes[0].set_title("Dependence: tenure")
    axes[1].set_title("Dependence: monthly charges")
    P.savefig(fig, figure_path(cfg, "shap_dependence.png"))


def _plot_interactions(
    inter_agg: pd.DataFrame, importance: pd.Series, model, out_space, cfg
) -> None:
    P.apply_style()
    top = list(importance.head(10).index)
    m = inter_agg.loc[top, top].to_numpy().copy()
    m = m + m.T  # symmetric total interaction strength between the two features
    np.fill_diagonal(m, np.nan)
    fig, ax = P.plt.subplots(figsize=(7.8, 6.4))
    im = ax.imshow(m, cmap=P.SEQUENTIAL, aspect="auto")
    vmax = np.nanmax(m)
    for i in range(len(top)):
        for j in range(len(top)):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if m[i, j] > 0.55 * vmax else P.INK,
                )
    ax.set_xticks(range(len(top)), top, rotation=40, ha="right")
    ax.set_yticks(range(len(top)), top)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.set_label(f"Mean |SHAP interaction| ({out_space})")
    ax.set_title(f"Pairwise SHAP interaction strength, {DISPLAY_NAMES[model]}")
    P.savefig(fig, figure_path(cfg, "shap_interactions.png"))


def _plot_customers(chart, base, unit, cfg) -> None:
    if not chart:
        return
    P.apply_style()
    fig, axes = P.plt.subplots(1, len(chart), figsize=(5.4 * len(chart), 4.6), sharex=True)
    for ax, (cid, prob, contrib, raw) in zip(np.atleast_1d(axes), chart):
        order = contrib.abs().sort_values(ascending=False)
        top = contrib[order.index[:7]]
        rest = contrib[order.index[7:]].sum()
        vals = list(top.values) + [rest]
        labels = [f"{f} = {_fmt_value(raw[f])}" for f in top.index] + [
            f"{len(order) - 7} other features"
        ]
        vals, labels = vals[::-1], labels[::-1]
        colors = [P.RED if v > 0 else P.BLUE for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, height=0.6)
        ax.axvline(0, color=P.INK_2, lw=1)
        ax.set_yticks(range(len(vals)), labels, fontsize=8.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{cid}: p(churn) = {prob:.0%}", fontsize=11)
        ax.set_xlabel(f"Contribution ({unit})")
    fig.suptitle(
        "Per-customer explanations: high, medium and low predicted risk (red raises risk, blue lowers it)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    P.savefig(fig, figure_path(cfg, "customer_explanations.png"))
