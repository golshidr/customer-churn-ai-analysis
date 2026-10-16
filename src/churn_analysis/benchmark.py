"""Stage 1: reproduce the original result, then benchmark tuned models with proper CV.

Protocol
--------
* One stratified 80/20 train/test split (seed 42). The test set is only used for the
  final evaluation.
* Each tunable model gets ``tuning.n_trials`` Optuna trials, scored by mean 5-fold
  stratified CV ROC-AUC on the training set.
* The model used downstream ("selected model") is the one with the best CV ROC-AUC, so
  the choice never looks at the test set.
* Test metrics come with 95% percentile-bootstrap intervals, and differences against the
  reference model with a paired bootstrap.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from . import plotting as P
from .data import split_xy
from .evaluate import PROBA_METRICS, metrics_with_ci, paired_bootstrap_diff
from .features import legacy_encode
from .models import DISPLAY_NAMES, FIXED_PARAMS, make_pipeline, model_device, resolve_device
from .runtime import (
    artifact_path,
    figure_path,
    get_split,
    load_raw_frame,
    results_path,
    save_json,
    save_model,
    to_markdown,
)
from .tuning import cross_val_oof, make_folds, tune_model

log = logging.getLogger("churn_analysis")


# --------------------------------------------------------------------------- original
def reproduce_original(cfg: dict) -> dict:
    """Re-run the notebook's exact pipeline (get_dummies, unstratified split, RF-200)."""
    raw = load_raw_frame(cfg).copy()
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    raw = raw.dropna()
    X, y = legacy_encode(raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    report = classification_report(y_te, pred, output_dict=True, zero_division=0)
    out = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "test_size": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "majority_class_accuracy": float(1 - y_te.mean()),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "churn_precision": float(report["1"]["precision"]),
        "churn_recall": float(report["1"]["recall"]),
        "churn_f1": float(report["1"]["f1-score"]),
        "top10_impurity_importance": pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
        .round(4)
        .to_dict(),
    }
    save_json(out, results_path(cfg, "original_baseline.json"))
    log.info("original pipeline: accuracy %.4f, ROC-AUC %.4f", out["accuracy"], out["roc_auc"])
    return out


# --------------------------------------------------------------------------- benchmark
def run_benchmark(cfg: dict) -> pd.DataFrame:
    seed, n_jobs = cfg["seed"], cfg["compute"]["n_jobs"]
    gpu = resolve_device(cfg["compute"]["device"])
    log.info(
        "devices: xgboost=%s catboost=%s",
        model_device("xgboost", cfg),
        model_device("catboost", cfg),
    )
    train, test = get_split(cfg)
    X_tr, y_tr = split_xy(train)
    X_te, y_te = split_xy(test)
    folds = make_folds(y_tr, cfg["cv"]["n_splits"], seed)

    names = ["dummy", "rf_original", *cfg["tuning"]["models"]]
    oof = pd.DataFrame({"customerID": train["customerID"], "y": y_tr})
    preds = pd.DataFrame({"customerID": test["customerID"], "y": y_te})
    rows, best, histories = [], {}, {}
    for name in names:
        device = model_device(name, cfg) if name in ("xgboost", "catboost") else "cpu"
        t0 = time.perf_counter()
        n_trials = 0
        if name in FIXED_PARAMS:
            params = FIXED_PARAMS[name]
        else:
            params, study = tune_model(
                name,
                X_tr,
                y_tr,
                folds,
                n_trials=cfg["tuning"]["n_trials"],
                timeout=cfg["tuning"]["timeout_per_model"],
                device=device,
                n_jobs=n_jobs,
                seed=seed,
            )
            n_trials = len(study.trials)
            trials = study.trials_dataframe()
            trials.to_csv(results_path(cfg, f"tuning/{name}_trials.csv"), index=False)
            histories[name] = trials
        tune_s = time.perf_counter() - t0
        best[name] = params

        oof[name], fold_scores = cross_val_oof(
            name, params, X_tr, y_tr, folds, device=device, n_jobs=n_jobs, seed=seed
        )
        t1 = time.perf_counter()
        pipe = make_pipeline(name, params, device=device, n_jobs=n_jobs, seed=seed).fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t1
        preds[name] = pipe.predict_proba(X_te)[:, 1]
        save_model(pipe, cfg, name)

        m = metrics_with_ci(y_te, preds[name], n_boot=cfg["evaluation"]["n_bootstrap"], seed=seed)
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(np.mean(fold_scores)),
                "cv_roc_auc_std": float(np.std(fold_scores, ddof=1)),
                **{f"test_{k}": v for k, v in m.items()},
                "n_trials": n_trials,
                "tuning_seconds": tune_s,
                "final_fit_seconds": fit_s,
                "device": device,
            }
        )
        log.info("%s: CV AUC %.4f | test AUC %.4f", name, np.mean(fold_scores), m["roc_auc"])

    table = pd.DataFrame(rows)
    candidates = table[table.model != "dummy"]
    selected = candidates.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"]
    tree = candidates[candidates.model.isin(["random_forest", "xgboost", "lightgbm", "catboost"])]
    best_tree = (
        tree.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"] if len(tree) else None
    )

    ref = cfg["evaluation"]["reference_model"]
    diffs = []
    if ref in preds:
        for name in names:
            if name == ref:
                continue
            for metric in ("roc_auc", "pr_auc"):
                d = paired_bootstrap_diff(
                    y_te,
                    preds[name],
                    preds[ref],
                    PROBA_METRICS[metric],
                    n_boot=cfg["evaluation"]["n_bootstrap"],
                    seed=seed,
                )
                diffs.append({"model": name, "reference": ref, "metric": metric, **d})
    diffs = pd.DataFrame(diffs)

    table.to_csv(results_path(cfg, "benchmark.csv"), index=False)
    diffs.to_csv(results_path(cfg, "benchmark_paired_diffs.csv"), index=False)
    oof.to_csv(artifact_path(cfg, "oof_train.csv"), index=False)
    preds.to_csv(artifact_path(cfg, "test_predictions.csv"), index=False)
    save_json(best, results_path(cfg, "best_params.json"))
    save_json(
        {
            "selected_model": selected,
            "best_tree_model": best_tree,
            "rule": "highest mean CV ROC-AUC",
            "devices": {
                "xgboost": model_device("xgboost", cfg),
                "catboost": model_device("catboost", cfg),
            },
        },
        results_path(cfg, "selected_model.json"),
    )
    results_path(cfg, "benchmark.md").write_text(benchmark_markdown(table, diffs))

    _plot_curves(y_te, preds, names, figure_path(cfg, "benchmark_roc_pr.png"))
    _plot_auc_forest(table, figure_path(cfg, "benchmark_test_auc.png"))
    if histories:
        _plot_histories(histories, figure_path(cfg, "tuning_history.png"))
    if gpu == "cuda":
        gpu_cpu_timing(cfg, best, X_tr, y_tr)
    return table


def benchmark_markdown(table: pd.DataFrame, diffs: pd.DataFrame) -> str:
    t = table.copy()
    t["Model"] = t.model.map(DISPLAY_NAMES)
    t["CV ROC-AUC"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t.cv_roc_auc_mean, t.cv_roc_auc_std)]
    t["Test ROC-AUC [95% CI]"] = [
        f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
        for a, lo, hi in zip(t.test_roc_auc, t.test_roc_auc_ci_low, t.test_roc_auc_ci_high)
    ]
    t["Test PR-AUC [95% CI]"] = [
        f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
        for a, lo, hi in zip(t.test_pr_auc, t.test_pr_auc_ci_low, t.test_pr_auc_ci_high)
    ]
    t["Brier"] = t.test_brier.map("{:.3f}".format)
    t["Acc@0.5"] = t.test_accuracy.map("{:.3f}".format)
    t["Recall@0.5"] = t.test_recall.map("{:.3f}".format)
    cols = [
        "Model",
        "CV ROC-AUC",
        "Test ROC-AUC [95% CI]",
        "Test PR-AUC [95% CI]",
        "Brier",
        "Acc@0.5",
        "Recall@0.5",
    ]
    out = to_markdown(t[cols])
    if len(diffs):
        d = diffs[diffs.metric == "roc_auc"].copy()
        d["Model"] = d.model.map(DISPLAY_NAMES)
        d["ΔROC-AUC vs " + DISPLAY_NAMES[d.reference.iloc[0]]] = [
            f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]" for x, lo, hi in zip(d["diff"], d.ci_low, d.ci_high)
        ]
        d["p (bootstrap)"] = d.p_value.map("{:.3f}".format)
        out += "\n" + to_markdown(d[["Model", d.columns[-2], "p (bootstrap)"]])
    return out


def gpu_cpu_timing(
    cfg: dict, best: dict, X: pd.DataFrame, y: pd.Series, repeats: int = 3
) -> pd.DataFrame:
    """Wall-clock fit time of the tuned XGBoost / CatBoost on CPU vs GPU (median of runs)."""
    rows = []
    for name in ("xgboost", "catboost"):
        if name not in best:
            continue
        for device in ("cpu", "cuda"):
            times = []
            for _ in range(repeats):
                pipe = make_pipeline(
                    name,
                    best[name],
                    device=device,
                    n_jobs=cfg["compute"]["n_jobs"],
                    seed=cfg["seed"],
                )
                t0 = time.perf_counter()
                pipe.fit(X, y)
                times.append(time.perf_counter() - t0)
            rows.append(
                {
                    "model": name,
                    "device": device,
                    "median_fit_seconds": float(np.median(times)),
                    "n_jobs_cpu": cfg["compute"]["n_jobs"],
                    "n_rows": len(X),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(results_path(cfg, "gpu_cpu_timing.csv"), index=False)
    return df


# --------------------------------------------------------------------------- figures
def _plot_curves(y, preds: pd.DataFrame, names: list[str], path) -> None:
    P.apply_style()
    fig, (ax1, ax2) = P.plt.subplots(1, 2, figsize=(11, 4.6))
    for name in names:
        if name == "dummy":
            continue
        fpr, tpr, _ = roc_curve(y, preds[name])
        prec, rec, _ = precision_recall_curve(y, preds[name])
        auc = roc_auc_score(y, preds[name])
        ls = "--" if name == "rf_original" else "-"
        lw = 1.6 if name == "rf_original" else 2.0
        ax1.plot(
            fpr,
            tpr,
            ls,
            lw=lw,
            color=P.model_color(name),
            label=f"{DISPLAY_NAMES[name]} ({auc:.3f})",
        )
        ax2.plot(rec, prec, ls, lw=lw, color=P.model_color(name), label=DISPLAY_NAMES[name])
    ax1.plot([0, 1], [0, 1], color=P.AXIS, lw=1)
    ax2.axhline(np.mean(y), color=P.AXIS, lw=1)
    ax2.text(
        0.99,
        np.mean(y) + 0.015,
        f"base rate {np.mean(y):.2f}",
        ha="right",
        color=P.INK_2,
        fontsize=9,
    )
    ax1.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curves (test set)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax2.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curves (test set)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax1.legend(loc="lower right", title="ROC-AUC", title_fontsize=9.5)
    P.savefig(fig, path)


def _plot_auc_forest(table: pd.DataFrame, path) -> None:
    P.apply_style()
    t = table[table.model != "dummy"].sort_values("test_roc_auc")
    fig, axes = P.plt.subplots(1, 2, figsize=(11, 0.55 * len(t) + 1.4), sharey=True)
    for ax, metric, label in zip(axes, ("roc_auc", "pr_auc"), ("ROC-AUC", "PR-AUC")):
        y = np.arange(len(t))
        lo = t[f"test_{metric}"] - t[f"test_{metric}_ci_low"]
        hi = t[f"test_{metric}_ci_high"] - t[f"test_{metric}"]
        for i, (_, r) in enumerate(t.iterrows()):
            ax.errorbar(
                r[f"test_{metric}"],
                i,
                xerr=[[lo.iloc[i]], [hi.iloc[i]]],
                fmt="o",
                ms=7,
                color=P.model_color(r.model),
                ecolor=P.model_color(r.model),
                elinewidth=2,
                capsize=0,
                mec="white",
                mew=1.5,
            )
            ax.text(
                r[f"test_{metric}_ci_high"] + 0.003,
                i,
                f"{r[f'test_{metric}']:.3f}",
                va="center",
                fontsize=9,
                color=P.INK_2,
            )
        ax.set_yticks(y, [DISPLAY_NAMES[m] for m in t.model])
        ax.set_xlabel(f"Test {label} (95% bootstrap CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_title("Held-out performance with uncertainty")
    P.savefig(fig, path)


def _plot_histories(histories: dict[str, pd.DataFrame], path) -> None:
    P.apply_style()
    fig, ax = P.plt.subplots(figsize=(7.5, 4.4))
    for name, df in histories.items():
        done = df[df.state == "COMPLETE"].sort_values("number")
        if done.empty:
            continue
        best = done.value.cummax()
        ax.plot(
            done.number,
            best,
            color=P.model_color(name),
            label=DISPLAY_NAMES[name],
            drawstyle="steps-post",
        )
        ax.plot(done.number, done.value, "o", ms=3, alpha=0.35, color=P.model_color(name))
    ax.set(
        xlabel="Optuna trial",
        ylabel="Mean 5-fold CV ROC-AUC",
        title="Hyper-parameter search: best-so-far CV ROC-AUC",
    )
    lo = min(
        df[df.state == "COMPLETE"].value.quantile(0.1)
        for df in histories.values()
        if (df.state == "COMPLETE").any()
    )
    ax.set_ylim(bottom=lo)
    ax.legend(loc="lower right")
    P.savefig(fig, path)
