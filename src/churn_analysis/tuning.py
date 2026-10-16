"""Hyper-parameter tuning with Optuna and stratified cross-validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .models import complete_params, make_pipeline, suggest_params

log = logging.getLogger("churn_analysis")


def make_folds(y: pd.Series, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def cross_val_oof(
    name: str,
    params: dict[str, Any] | None,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
    n_jobs: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold churn probabilities and the ROC-AUC of each fold."""
    oof = np.zeros(len(y))
    scores = []
    for tr, va in folds:
        pipe = make_pipeline(name, params, device=device, n_jobs=n_jobs, seed=seed)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], oof[va]))
    return oof, scores


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    timeout: float | None = None,
    device: str = "cpu",
    n_jobs: int = 4,
    seed: int = 42,
) -> tuple[dict[str, Any], optuna.Study]:
    """Maximise mean CV ROC-AUC with TPE; unpromising trials are pruned fold by fold."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        scores: list[float] = []
        for k, (tr, va) in enumerate(folds):
            pipe = make_pipeline(name, params, device=device, n_jobs=n_jobs, seed=seed)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            scores.append(roc_auc_score(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % 10 == 0:
            done = [t for t in study.trials if t.value is not None]
            best = max((t.value for t in done), default=float("nan"))
            log.info("%s: %d trials, best CV ROC-AUC so far %.4f", name, trial.number + 1, best)

    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True, callbacks=[progress]
    )
    log.info("%s: best CV ROC-AUC %.4f after %d trials", name, study.best_value, len(study.trials))
    return complete_params(name, study.best_params), study
