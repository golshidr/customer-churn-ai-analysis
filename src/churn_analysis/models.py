"""Model zoo: scikit-learn pipelines and Optuna search spaces."""

from __future__ import annotations

import shutil
from typing import Any

import optuna
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

TREE_MODELS = ("random_forest", "rf_original", "xgboost", "lightgbm", "catboost")
ALL_MODELS = ("dummy", "logreg", *TREE_MODELS)

#: Display names used in tables and figures.
DISPLAY_NAMES = {
    "dummy": "Majority class",
    "logreg": "Logistic regression",
    "rf_original": "Random forest (original config)",
    "random_forest": "Random forest (tuned)",
    "xgboost": "XGBoost (tuned)",
    "lightgbm": "LightGBM (tuned)",
    "catboost": "CatBoost (tuned)",
}

#: Untuned baselines. ``rf_original`` is the notebook's configuration.
FIXED_PARAMS: dict[str, dict[str, Any]] = {
    "dummy": {},
    "rf_original": {"n_estimators": 200},
}


def resolve_device(device: str) -> str:
    """``auto`` -> ``cuda`` when an NVIDIA GPU is visible, else ``cpu``."""
    if device != "auto":
        return device
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return "cuda" if out.returncode == 0 and "GPU" in out.stdout else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def model_device(name: str, cfg: dict) -> str:
    """Device for one model: ``compute.device``, with a per-model override for CatBoost."""
    compute = cfg["compute"]
    if name == "catboost" and compute.get("catboost_device"):
        return resolve_device(compute["catboost_device"])
    return resolve_device(compute["device"])


def make_estimator(
    name: str,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    n_jobs: int = 4,
    seed: int = 42,
):
    params = dict(params or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=5000, **params)
    if name in ("random_forest", "rf_original"):
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            tree_method="hist",
            device="cuda" if device == "cuda" else "cpu",
            n_jobs=n_jobs,
            random_state=seed,
            eval_metric="logloss",
            **params,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **params)
    if name == "catboost":
        from catboost import CatBoostClassifier

        # CatBoost reserves 95% of GPU memory by default; this data needs well under 1 GB,
        # so cap it to leave a shared GPU usable for other jobs.
        gpu = (
            {"task_type": "GPU", "devices": "0", "gpu_ram_part": 0.05}
            if device == "cuda"
            else {"task_type": "CPU"}
        )
        return CatBoostClassifier(
            random_seed=seed,
            thread_count=n_jobs,
            verbose=0,
            allow_writing_files=False,
            **gpu,
            **params,
        )
    raise ValueError(f"unknown model {name!r}; choose from {ALL_MODELS}")


def make_pipeline(
    name: str,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    n_jobs: int = 4,
    seed: int = 42,
) -> Pipeline:
    """Preprocessing + estimator. Only the linear model needs scaled numerics."""
    if params is None:
        params = FIXED_PARAMS.get(name, {})
    return Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=(name == "logreg"))),
            ("model", make_estimator(name, params, device=device, n_jobs=n_jobs, seed=seed)),
        ]
    )


def suggest_params(name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Optuna search space for each tunable model."""
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
    raise ValueError(f"model {name!r} has no search space")


def complete_params(name: str, best: dict[str, Any]) -> dict[str, Any]:
    """Add fixed (non-searched) parameters back to Optuna's ``best_params``."""
    params = dict(best)
    if name == "lightgbm":
        params.setdefault("subsample_freq", 1)
    return params
