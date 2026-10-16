"""Model bundle for the scoring service, and the scoring logic used by the API.

A bundle is one joblib file holding:

* ``model``: the (calibrated) classifier that produces churn probabilities,
* ``explainer_pipeline``: the uncalibrated pipeline used for reason codes,
* ``business``: the campaign economics used for the expected-value recommendation,
* ``metadata``: model name, parameters, test metrics and library versions.
"""

from __future__ import annotations

import datetime as dt
import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from .config import resolve_path
from .data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, split_xy
from .decision import customer_economics, expected_offer_value
from .features import aggregate_by_group, feature_groups
from .models import TREE_MODELS, make_pipeline
from .runtime import best_params, get_split, load_json, results_path, save_json

log = logging.getLogger("churn_analysis")

RISK_BANDS = ((0.5, "high"), (0.2, "medium"), (0.0, "low"))


def build_bundle(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    business: dict,
    calibration: str = "uncalibrated",
    seed: int = 42,
    n_jobs: int = 2,
    metadata: dict | None = None,
) -> dict:
    """Fit the scoring and explanation models on CPU and package them."""
    pipe = make_pipeline(model_name, params, device="cpu", n_jobs=n_jobs, seed=seed).fit(X, y)
    if calibration in ("sigmoid", "isotonic"):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        model = CalibratedClassifierCV(
            make_pipeline(model_name, params, device="cpu", n_jobs=n_jobs, seed=seed),
            method=calibration,
            cv=cv,
        ).fit(X, y)
    else:
        model = pipe
    background = pipe.named_steps["prep"].transform(X).mean(axis=0)
    import sklearn

    meta = {
        "model": model_name,
        "params": params,
        "calibration": calibration,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_train": int(len(X)),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        **(metadata or {}),
    }
    return {
        "model": model,
        "explainer_pipeline": pipe,
        "explainer_kind": "tree" if model_name in TREE_MODELS else "linear",
        "background_mean": background,
        "business": dict(business),
        "metadata": meta,
    }


def prepare_features(records: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Turn API/CSV records into the cleaned feature frame the pipelines expect."""
    df = pd.DataFrame(records).copy()
    if "SeniorCitizen" in df and pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df[FEATURE_COLUMNS]


def contributions(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contributions (log-odds or probability for RF) of the explanation model."""
    pipe = bundle["explainer_pipeline"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    groups = feature_groups(prep)
    Xt = prep.transform(X)
    if bundle["explainer_kind"] == "tree":
        import shap

        from .explain import positive_class

        explainer = bundle.get("_explainer")
        if explainer is None:
            explainer = bundle["_explainer"] = shap.TreeExplainer(model)
        values = positive_class(
            explainer.shap_values(pd.DataFrame(Xt, columns=prep.get_feature_names_out()))
        )
    else:  # linear model: exact SHAP values under feature independence
        values = (Xt - bundle["background_mean"]) * model.coef_[0]
    return aggregate_by_group(values, groups)


def risk_band(p: float) -> str:
    return next(label for cut, label in RISK_BANDS if p >= cut)


def score(bundle: dict, records: list[dict] | pd.DataFrame, top_k: int = 3) -> list[dict[str, Any]]:
    X = prepare_features(records)
    p = bundle["model"].predict_proba(X)[:, 1]
    biz = bundle["business"]
    value, incentive = customer_economics(X["MonthlyCharges"], biz)
    ev = expected_offer_value(p, value, incentive, biz["contact_cost"], biz["acceptance_rate"])
    contrib = contributions(bundle, X)
    ids = pd.DataFrame(records).get("customerID")
    out = []
    for i in range(len(X)):
        row = contrib.iloc[i].sort_values(ascending=False)
        reasons = [
            {"feature": f, "value": str(X.iloc[i][f]), "contribution": round(float(c), 4)}
            for f, c in row.head(top_k).items()
            if c > 0
        ]
        out.append(
            {
                "customerID": None if ids is None or pd.isna(ids.iloc[i]) else str(ids.iloc[i]),
                "churn_probability": round(float(p[i]), 4),
                "risk_band": risk_band(float(p[i])),
                "expected_offer_value": round(float(ev[i]), 2),
                "recommend_offer": bool(ev[i] > 0),
                "top_reasons": reasons,
            }
        )
    return out


def export_bundle(cfg: dict) -> Path:
    """Fit the selected model (with its chosen calibration) on the training split and save."""
    selected = load_json(results_path(cfg, "selected_model.json"))["selected_model"]
    decision = load_json(results_path(cfg, "decision_summary.json"))
    bench = pd.read_csv(results_path(cfg, "benchmark.csv")).set_index("model")
    train, test = get_split(cfg)
    X_tr, y_tr = split_xy(train)
    bundle = build_bundle(
        selected,
        best_params(cfg)[selected],
        X_tr,
        y_tr,
        cfg["business"],
        calibration=decision["calibration_chosen"],
        seed=cfg["seed"],
        n_jobs=cfg["compute"]["n_jobs"],
        metadata={
            "test_roc_auc_uncalibrated": float(bench.loc[selected, "test_roc_auc"]),
            "profit_optimal_threshold": decision["profit_optimal_threshold"],
        },
    )
    path = resolve_path(cfg, "models") / cfg["serving"]["bundle_name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    # Record an example request/response, produced by the real service code.
    from fastapi.testclient import TestClient

    from .api import create_app

    sample = test.sort_values("customerID").head(3)
    records = [_to_api_record(r) for _, r in sample.iterrows()]
    with TestClient(create_app(bundle=bundle)) as client:
        resp = client.post("/predict", json={"customers": records})
        resp.raise_for_status()
    save_json(
        {"request": {"customers": records}, "response": resp.json()},
        results_path(cfg, "api_example.json"),
    )
    log.info("saved bundle to %s", path)
    return path


def _to_api_record(row: pd.Series) -> dict:
    rec = {c: row[c] for c in ["customerID", *FEATURE_COLUMNS]}
    rec["SeniorCitizen"] = 1 if rec["SeniorCitizen"] == "Yes" else 0
    rec["tenure"] = int(rec["tenure"])
    for c in ("MonthlyCharges", "TotalCharges"):
        rec[c] = float(rec[c])
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in rec.items()}
