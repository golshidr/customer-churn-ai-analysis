"""Synthetic data with the Telco schema, for tests, CI and smoke runs.

The generator is *not* a model of the real data. It only produces rows with valid
categories and a churn label that depends on contract, tenure, internet service and
charges, so that models have a signal to learn in fast tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORY_LEVELS, ID_COLUMN, TARGET

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_synthetic_telco(n: int = 600, seed: int = 0, raw: bool = True) -> pd.DataFrame:
    """Return ``n`` synthetic customers.

    With ``raw=True`` the frame mimics the downloaded CSV (``TotalCharges`` as text,
    ``SeniorCitizen`` as 0/1, ``Churn`` as "Yes"/"No"), so it can go through
    :func:`churn_analysis.data.clean` exactly like the real file.
    """
    rng = np.random.default_rng(seed)

    def pick(col: str, p=None, size=n):
        return rng.choice(CATEGORY_LEVELS[col], size=size, p=p)

    df = pd.DataFrame({ID_COLUMN: [f"{i:04d}-SYNTH" for i in range(n)]})
    df["gender"] = pick("gender")
    df["SeniorCitizen"] = rng.binomial(1, 0.16, n)
    df["Partner"] = pick("Partner")
    df["Dependents"] = pick("Dependents", p=[0.7, 0.3])
    df["tenure"] = rng.integers(0, 73, n)
    df["PhoneService"] = pick("PhoneService", p=[0.1, 0.9])
    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service", rng.choice(["No", "Yes"], n)
    )
    df["InternetService"] = pick("InternetService", p=[0.35, 0.44, 0.21])
    for col in ADDON_COLUMNS:
        df[col] = np.where(
            df["InternetService"] == "No", "No internet service", rng.choice(["No", "Yes"], n)
        )
    df["Contract"] = pick("Contract", p=[0.55, 0.21, 0.24])
    df["PaperlessBilling"] = pick("PaperlessBilling", p=[0.4, 0.6])
    df["PaymentMethod"] = pick("PaymentMethod", p=[0.34, 0.23, 0.22, 0.21])
    base = np.select(
        [df["InternetService"] == "Fiber optic", df["InternetService"] == "DSL"],
        [80.0, 55.0],
        default=22.0,
    )
    df["MonthlyCharges"] = np.round(base + rng.normal(0, 8, n).clip(-15, 30), 2)
    total = np.round(df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.95, 1.05, n), 2)

    logit = (
        -1.0
        + 1.4 * (df["Contract"] == "Month-to-month")
        - 1.2 * (df["Contract"] == "Two year")
        + 0.9 * (df["InternetService"] == "Fiber optic")
        + 0.5 * (df["PaymentMethod"] == "Electronic check")
        - 0.04 * df["tenure"]
        + 0.01 * (df["MonthlyCharges"] - 60)
        - 0.4 * (df["TechSupport"] == "Yes")
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    if raw:
        df["TotalCharges"] = [" " if t == 0 else f"{v:.2f}" for t, v in zip(df["tenure"], total)]
        df[TARGET] = np.where(churn == 1, "Yes", "No")
    else:
        df["SeniorCitizen"] = np.where(df["SeniorCitizen"] == 1, "Yes", "No")
        df["TotalCharges"] = total.astype(float)
        df[TARGET] = churn
    return df
