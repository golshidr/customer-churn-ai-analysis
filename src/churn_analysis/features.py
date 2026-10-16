"""Feature preprocessing shared by every model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, CATEGORY_LEVELS, NUMERIC_COLUMNS


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """One-hot encode categoricals (binary columns get a single indicator) and optionally
    standardise the numeric columns. Unknown categories at prediction time map to all-zeros.

    Category levels are fixed from :data:`CATEGORY_LEVELS`, so the encoded columns are the
    same whatever subset of the data the transformer is fitted on (important inside CV).
    """
    encoder = OneHotEncoder(
        categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_COLUMNS],
        drop="if_binary",
        handle_unknown="ignore",
        sparse_output=False,
    )
    numeric = (
        StandardScaler() if scale_numeric else FunctionTransformer(feature_names_out="one-to-one")
    )
    return ColumnTransformer(
        [("cat", encoder, CATEGORICAL_COLUMNS), ("num", numeric, NUMERIC_COLUMNS)],
        verbose_feature_names_out=False,
    )


def feature_groups(preprocessor: ColumnTransformer) -> dict[str, list[int]]:
    """Map each original column to the indices of its encoded columns.

    Used to add up SHAP values (or impurity importances) of the one-hot columns that
    belong to the same original feature, e.g. ``Contract`` -> 3 indicator columns.
    """
    groups: dict[str, list[int]] = {}
    idx = 0
    encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    for col, cats, drop in zip(CATEGORICAL_COLUMNS, encoder.categories_, encoder.drop_idx_):
        width = len(cats) - (0 if drop is None else 1)
        groups[col] = list(range(idx, idx + width))
        idx += width
    for col in NUMERIC_COLUMNS:
        groups[col] = [idx]
        idx += 1
    n_out = len(preprocessor.get_feature_names_out())
    if idx != n_out:  # defensive: layout must match the transformer output
        raise RuntimeError(f"feature group layout mismatch: {idx} != {n_out}")
    return groups


def aggregate_by_group(values: np.ndarray, groups: dict[str, list[int]]) -> pd.DataFrame:
    """Sum columns of ``values`` (n_samples x n_encoded) within each feature group."""
    values = np.asarray(values)
    return pd.DataFrame({g: values[:, idx].sum(axis=1) for g, idx in groups.items()})


def legacy_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encoding used by the original notebook: ``pd.get_dummies(drop_first=True)``.

    Expects the raw CSV with ``TotalCharges`` already numeric and missing rows dropped.
    """
    out = df.drop(columns=["customerID"]).copy()
    out["Churn"] = (
        out["Churn"].map({"Yes": 1, "No": 0}) if out["Churn"].dtype != int else out["Churn"]
    )
    out = pd.get_dummies(out, drop_first=True)
    return out.drop(columns="Churn"), out["Churn"].astype(int)
