import numpy as np
import pandas as pd

from churn_analysis.data import FEATURE_COLUMNS
from churn_analysis.features import (
    aggregate_by_group,
    build_preprocessor,
    feature_groups,
    legacy_encode,
)
from churn_analysis.synthetic import make_synthetic_telco


def test_preprocessor_output_is_finite_and_grouped(xy):
    X, _ = xy
    prep = build_preprocessor().fit(X)
    Xt = prep.transform(X)
    assert np.isfinite(Xt).all()
    groups = feature_groups(prep)
    assert list(groups) == FEATURE_COLUMNS
    assert sum(len(v) for v in groups.values()) == Xt.shape[1]
    assert len(groups["Contract"]) == 3  # multi-level: one column per level
    assert len(groups["gender"]) == 1  # binary: a single indicator


def test_preprocessor_layout_does_not_depend_on_fit_subset(xy):
    X, _ = xy
    small = X[X.Contract == "Month-to-month"].head(20)
    a = build_preprocessor().fit(X).get_feature_names_out()
    b = build_preprocessor().fit(small).get_feature_names_out()
    assert list(a) == list(b)


def test_unknown_category_maps_to_zeros(xy):
    X, _ = xy
    prep = build_preprocessor().fit(X)
    row = X.head(1).copy()
    row["PaymentMethod"] = "Crypto"
    out = prep.transform(row)
    idx = feature_groups(prep)["PaymentMethod"]
    assert np.allclose(out[0, idx], 0)


def test_aggregate_by_group_sums_blocks():
    values = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    out = aggregate_by_group(values, {"a": [0, 1], "b": [2]})
    assert out.a.tolist() == [3.0, 0.0]
    assert out.b.tolist() == [3.0, 1.0]


def test_legacy_encoding_matches_notebook_layout():
    raw = make_synthetic_telco(200, seed=3)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    X, y = legacy_encode(raw.dropna())
    assert X.shape[1] == 30  # the notebook's 30 dummy-encoded features
    assert "gender_Male" in X.columns and "Contract_Two year" in X.columns
    assert set(y.unique()) <= {0, 1}
