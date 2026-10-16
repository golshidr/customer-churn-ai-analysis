import numpy as np
import pandas as pd
import pytest

from churn_analysis.explain import aggregate_interactions, positive_class, reason_codes, tree_shap
from churn_analysis.models import make_pipeline


def test_positive_class_handles_all_shap_layouts():
    """Regression test for the original notebook bug (3-D SHAP output)."""
    pos = np.arange(12, dtype=float).reshape(4, 3)
    three_d = np.stack([-pos, pos], axis=-1)  # (n, features, classes), SHAP >= 0.45
    assert np.array_equal(positive_class(three_d), pos)
    assert np.array_equal(positive_class([-pos, pos]), pos)  # older list layout
    assert np.array_equal(positive_class(pos), pos)  # single-output models
    with pytest.raises(ValueError):
        positive_class(np.zeros(3))


@pytest.mark.parametrize(
    "name,params",
    [("rf_original", {"n_estimators": 25}), ("xgboost", {"n_estimators": 25, "max_depth": 3})],
)
def test_tree_shap_is_additive(name, params, xy):
    X, y = xy
    pipe = make_pipeline(name, params, device="cpu", n_jobs=1).fit(X, y)
    values, base, Xt, _ = tree_shap(pipe, X.head(50))
    assert values.shape == Xt.shape
    p = pipe.predict_proba(X.head(50))[:, 1]
    target = p if name == "rf_original" else np.log(p / (1 - p))
    assert np.allclose(base + values.sum(axis=1), target, atol=1e-4)


def test_aggregate_interactions_symmetric_blocks():
    inter = np.ones((5, 3, 3))
    out = aggregate_interactions(inter, {"a": [0, 1], "b": [2]})
    assert out.loc["a", "a"] == 4 and out.loc["a", "b"] == 2 and out.loc["b", "b"] == 1


def test_reason_codes_only_positive_contributions():
    contrib = pd.Series({"tenure": 0.4, "Contract": 0.9, "TechSupport": -0.2})
    raw = pd.Series({"tenure": 2, "Contract": "Month-to-month", "TechSupport": "No"})
    codes = reason_codes(contrib, raw, k=3)
    assert [c["feature"] for c in codes] == ["Contract", "tenure"]
    assert codes[0]["value"] == "Month-to-month"
