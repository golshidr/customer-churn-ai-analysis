import pandas as pd
import pytest

from churn_analysis.data import (
    CATEGORICAL_COLUMNS,
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    clean,
    normalized_sha256,
    train_test,
    validate_schema,
)


def test_clean_coerces_and_drops_blank_total_charges(raw_df):
    n_blank = (raw_df.TotalCharges == " ").sum()
    assert n_blank > 0  # the generator reproduces the blank strings of tenure-0 rows
    out = clean(raw_df)
    assert len(out) == len(raw_df) - n_blank
    assert out.TotalCharges.dtype.kind == "f"
    assert set(out.Churn.unique()) <= {0, 1}
    assert set(out.SeniorCitizen.unique()) <= {"No", "Yes"}


def test_clean_can_impute_instead_of_drop(raw_df):
    out = clean(raw_df, drop_missing_total_charges=False)
    assert len(out) == len(raw_df)
    assert (out.loc[out.tenure == 0, "TotalCharges"] == 0).all()


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_validate_schema_rejects_missing_and_unknown(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(raw_df.drop(columns=["Contract"]))
    bad = raw_df.copy()
    bad.loc[0, "Contract"] = "Three year"
    with pytest.raises(ValueError, match="unexpected values in Contract"):
        validate_schema(bad)


def test_clean_rejects_bad_labels(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        clean(bad)


def test_stratified_split_preserves_churn_rate(clean_df):
    train, test = train_test(clean_df, test_size=0.25, seed=0)
    assert len(train) + len(test) == len(clean_df)
    assert abs(train.Churn.mean() - test.Churn.mean()) < 0.02
    assert set(train.customerID).isdisjoint(test.customerID)


def test_normalized_checksum_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\r\n1,2\r\n")
    b.write_bytes(b"x,y\n1,2\n")
    assert normalized_sha256(a) == normalized_sha256(b)
    assert len(EXPECTED_SHA256) == 64


def test_feature_lists_are_consistent():
    assert set(CATEGORICAL_COLUMNS) < set(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 19
