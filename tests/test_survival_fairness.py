import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from churn_analysis.fairness import auc_gap_ci, disparity, group_metrics
from churn_analysis.survival import km_table, survival_design


def test_survival_design_has_no_duplicate_internet_indicators(clean_df):
    d = survival_design(clean_df)
    assert {"duration", "event"} <= set(d.columns)
    assert "TotalCharges" not in d.columns and "MonthlyCharges" not in d.columns
    assert not any("No internet service" in c for c in d.columns)
    assert d.drop(columns=["duration", "event"]).isin([0.0, 1.0]).all().all()


def test_cox_model_fits_and_ranks(clean_df):
    d = survival_design(clean_df)
    cph = CoxPHFitter(penalizer=0.1).fit(d, "duration", "event")
    assert 0.5 < cph.concordance_index_ <= 1
    assert cph.summary.loc["Contract=Two year", "coef"] < 0  # generator lowers churn for 2-year


def test_km_table_one_row_per_group(clean_df):
    t = km_table(clean_df, "Contract", horizon=72)
    assert len(t) == 3
    assert ((t.survival_12m >= 0) & (t.survival_12m <= 1)).all()


def test_group_metrics_and_disparity():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 600)
    p = np.clip(0.5 * y + rng.uniform(0, 0.5, 600), 0, 1)
    groups = np.where(np.arange(600) % 2 == 0, "A", "B")
    t = group_metrics(y, p, p > 0.5, groups, n_boot=50)
    assert set(t.group) == {"A", "B"} and t.n.sum() == 600
    assert (t.roc_auc_ci_low <= t.roc_auc).all()
    gaps = disparity(t)
    assert gaps["tpr_gap"] >= 0 and 0 < gaps["contacted_ratio"] <= 1
    lo, hi = auc_gap_ci(y, p, pd.Series(groups), n_boot=50)
    assert lo <= hi
