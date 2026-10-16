import pytest

from churn_analysis.data import clean
from churn_analysis.synthetic import make_synthetic_telco


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic_telco(n=500, seed=1)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean(raw_df)


@pytest.fixture(scope="session")
def xy(clean_df):
    from churn_analysis.data import split_xy

    return split_xy(clean_df)
