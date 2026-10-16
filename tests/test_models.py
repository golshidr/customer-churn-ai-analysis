import optuna
import pytest
from sklearn.metrics import roc_auc_score

from churn_analysis.models import ALL_MODELS, make_pipeline, suggest_params
from churn_analysis.tuning import cross_val_oof, make_folds, tune_model

FAST = {
    "random_forest": {"n_estimators": 30, "max_depth": 4},
    "rf_original": {"n_estimators": 30},
    "xgboost": {"n_estimators": 30, "max_depth": 3},
    "lightgbm": {"n_estimators": 30, "num_leaves": 8},
    "catboost": {"iterations": 30, "depth": 3},
    "logreg": {"C": 1.0},
    "dummy": {},
}


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_model_fits_and_predicts_probabilities(name, xy):
    X, y = xy
    pipe = make_pipeline(name, FAST[name], device="cpu", n_jobs=1).fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()
    if name != "dummy":
        assert roc_auc_score(y, p) > 0.6  # synthetic data carries a real signal


@pytest.mark.parametrize("name", ["logreg", "random_forest", "xgboost", "lightgbm", "catboost"])
def test_search_space_samples_valid_params(name):
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params(name, trial)
    assert params


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_pipeline("svm")


def test_tuning_and_oof_are_reproducible(xy):
    X, y = xy
    folds = make_folds(y, 3, seed=0)
    best_a, study = tune_model("logreg", X, y, folds, n_trials=4, seed=0)
    best_b, _ = tune_model("logreg", X, y, folds, n_trials=4, seed=0)
    assert best_a == best_b
    assert len(study.trials) == 4
    oof, scores = cross_val_oof("logreg", best_a, X, y, folds)
    assert oof.shape == (len(y),) and len(scores) == 3
    assert 0.5 < roc_auc_score(y, oof) <= 1
