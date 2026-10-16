import numpy as np
import pytest

from churn_analysis.campaign import expected_best, simulate_campaign
from churn_analysis.decision import (
    best_threshold,
    campaign_profit,
    customer_economics,
    expected_offer_value,
    profit_curve,
)

BIZ = {"horizon_months": 12, "incentive_months": 1.0, "contact_cost": 5.0, "acceptance_rate": 0.3}


def test_customer_economics():
    v, d = customer_economics([50.0, 100.0], BIZ)
    assert v.tolist() == [600.0, 1200.0] and d.tolist() == [50.0, 100.0]


def test_expected_offer_value_hand_computed():
    # p=0.5, V=600, d=50, f=5, g=0.3 -> 0.5*0.3*550 - 0.5*50 - 5 = 82.5 - 25 - 5 = 52.5
    assert expected_offer_value(0.5, 600.0, 50.0, 5.0, 0.3) == pytest.approx(52.5)
    # a sure stayer is never worth contacting
    assert expected_offer_value(0.0, 600.0, 50.0, 5.0, 0.3) == pytest.approx(-55.0)


def test_campaign_profit_hand_computed():
    y = np.array([1, 0, 1])
    v, d = np.array([600.0, 600.0, 1200.0]), np.array([50.0, 50.0, 100.0])
    target = np.array([True, True, False])
    # churner: 0.3*550 - 5 = 160 ; stayer: -50 - 5 = -55
    assert campaign_profit(y, target, v, d, 5.0, 0.3) == pytest.approx(105.0)


def test_profit_curve_endpoints_and_best_threshold():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    p = np.clip(0.5 * y + rng.uniform(0, 0.5, 500), 0, 1)
    v, d = customer_economics(np.full(500, 70.0), BIZ)
    frac, prof = profit_curve(y, p, v, d, 5.0, 0.3)
    assert frac[0] == 0 and frac[-1] == 1 and prof[0] == 0
    assert prof[-1] == pytest.approx(campaign_profit(y, np.ones(500, bool), v, d, 5.0, 0.3))
    t, best = best_threshold(y, p, v, d, 5.0, 0.3)
    assert best >= campaign_profit(y, p >= 0.5, v, d, 5.0, 0.3)
    assert 0 <= t <= 1


def test_simulation_model_beats_random_on_informative_scores():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 800)
    p = np.clip(0.6 * y + rng.uniform(0, 0.4, 800), 0, 1)
    v, d = customer_economics(np.full(800, 70.0), BIZ)
    model = simulate_campaign(y, p, v, d, 5.0, 0.2, "model", n_sim=200, seed=0)
    rand = simulate_campaign(y, p, v, d, 5.0, 0.2, "random", n_sim=200, seed=0)
    assert model.mean() > rand.mean()
    ev = simulate_campaign(y, p, v, d, 5.0, 0.0, "ev", n_sim=50, seed=0, ev_acceptance=0.3)
    assert ev.shape == (50,)
    best, share, ev_profit, ev_share = expected_best(y, p, v, d, 5.0, 0.3)
    assert best >= ev_profit - 1e-9 and 0 <= share <= 1 and 0 <= ev_share <= 1


def test_paired_profit_diff_sign_and_identity():
    from churn_analysis.decision import paired_profit_diff

    y = np.array([1, 1, 0, 0])
    v, d = np.full(4, 600.0), np.full(4, 50.0)
    churners_only = y == 1
    everyone = np.ones(4, bool)
    out = paired_profit_diff(y, churners_only, everyone, v, d, 5.0, 0.3, n_boot=200)
    # dropping the two stayers saves 2 * 55 = 110 over 4 customers -> 27,500 per 1,000
    assert out["diff"] == pytest.approx(27_500)
    same = paired_profit_diff(y, everyone, everyone, v, d, 5.0, 0.3, n_boot=50)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
