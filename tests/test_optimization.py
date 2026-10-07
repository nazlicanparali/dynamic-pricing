import numpy as np
import pytest

from pricing.optimization import profit_ratio, recommend, unconstrained_optimum


def test_no_change_means_no_lift():
    assert profit_ratio(1.0, -2.0, 0.4) == pytest.approx(1.0)


def test_lerner_rule():
    # e = -3, margin 40% -> cost = 0.6 -> optimal price = 0.6 * 3/2 = 0.9
    assert unconstrained_optimum(-3.0, 0.4) == pytest.approx(0.9)
    # numerical check of the closed form
    grid = np.linspace(0.5, 1.5, 10001)
    best = grid[np.argmax(profit_ratio(grid, -3.0, 0.4))]
    assert best == pytest.approx(0.9, abs=1e-3)


def test_inelastic_demand_has_no_interior_optimum():
    assert unconstrained_optimum(-0.5, 0.4) == np.inf


def test_recommendation_respects_guardrail():
    rng = np.random.default_rng(0)
    for e in [-0.3, -1.5, -6.0]:
        rec = recommend(e, 0.2, 0.4, max_change=0.1, step=0.01, n_draws=2000, rng=rng)
        assert -0.1 - 1e-9 <= rec["recommended_change"] <= 0.1 + 1e-9


def test_very_elastic_demand_gets_a_price_cut():
    rec = recommend(-6.0, 0.05, 0.4, 0.1, 0.01, 2000, np.random.default_rng(0))
    assert rec["recommended_change"] < 0
    assert rec["action"] == "change price"


def test_noisy_estimate_is_sent_to_testing():
    rec = recommend(-0.3, 1.0, 0.4, 0.1, 0.01, 2000, np.random.default_rng(0))
    assert rec["action"].startswith("test first")
