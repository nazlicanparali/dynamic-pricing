import numpy as np
import pytest

from pricing.rl_agent import train_q_learning
from pricing.simulator import PricingEnv, evaluate_static, policy_value, simulate, solve_dp


@pytest.fixture
def env():
    return PricingEnv(base_demand=1.0, weekday_factors=np.ones(7), elasticity=-2.0,
                      multipliers=np.array([0.8, 1.0, 1.2, 1.4]), horizon=8, inventory=6,
                      salvage=0.3)


def test_dp_beats_every_static_price(env):
    V, policy = solve_dp(env)
    for a in range(env.n_actions):
        assert V[0, env.inventory] >= evaluate_static(env, a) - 1e-9


def test_policy_value_matches_dp(env):
    V, policy = solve_dp(env)
    assert policy_value(env, policy) == pytest.approx(V[0, env.inventory])


def test_simulation_agrees_with_exact_value(env):
    V, policy = solve_dp(env)
    u = np.random.default_rng(0).random((20000, env.horizon))
    sim = simulate(env, policy, u)
    assert sim["revenue"].mean() == pytest.approx(V[0, env.inventory], rel=0.02)
    assert np.all(sim["leftover"] >= 0)


def test_q_learning_gets_close_to_optimum(env):
    V, _ = solve_dp(env)
    Q, _, _ = train_q_learning(env, episodes=20000, alpha_power=0.7, eps_start=1.0, eps_end=0.01,
                               rng=np.random.default_rng(0))
    assert policy_value(env, Q.argmax(axis=2)) / V[0, env.inventory] > 0.97
