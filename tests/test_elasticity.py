import numpy as np
import pandas as pd

from pricing.elasticity import fit_ols_positive, fit_ppml, shrink

TRUE_E = -1.5


def synthetic_listings(n_listings=60, n_weeks=40, seed=1):
    """Listings with their own demand level, common week shocks and random price changes."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2018-01-01", periods=n_weeks, freq="7D")
    week_effect = rng.normal(0, 0.3, n_weeks)
    rows = []
    for i in range(n_listings):
        level = rng.normal(-0.3, 0.5)
        base_price = rng.uniform(20, 200)
        price = base_price * np.exp(rng.normal(0, 0.15, n_weeks))
        freight = rng.uniform(5, 30, n_weeks)
        lam = np.exp(level + week_effect + TRUE_E * np.log(price / base_price))
        rows.append(pd.DataFrame({"listing": f"l{i}", "week": weeks, "price": price,
                                  "log_price": np.log(price), "log_freight": np.log1p(freight),
                                  "units": rng.poisson(lam)}))
    return pd.concat(rows, ignore_index=True)


def test_ppml_recovers_true_elasticity():
    res = fit_ppml(synthetic_listings())
    assert abs(res["elasticity"] - TRUE_E) < 3 * res["se"]
    assert abs(res["elasticity"] - TRUE_E) < 0.3


def test_dropping_zero_weeks_biases_towards_zero():
    """The reason the main model is Poisson and not log-log OLS."""
    df = synthetic_listings()
    assert (df["units"] == 0).mean() > 0.3  # plenty of zero weeks, like the real data
    ols, ppml = fit_ols_positive(df), fit_ppml(df)
    assert abs(ols["elasticity"]) < abs(ppml["elasticity"])


def test_shrinkage_pulls_noisy_estimates_harder():
    b = np.array([-1.0, -1.2, -0.8, -3.0, -1.1])
    se = np.array([0.2, 0.2, 0.2, 1.5, 0.2])
    shrunk, shrunk_se, prior = shrink(b, se)
    assert np.all(shrunk_se <= se + 1e-12)
    # the -3.0 estimate is very uncertain, it should move the most
    moves = np.abs(shrunk - b)
    assert moves.argmax() == 3
    assert min(b) <= shrunk.min() and shrunk.max() <= max(b)


def test_no_spread_means_full_shrinkage():
    b = np.array([-1.0, -1.01, -0.99, -1.0])
    shrunk, _, prior = shrink(b, np.full(4, 0.5))
    assert prior["tau"] == 0
    np.testing.assert_allclose(shrunk, prior["mean"])
