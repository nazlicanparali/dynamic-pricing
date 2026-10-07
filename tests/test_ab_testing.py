import numpy as np
import pandas as pd

from pricing.ab_testing import aa_tests, analyse, assign, cuped, make_pairs


def fake_regions(n=300, seed=0):
    """Regions of very different size whose test-period value follows the pre-period."""
    rng = np.random.default_rng(seed)
    size = rng.lognormal(3, 1, n)
    pre = rng.poisson(size * 4)
    test = rng.poisson(size * 4 * rng.lognormal(0, 0.2, n))
    return pd.DataFrame({"pre_units": pre.astype(float), "test_units": test.astype(float)})


def test_aa_false_positive_rate_is_about_alpha():
    aa = aa_tests(fake_regions(), "units", n_sims=600, alpha=0.05, rng=np.random.default_rng(1))
    for design, g in aa.groupby("design"):
        fpr = (g["p_value"] < 0.05).mean()
        assert 0.02 < fpr < 0.09, (design, fpr)


def test_cuped_reduces_variance():
    r = fake_regions()
    y, x = r["test_units"].to_numpy(), r["pre_units"].to_numpy()
    assert cuped(y, x).var() < 0.5 * y.var()
    assert abs(cuped(y, x).mean() - y.mean()) < 1e-9  # same mean, less noise


def test_large_effect_is_detected():
    r = fake_regions()
    y, x = r["test_units"].to_numpy(), r["pre_units"].to_numpy()
    treat = assign(len(y), "cuped", np.random.default_rng(2))
    res = analyse(np.where(treat, y * 1.3, y), x, treat, "cuped")
    assert res["p_value"] < 0.01
    assert res["ci_low"] < 0.3 < res["ci_high"]


def test_pairs_cover_every_region_once():
    pairs = make_pairs(np.arange(11, dtype=float))
    assert pairs.shape == (5, 2)
    assert len(np.unique(pairs)) == 10
    treat = assign(11, "paired_cuped", np.random.default_rng(0), pairs)
    assert treat[pairs].sum(axis=1).tolist() == [1] * 5
