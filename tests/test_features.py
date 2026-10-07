import numpy as np
import pandas as pd
import pytest

from pricing.features import FEATURES, add_lag_features, build_training_frame


def toy_panel(n_days=90, cats=("a", "b")):
    rng = np.random.default_rng(0)
    days = pd.date_range("2018-01-01", periods=n_days)
    rows = []
    for c in cats:
        rows.append(pd.DataFrame({
            "category": c, "date": days,
            "units": rng.poisson(10, n_days),
            "avg_price": rng.uniform(50, 60, n_days),
            "avg_freight": rng.uniform(10, 20, n_days),
            "n_sellers": rng.integers(5, 10, n_days),
            "platform_units": rng.poisson(200, n_days),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("lead", [1, 3, 7])
def test_last_value_is_lead_days_back(lead):
    panel = toy_panel()
    df = add_lag_features(panel, lead).set_index(["category", "date"])
    day = pd.Timestamp("2018-03-01")
    expected = panel.set_index(["category", "date"]).loc[("a", day - pd.Timedelta(days=lead)), "units"]
    assert df.loc[("a", day), "last_value"] == expected


@pytest.mark.parametrize("lead", [1, 4, 7])
def test_no_feature_uses_data_closer_than_lead(lead):
    """Scramble everything in the last `lead` days before a target day: features must not move."""
    panel = toy_panel()
    target = pd.Timestamp("2018-03-10")
    before = build_training_frame(panel, lead)

    changed = panel.copy()
    window = (changed["date"] > target - pd.Timedelta(days=lead)) & (changed["date"] <= target)
    for col in ["units", "avg_price", "avg_freight", "n_sellers", "platform_units"]:
        changed.loc[window, col] = changed.loc[window, col] * 10 + 1000
    after = build_training_frame(changed, lead)

    row_b = before[before["date"] == target][FEATURES].reset_index(drop=True)
    row_a = after[after["date"] == target][FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(row_b, row_a)


def test_lead_out_of_range():
    with pytest.raises(ValueError):
        add_lag_features(toy_panel(), 8)
