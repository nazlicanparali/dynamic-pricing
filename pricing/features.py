"""Feature engineering for the demand forecast.

I train one model per lead time h (1..7 days ahead). For lead h, a row for
day d may only use information from day d - h or earlier, plus things that
are known in advance (calendar, holidays). tests/test_features.py checks
that nothing closer than h days leaks in.
"""
import numpy as np
import pandas as pd

# National holidays in the data window. Carnival Monday/Tuesday is not an
# official holiday but almost everyone treats it as one.
BR_HOLIDAYS = pd.to_datetime([
    "2017-01-01", "2017-02-27", "2017-02-28", "2017-04-14", "2017-04-21", "2017-05-01",
    "2017-06-15", "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
    "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01",
    "2018-05-31", "2018-09-07", "2018-10-12", "2018-11-02", "2018-11-15", "2018-12-25",
])
BLACK_FRIDAYS = pd.to_datetime(["2017-11-24", "2018-11-23"])

WEEKLY_LAGS = ["lag_7", "lag_14", "lag_21", "lag_28"]

FEATURES = [
    "category_code", "dow", "day_of_month", "is_holiday", "black_friday_week",
    "last_value", *WEEKLY_LAGS, "roll_mean_7", "roll_mean_28", "roll_std_28",
    "same_dow_mean", "trend_ratio",
    "price_ratio", "avg_price_r28", "n_sellers_r28", "avg_freight_r28",
    "platform_units_r7", "platform_trend",
]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["day_of_month"] = d.dt.day
    df["is_holiday"] = d.isin(BR_HOLIDAYS).astype(int)
    # Black Friday "week": Monday before to Monday after (Cyber Monday)
    bf = np.zeros(len(df), dtype=bool)
    for day in BLACK_FRIDAYS:
        bf |= ((d >= day - pd.Timedelta(days=4)) & (d <= day + pd.Timedelta(days=3))).to_numpy()
    df["black_friday_week"] = bf.astype(int)
    return df


def _rolling_mean(s: pd.Series, by: pd.Series, window: int) -> pd.Series:
    return s.groupby(by, sort=False).transform(lambda x: x.rolling(window, min_periods=window).mean())


def add_lag_features(panel: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Lagged target and lagged exogenous series for a given lead time (1..7)."""
    if not 1 <= lead <= 7:
        raise ValueError("lead must be between 1 and 7 (the weekly lags assume it)")

    df = panel.sort_values(["category", "date"]).copy()
    cat = df["category"]
    g = df.groupby("category", sort=False)

    df["last_value"] = g["units"].shift(lead)          # most recent known day
    for lag in (7, 14, 21, 28):                         # same weekday, 1-4 weeks back
        df[f"lag_{lag}"] = g["units"].shift(lag)

    # rolling windows that end `lead` days before the target day
    recent = g["units"].shift(lead)
    df["roll_mean_7"] = _rolling_mean(recent, cat, 7)
    df["roll_mean_28"] = _rolling_mean(recent, cat, 28)
    df["roll_std_28"] = recent.groupby(cat, sort=False).transform(
        lambda x: x.rolling(28, min_periods=28).std())
    df["same_dow_mean"] = df[WEEKLY_LAGS].mean(axis=1)
    df["trend_ratio"] = df["roll_mean_7"] / (df["roll_mean_28"] + 1)

    # exogenous series: what's happening to prices, supply and the whole site
    for col, window in [("avg_price", 7), ("avg_price", 28), ("n_sellers", 28),
                        ("platform_units", 7), ("platform_units", 28), ("avg_freight", 28)]:
        df[f"{col}_r{window}"] = _rolling_mean(g[col].shift(lead), cat, window)
    df["price_ratio"] = df["avg_price_r7"] / df["avg_price_r28"]
    df["platform_trend"] = df["platform_units_r7"] / df["platform_units_r28"]

    return add_calendar(df)


def build_training_frame(panel: pd.DataFrame, lead: int) -> pd.DataFrame:
    df = add_lag_features(panel, lead)
    df["category_code"] = df["category"].astype("category").cat.codes
    # the first ~35 days of each series don't have complete windows yet
    return df.dropna(subset=FEATURES).reset_index(drop=True)
