"""Daily demand forecast per category, 1 to 7 days ahead.

Setup
- one global model for all 15 categories (more data per model than 15 separate ones)
- one model per lead time h = 1..7 ("direct" multi-step forecasting)
- Poisson loss, since the target is a daily count and many days are small
- the target is divided by the category's recent 28-day level, so the model
  learns relative patterns that transfer between big and small categories

Evaluation is a rolling-origin backtest: 6 folds of 28 days, the models are
retrained at the start of each fold, and a lead-h forecast only ever sees
data up to h days before the day it predicts.
"""
import json
import logging

import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator, ScalarFormatter
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from . import plotting
from .data import daily_category_panel, load_items, top_categories
from .features import FEATURES, build_training_frame
from .tracking import log_run

log = logging.getLogger(__name__)
plt = plotting.plt


class ScaledPoissonGBM(BaseEstimator, RegressorMixin):
    """Gradient boosting on y / level with sample weight = level.

    With a Poisson loss this is the same as fitting log(level) as an offset,
    which HistGradientBoosting doesn't support directly.
    """

    def __init__(self, level_col="roll_mean_28", seed=0, **params):
        self.level_col = level_col
        self.seed = seed
        self.params = params

    def _level(self, X):
        return X[self.level_col].to_numpy() + 0.5

    def fit(self, X, y):
        level = self._level(X)
        self.model_ = HistGradientBoostingRegressor(
            loss="poisson", categorical_features=[FEATURES.index("category_code")],
            random_state=self.seed, **self.params)
        self.model_.fit(X[FEATURES], np.asarray(y) / level, sample_weight=level)
        return self

    def predict(self, X):
        return self.model_.predict(X[FEATURES]) * self._level(X)


def wape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).sum() / y_true.sum())


def bias(y_true, y_pred) -> float:
    return float((np.asarray(y_pred) - np.asarray(y_true)).sum() / np.asarray(y_true).sum())


def poisson_mad(lam):
    """E|Y - lam| for Y ~ Poisson(lam). Used as a noise floor for the errors."""
    lam = np.asarray(lam, dtype=float)
    return 2 * lam * stats.poisson.pmf(np.floor(lam), lam)


def fold_starts(last_day: pd.Timestamp, n_folds: int, fold_days: int) -> list[pd.Timestamp]:
    return [last_day - pd.Timedelta(days=fold_days * (n_folds - k) - 1) for k in range(n_folds)]


def backtest(frames: dict[int, pd.DataFrame], cfg: dict, seed: int) -> pd.DataFrame:
    fc = cfg["forecast"]
    last_day = frames[1]["date"].max()
    preds = []
    for k, start in enumerate(fold_starts(last_day, fc["n_folds"], fc["fold_days"])):
        end = start + pd.Timedelta(days=fc["fold_days"])
        for lead, df in frames.items():
            train = df[df["date"] < start]
            test = df[(df["date"] >= start) & (df["date"] < end)].copy()
            model = ScaledPoissonGBM(seed=seed, **fc["model"]).fit(train, train["units"])
            test["pred_model"] = model.predict(test)
            test["pred_naive"] = test["lag_7"]              # same weekday last week
            test["pred_dow_mean"] = test["same_dow_mean"]   # same weekday, last 4 weeks
            test["fold"], test["lead"] = k, lead
            preds.append(test)
        fold = pd.concat(preds[-len(frames):])
        log.info("fold %d (%s .. %s): WAPE model %.3f (lead 1: %.3f, lead 7: %.3f) | naive %.3f | "
                 "weekday mean %.3f", k, start.date(), (end - pd.Timedelta(days=1)).date(),
                 wape(fold["units"], fold["pred_model"]),
                 wape(*fold.loc[fold["lead"] == 1, ["units", "pred_model"]].T.values),
                 wape(*fold.loc[fold["lead"] == 7, ["units", "pred_model"]].T.values),
                 wape(fold["units"], fold["pred_naive"]), wape(fold["units"], fold["pred_dow_mean"]))
    return pd.concat(preds, ignore_index=True)


def summarise(bt: pd.DataFrame) -> dict:
    """Metrics pooled over all leads (= a forecast made at a random time of the week)."""
    out = {}
    baselines = bt[bt["lead"] == 1]  # baselines don't depend on the lead
    for name, frame in [("model", bt), ("naive", baselines), ("dow_mean", baselines)]:
        col = f"pred_{name}"
        total = frame.groupby(["date", "lead"])[["units", col]].sum()
        out[name] = {
            "wape": wape(frame["units"], frame[col]),
            "mae": float(np.abs(frame["units"] - frame[col]).mean()),
            "bias": bias(frame["units"], frame[col]),
            # forecasts summed over the 15 categories -> daily total for the platform
            "wape_total": wape(total["units"], total[col]),
        }
    out["model_by_lead"] = {int(h): wape(g["units"], g["pred_model"]) for h, g in bt.groupby("lead")}
    out["poisson_floor_wape"] = float(poisson_mad(bt["pred_model"]).sum() / bt["units"].sum())
    out["wape_reduction_vs_naive"] = 1 - out["model"]["wape"] / out["naive"]["wape"]
    out["wape_reduction_vs_dow_mean"] = 1 - out["model"]["wape"] / out["dow_mean"]["wape"]
    return out


def error_analysis(bt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bt = bt.assign(abs_err=(bt["units"] - bt["pred_model"]).abs(),
                   floor=poisson_mad(bt["pred_model"]))

    by_cat = bt.groupby("category").agg(avg_daily_units=("units", "mean"), abs_err=("abs_err", "sum"),
                                        floor=("floor", "sum"), units=("units", "sum"))
    by_cat["wape_model"] = by_cat["abs_err"] / by_cat["units"]
    by_cat["poisson_floor"] = by_cat["floor"] / by_cat["units"]
    lead1 = bt[bt["lead"] == 1]
    by_cat["wape_naive"] = lead1.groupby("category").apply(
        lambda g: wape(g["units"], g["pred_naive"]), include_groups=False)
    by_cat["wape_dow_mean"] = lead1.groupby("category").apply(
        lambda g: wape(g["units"], g["pred_dow_mean"]), include_groups=False)
    by_cat = by_cat.drop(columns=["abs_err", "floor", "units"]).sort_values(
        "avg_daily_units", ascending=False)

    by_dow = bt.groupby("dow").apply(
        lambda g: pd.Series({"wape_model": wape(g["units"], g["pred_model"]),
                             "bias_model": bias(g["units"], g["pred_model"])}),
        include_groups=False)
    by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # biggest misses at platform level, for the weekly-planning forecast (lead 7)
    total = bt[bt["lead"] == 7].groupby("date")[["units", "pred_model"]].sum()
    total["error"] = total["pred_model"] - total["units"]
    total["pct_error"] = total["error"] / total["units"]
    worst_days = total.reindex(total["error"].abs().sort_values(ascending=False).index).head(15)

    return {"by_category": by_cat, "by_dow": by_dow, "worst_days": worst_days}


def plot_backtest(bt: pd.DataFrame, path):
    total = bt.pivot_table(index="date", columns="lead", values="pred_model", aggfunc="sum")
    actual = bt[bt["lead"] == 1].groupby("date")["units"].sum()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(actual.index, actual, color="black", lw=1.2, label="actual")
    ax.plot(total.index, total[1], color=plotting.BLUE, lw=1.2, label="1 day ahead")
    ax.plot(total.index, total[7], color=plotting.ORANGE, lw=1.2, ls="--", label="7 days ahead")
    for start in bt.groupby("fold")["date"].min():
        ax.axvline(start, color=plotting.GREY, lw=0.6, alpha=0.5)
    # the two biggest misses line up with outside events (see README)
    for start, end, label in [("2018-05-21", "2018-05-31", " truckers' strike"),
                              ("2018-07-06", "2018-07-15", " end of World Cup")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=plotting.GREY, alpha=0.12, lw=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.98, label, fontsize=8, va="top",
                color=plotting.GREY)
    ax.set_title("Backtest: daily items sold, summed over the top 15 categories")
    ax.set_ylabel("items per day")
    ax.legend(loc="lower left", frameon=False, ncol=3)
    plotting.save(fig, path)


def plot_error_by_lead(metrics: dict, path):
    leads = list(metrics["model_by_lead"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(leads, list(metrics["model_by_lead"].values()), marker="o", color=plotting.BLUE,
            label="gradient boosting")
    ax.axhline(metrics["dow_mean"]["wape"], color=plotting.GREY, ls="--",
               label="same weekday, 4-week mean")
    ax.axhline(metrics["naive"]["wape"], color=plotting.GREY, ls=":", label="same day last week")
    ax.axhline(metrics["poisson_floor_wape"], color=plotting.ORANGE, lw=1, label="Poisson noise floor")
    ax.set_xlabel("lead time (days ahead)")
    ax.set_ylabel("WAPE (category x day)")
    ax.set_title("Error grows with lead time")
    ax.legend(frameon=False, fontsize=8)
    plotting.save(fig, path)


def plot_error_by_category(by_cat: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = by_cat["avg_daily_units"]
    ax.scatter(x, by_cat["wape_dow_mean"], color=plotting.GREY, label="weekday mean baseline", s=26)
    ax.scatter(x, by_cat["wape_model"], color=plotting.BLUE, label="model", s=32)
    order = np.argsort(x.to_numpy())
    ax.plot(x.to_numpy()[order], by_cat["poisson_floor"].to_numpy()[order], color=plotting.ORANGE,
            lw=1.5, label="Poisson noise floor")
    for cat in ["health_beauty", "bed_bath_table", "sports_leisure", "electronics", "cool_stuff",
                "garden_tools"]:
        if cat in by_cat.index:
            row = by_cat.loc[cat]
            ax.annotate(cat, (row["avg_daily_units"], row["wape_model"]), fontsize=7,
                        xytext=(4, -10), textcoords="offset points", color=plotting.GREY)
    ax.set_xscale("log")
    ax.set_xticks([5, 7, 10, 15, 20, 30])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("average items per day (log scale)")
    ax.set_ylabel("WAPE")
    ax.set_title("Error by category")
    ax.legend(frameon=False, fontsize=8)
    plotting.save(fig, path)


def plot_importance(imp: pd.Series, path):
    # roll_mean_28 is also the level the target is divided by, so shuffling it rescales every
    # prediction and it dwarfs everything else. Leave it out of the bars and just print it.
    level = imp["roll_mean_28"]
    imp = imp.drop("roll_mean_28").sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.barh(imp.index, imp.values, color=plotting.BLUE)
    ax.set_xlabel("increase in MAE when the feature is shuffled")
    ax.set_title("Permutation importance (lead-1 model, last fold)")
    ax.text(0.98, 0.04, f"not shown: roll_mean_28 = {level:.2f}\n(it also sets the scale)",
            transform=ax.transAxes, ha="right", fontsize=8, color=plotting.GREY)
    plotting.save(fig, path)


def forecast_next_days(panel: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Fit each lead model on all data and forecast the week after the data ends."""
    horizon = cfg["forecast"]["horizon"]
    last = panel["date"].max()
    future = pd.MultiIndex.from_product(
        [panel["category"].unique(), pd.date_range(last + pd.Timedelta(days=1), periods=horizon)],
        names=["category", "date"]).to_frame(index=False)
    extended = pd.concat([panel, future], ignore_index=True)

    rows = []
    for lead in range(1, horizon + 1):
        df = build_training_frame(extended, lead)
        train = df[df["date"] <= last]
        model = ScaledPoissonGBM(seed=seed, **cfg["forecast"]["model"]).fit(train, train["units"])
        day = df[df["date"] == last + pd.Timedelta(days=lead)].copy()
        day["forecast"] = model.predict(day)
        rows.append(day)
    fut = pd.concat(rows)
    # rough 80% interval from the Poisson assumption (real demand is more spread out than this)
    fut["lower_80"] = stats.poisson.ppf(0.10, fut["forecast"])
    fut["upper_80"] = stats.poisson.ppf(0.90, fut["forecast"])
    fut["forecast"] = fut["forecast"].round(2)
    return fut[["category", "date", "forecast", "lower_80", "upper_80"]]


def run(cfg: dict) -> dict:
    seed = cfg["seed"]
    out, figs = cfg["paths"]["outputs"], cfg["paths"]["figures"]

    items = load_items(cfg["data"]["raw_dir"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    cats = top_categories(items, cfg["forecast"]["n_categories"])
    panel = daily_category_panel(items, cats)
    log.info("top %d categories cover %.0f%% of items", len(cats),
             100 * items["category"].isin(cats).mean())

    frames = {h: build_training_frame(panel, h) for h in range(1, cfg["forecast"]["horizon"] + 1)}
    bt = backtest(frames, cfg, seed)
    metrics = summarise(bt)
    analysis = error_analysis(bt)

    # permutation importance for the lead-1 model on the last fold
    last_start = bt.loc[bt["fold"] == bt["fold"].max(), "date"].min()
    df1 = frames[1]
    train, test = df1[df1["date"] < last_start], df1[df1["date"] >= last_start]
    model = ScaledPoissonGBM(seed=seed, **cfg["forecast"]["model"]).fit(train, train["units"])
    pi = permutation_importance(model, test, test["units"], scoring="neg_mean_absolute_error",
                                n_repeats=5, random_state=seed)
    importance = pd.Series(pi.importances_mean, index=test.columns)[FEATURES]

    fut = forecast_next_days(panel, cfg, seed)

    bt[["category", "date", "fold", "lead", "units", "pred_model", "pred_naive", "pred_dow_mean"]] \
        .round({"pred_model": 3}).to_csv(out / "forecast_backtest.csv", index=False)
    analysis["by_category"].round(4).to_csv(out / "forecast_error_by_category.csv")
    analysis["by_dow"].round(4).to_csv(out / "forecast_error_by_weekday.csv")
    analysis["worst_days"].round(2).to_csv(out / "forecast_worst_days.csv")
    importance.sort_values(ascending=False).round(4).to_csv(
        out / "forecast_feature_importance.csv", header=["mae_increase"])
    fut.to_csv(out / "forecast_next_7_days.csv", index=False)
    with open(out / "forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_backtest(bt, figs / "forecast_backtest.png")
    plot_error_by_lead(metrics, figs / "forecast_error_by_lead.png")
    plot_error_by_category(analysis["by_category"], figs / "forecast_error_by_category.png")
    plot_importance(importance, figs / "forecast_importance.png")

    log.info("WAPE (all leads): model %.3f | naive %.3f | weekday mean %.3f | Poisson floor %.3f",
             metrics["model"]["wape"], metrics["naive"]["wape"], metrics["dow_mean"]["wape"],
             metrics["poisson_floor_wape"])
    log_run("forecast", cfg["forecast"], metrics, out, [out / "forecast_metrics.json"])
    return metrics
