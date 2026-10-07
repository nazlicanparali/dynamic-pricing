"""Price elasticity of demand per category.

Unit of analysis: one listing (product x seller) in one week. Only listings
that sold in at least 8 weeks and whose price actually moved are used.

Main model, per category (Poisson regression with fixed effects, "PPML"):

    E[units] = exp(elasticity * log(price) + b * log(1 + freight)
                   + listing effect + week effect)

- listing effects: a listing is only compared with itself, so "expensive
  products sell less" (a between-product effect) can't leak in
- week effects: remove anything that hits all listings in the same week
  (Black Friday, the truckers' strike, general growth of the platform)
- standard errors clustered by listing

Why Poisson and not the usual log-log OLS: about two thirds of listing-weeks
have zero sales, and log(0) is undefined. Dropping those weeks throws away
exactly the weeks where a high price meant nobody bought, which pulls the
estimate towards zero. I keep the log-log OLS on positive weeks as a
comparison, and the gap between the two is one of the main results.

Category estimates are noisy, so they are shrunk towards the overall mean
with an empirical Bayes step (DerSimonian-Laird random effects).
"""
import json
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import plotting
from .data import load_items, weekly_listing_panel
from .tracking import log_run

log = logging.getLogger(__name__)
plt = plotting.plt


def filter_listings(weekly: pd.DataFrame, min_weeks: int, min_price_cv: float) -> pd.DataFrame:
    """Keep listings that sold often enough and whose price actually changed."""
    g = weekly.groupby("listing")
    n_weeks = g["week"].transform("size")
    price_cv = g["price"].transform("std") / g["price"].transform("mean")
    return weekly[(n_weeks >= min_weeks) & (price_cv >= min_price_cv)].copy()


def add_zero_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Add the weeks between a listing's first and last sale where it sold nothing.

    Olist only records prices when something is sold, so for zero weeks I
    carry the last observed price forward. That's an assumption (the seller
    may have changed the price in a week with no sales) and it is listed in
    the README limitations.
    """
    parts = []
    for listing, g in df.groupby("listing", sort=False):
        weeks = pd.date_range(g["week"].min(), g["week"].max(), freq="7D")
        full = g.set_index("week").reindex(weeks)
        full["units"] = full["units"].fillna(0)
        full[["price", "freight", "category"]] = full[["price", "freight", "category"]].ffill()
        full["listing"] = listing
        parts.append(full.rename_axis("week").reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["log_price"] = np.log(out["price"])
    out["log_freight"] = np.log1p(out["freight"])
    return out[["listing", "week", "category", "units", "price", "log_price", "log_freight"]]


def _clusters(df: pd.DataFrame) -> np.ndarray:
    return df["listing"].astype("category").cat.codes.to_numpy()


def fit_ppml(df: pd.DataFrame, week_fe: bool = True) -> dict:
    """Poisson regression with listing (and week) dummies."""
    # weeks where no listing in the group sold anything have a week effect of
    # minus infinity; those rows carry no information, so drop them
    df = df[df.groupby("week")["units"].transform("sum") > 0]
    parts = [df[["log_price", "log_freight"]],
             pd.get_dummies(df["listing"], prefix="l", drop_first=True, dtype=float)]
    if week_fe:
        parts.append(pd.get_dummies(df["week"], prefix="w", drop_first=True, dtype=float))
    X = sm.add_constant(pd.concat(parts, axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels complains about the many dummies
        res = sm.GLM(df["units"], X, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": _clusters(df)})
    return {"elasticity": float(res.params["log_price"]), "se": float(res.bse["log_price"]),
            "n_obs": int(len(df)), "n_listings": int(df["listing"].nunique()),
            "share_zero_weeks": float((df["units"] == 0).mean())}


def fit_ols_positive(df: pd.DataFrame, week_fe: bool = True) -> dict:
    """The usual log-log regression, on weeks with at least one sale only."""
    df = df[df["units"] > 0].assign(log_units=lambda d: np.log(d["units"]))
    X = df[["log_price", "log_freight"]]
    if week_fe:
        X = pd.concat([X, pd.get_dummies(df["week"], prefix="w", drop_first=True, dtype=float)],
                      axis=1)
    X = X.loc[:, X.std() > 0]
    # listing effects removed by demeaning (Frisch-Waugh-Lovell)
    cols = list(X.columns)
    X_dm = X - X.groupby(df["listing"]).transform("mean")
    y_dm = df["log_units"] - df.groupby("listing")["log_units"].transform("mean")
    res = sm.OLS(y_dm, X_dm[cols]).fit(cov_type="cluster", cov_kwds={"groups": _clusters(df)})
    return {"elasticity": float(res.params["log_price"]), "se": float(res.bse["log_price"]),
            "n_obs": int(len(df)), "n_listings": int(df["listing"].nunique())}


def shrink(b: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Empirical Bayes shrinkage of noisy group estimates towards their mean.

    tau2 is the variance of the true elasticities across categories
    (DerSimonian-Laird moment estimator). An estimate with a big standard
    error gets pulled further towards the mean.
    """
    v = se ** 2
    w = 1 / v
    mu_fixed = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - mu_fixed) ** 2)
    k = len(b)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))

    w_re = 1 / (v + tau2)
    mu = np.sum(w_re * b) / np.sum(w_re)
    mu_se = np.sqrt(1 / np.sum(w_re))

    weight = tau2 / (tau2 + v)                   # 0 = fully shrunk, 1 = untouched
    b_shrunk = mu + weight * (b - mu)
    se_shrunk = np.sqrt(weight * v + (1 - weight) ** 2 * mu_se ** 2)
    prior = {"mean": float(mu), "mean_se": float(mu_se), "tau": float(np.sqrt(tau2)),
             "q_stat": float(q), "q_p_value": float(stats.chi2.sf(q, k - 1))}
    return b_shrunk, se_shrunk, prior


def plot_forest(est: pd.DataFrame, prior: dict, path):
    est = est.sort_values("elasticity")
    y = np.arange(len(est))
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(est) + 1.4))
    ax.errorbar(est["ppml_raw"], y + 0.18, xerr=1.96 * est["ppml_raw_se"], fmt="o",
                color=plotting.GREY, ms=4, lw=1, capsize=0, label="Poisson FE, raw (95% CI)")
    ax.errorbar(est["elasticity"], y, xerr=1.96 * est["se"], fmt="o", color=plotting.BLUE,
                ms=5, lw=1.6, capsize=0, label="Poisson FE, after shrinkage")
    ax.scatter(est["ols_positive"], y - 0.18, marker="x", color=plotting.ORANGE, s=22,
               label="log-log OLS, positive weeks only")
    ax.axvline(prior["mean"], color=plotting.BLUE, ls="--", lw=0.8)
    ax.axvline(0, color="black", lw=0.6)
    ax.axvline(-1, color=plotting.RED, ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({n})" for c, n in zip(est.index, est["n_listings"])], fontsize=8)
    ax.set_xlim(-6, 2.5)
    ax.set_xlabel("price elasticity  (number of listings in brackets, red line = -1)")
    ax.set_title("Within-listing price elasticity by category")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    plotting.save(fig, path)


def run(cfg: dict) -> pd.DataFrame:
    ec = cfg["elasticity"]
    out, figs = cfg["paths"]["outputs"], cfg["paths"]["figures"]

    items = load_items(cfg["data"]["raw_dir"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    weekly = weekly_listing_panel(items)
    listings = filter_listings(weekly, ec["min_weeks"], ec["min_price_cv"])
    panel = add_zero_weeks(listings)
    log.info("%s listings pass the filters (of %s); %s listing-weeks, %.0f%% of them with zero sales",
             f"{panel['listing'].nunique():,}", f"{weekly['listing'].nunique():,}",
             f"{len(panel):,}", 100 * (panel["units"] == 0).mean())

    rows = []
    for cat, sub in panel.groupby("category"):
        if sub["listing"].nunique() < ec["min_listings"]:
            continue
        ppml, ols = fit_ppml(sub), fit_ols_positive(sub)
        rows.append({"category": cat, "n_listings": ppml["n_listings"], "n_obs": ppml["n_obs"],
                     "share_zero_weeks": ppml["share_zero_weeks"],
                     "ppml_raw": ppml["elasticity"], "ppml_raw_se": ppml["se"],
                     "ols_positive": ols["elasticity"], "ols_positive_se": ols["se"]})
    est = pd.DataFrame(rows).set_index("category")

    b, se, prior = shrink(est["ppml_raw"].to_numpy(), est["ppml_raw_se"].to_numpy())
    est["elasticity"], est["se"] = b, se
    est["ci_low"], est["ci_high"] = b - 1.96 * se, b + 1.96 * se
    est["p_elastic"] = stats.norm.cdf((-1 - b) / se)      # P(elasticity < -1)

    # same comparison on all listings at once
    _, _, ols_prior = shrink(est["ols_positive"].to_numpy(), est["ols_positive_se"].to_numpy())
    comparison = pd.DataFrame([
        {"method": "log-log OLS, positive weeks, pooled (listing + week FE)",
         **fit_ols_positive(panel)},
        {"method": "log-log OLS, positive weeks, random-effects mean over categories",
         "elasticity": ols_prior["mean"], "se": ols_prior["mean_se"]},
        {"method": "Poisson FE incl. zero weeks, random-effects mean over categories (main)",
         "elasticity": prior["mean"], "se": prior["mean_se"]},
    ])

    log.info("random-effects mean elasticity: Poisson %.2f (se %.2f) vs log-log OLS %.2f (se %.2f)",
             prior["mean"], prior["mean_se"], ols_prior["mean"], ols_prior["mean_se"])
    for cat, r in est.sort_values("elasticity").iterrows():
        log.info("  %-24s Poisson %6.2f (se %.2f) -> shrunk %6.2f | OLS positive weeks %6.2f",
                 cat, r["ppml_raw"], r["ppml_raw_se"], r["elasticity"], r["ols_positive"])

    est.round(4).to_csv(out / "elasticities.csv")
    comparison.round(4).to_csv(out / "elasticity_method_comparison.csv", index=False)
    summary = {"prior": prior, "ols_positive_prior": ols_prior, "n_categories": int(len(est)),
               "n_listings": int(panel["listing"].nunique()), "n_listing_weeks": int(len(panel)),
               "share_zero_weeks": float((panel["units"] == 0).mean())}
    with open(out / "elasticity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_forest(est, prior, figs / "elasticity_by_category.png")
    log_run("elasticity", ec, {"mean_elasticity": prior["mean"], "mean_se": prior["mean_se"],
                               "tau": prior["tau"], "ols_mean_elasticity": ols_prior["mean"]}, out)
    return est
