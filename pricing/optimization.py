"""Price recommendations per category from the elasticity estimates.

With constant elasticity e, a price change by factor r changes
    units   by  r**e
    revenue by  r**(1 + e)
    profit  by  ((r - 1 + m) / m) * r**e      (m = gross margin at today's price)

so the answer only depends on r, e and m, not on the price level.

Two things make this more than "plug in the formula":
1. Olist has no cost data, so the margin m is an assumption. The sensitivity
   plot shows how the recommendation flips with it.
2. The elasticities are uncertain. I draw from each category's posterior and
   pick the price change with the best *expected* profit, inside a +-10%
   guardrail. If the chance that the change loses money is too high, or the
   estimate doesn't even clearly say that demand falls when price rises, the
   recommendation is "don't change, run a test instead".
"""
import logging

import numpy as np
import pandas as pd

from . import plotting
from .data import load_items
from .tracking import log_run

log = logging.getLogger(__name__)
plt = plotting.plt


def profit_ratio(r, e, margin):
    """Profit after the change divided by profit before it."""
    return (r - 1 + margin) / margin * np.power(r, e)


def revenue_ratio(r, e):
    return np.power(r, 1 + e)


def unconstrained_optimum(e: float, margin: float) -> float:
    """Profit-maximising price factor for constant elasticity (Lerner rule).

    Only finite when demand is elastic (e < -1); otherwise profit keeps
    rising with price and the answer is "as high as you're allowed to go".
    """
    if e >= -1:
        return np.inf
    return (1 - margin) * e / (1 + e)


def recommend(e_hat: float, e_se: float, margin: float, max_change: float, step: float,
              n_draws: int, rng: np.random.Generator, min_prob_gain: float = 0.8,
              min_prob_slope: float = 0.9) -> dict:
    grid = np.round(np.arange(1 - max_change, 1 + max_change + 1e-9, step), 4)
    draws = rng.normal(e_hat, e_se, size=n_draws)

    lift = profit_ratio(grid[:, None], draws[None, :], margin) - 1   # grid x draws
    expected = lift.mean(axis=1)
    best = int(np.argmax(expected))
    r = grid[best]
    prob_gain = float((lift[best] > 0).mean()) if r != 1 else 1.0
    # an estimate around zero (or positive) means we haven't really measured how
    # demand reacts to price, so a big "profit gain" from it shouldn't be trusted
    prob_slope = float((draws < 0).mean())

    if prob_slope < min_prob_slope:
        action = "test first: no clear price response"
    elif prob_gain < min_prob_gain:
        action = "test first: too uncertain"
    else:
        action = "change price"
    return {
        "recommended_change": float(r - 1),
        "expected_profit_lift": float(expected[best]),
        "profit_lift_p05": float(np.quantile(lift[best], 0.05)),
        "profit_lift_p95": float(np.quantile(lift[best], 0.95)),
        "prob_profit_gain": prob_gain,
        "expected_revenue_lift": float((revenue_ratio(r, draws) - 1).mean()),
        "expected_units_change": float((np.power(r, draws) - 1).mean()),
        "point_estimate_optimum": float(unconstrained_optimum(e_hat, margin) - 1),
        "prob_negative_elasticity": prob_slope,
        "action": action,
    }


def plot_recommendations(rec: pd.DataFrame, path):
    rec = rec.sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(rec) + 1.3))
    colors = [plotting.GREEN if a == "change price" else plotting.GREY for a in rec["action"]]
    y = np.arange(len(rec))
    ax.barh(y, 100 * rec["recommended_change"], color=colors, height=0.6)
    for yi, (_, row) in zip(y, rec.iterrows()):
        x = 100 * row["recommended_change"]
        ax.text(x + (0.4 if x >= 0 else -0.4), yi,
                f"P(gain)={row['prob_profit_gain']:.0%}  e={row['elasticity']:.2f}",
                va="center", ha="left" if x >= 0 else "right", fontsize=7, color=plotting.GREY)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(rec.index, fontsize=8)
    ax.set_xlim(-16, 22)
    ax.set_xlabel("recommended price change (%)  -  green: act, grey: test first")
    ax.set_title("Profit-maximising price change (40% margin, +-10% guardrail)")
    plotting.save(fig, path)


def plot_margin_sensitivity(rec: pd.DataFrame, margin: float, path):
    """Unconstrained optimal price change over (elasticity, margin)."""
    e = np.linspace(-4, -1.05, 200)
    m = np.linspace(0.1, 0.7, 200)
    E, M = np.meshgrid(e, m)
    opt = 100 * ((1 - M) * E / (1 + E) - 1)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    levels = [-40, -20, -10, -5, 0, 5, 10, 20, 50, 100]
    cs = ax.contourf(E, M, np.clip(opt, -40, 100), levels=levels, cmap="RdBu_r", alpha=0.85)
    ax.contour(E, M, opt, levels=[0], colors="black", linewidths=1.2)
    fig.colorbar(cs, ax=ax, label="optimal price change (%)")
    shown = rec[rec["elasticity"] < -1.05]
    ax.scatter(shown["elasticity"], np.full(len(shown), margin), color="black", s=14, zorder=3)
    ax.axhline(margin, color="black", lw=0.6, ls=":")
    ax.text(-3.95, margin + 0.015, "categories at the assumed 40% margin", fontsize=7)
    ax.text(-3.6, 0.6, "cut prices", fontsize=10, color="white", weight="bold")
    ax.text(-1.75, 0.15, "raise prices", fontsize=10, color="white", weight="bold")
    ax.set_xlabel("price elasticity")
    ax.set_ylabel("gross margin")
    ax.set_title("Cut or raise? It depends on margin as much as elasticity")
    plotting.save(fig, path)


def run(cfg: dict) -> pd.DataFrame:
    oc = cfg["optimization"]
    out, figs = cfg["paths"]["outputs"], cfg["paths"]["figures"]
    rng = np.random.default_rng(cfg["seed"])

    est = pd.read_csv(out / "elasticities.csv", index_col="category")

    # typical current price per category = median item price in the last 8 weeks of data
    items = load_items(cfg["data"]["raw_dir"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    recent = items[items["date"] > items["date"].max() - pd.Timedelta(weeks=8)]
    median_price = recent.groupby("category")["price"].median()

    rows = []
    for cat, r in est.iterrows():
        res = recommend(r["elasticity"], r["se"], oc["gross_margin"], oc["max_price_change"],
                        oc["grid_step"], oc["n_draws"], rng)
        res.update(category=cat, elasticity=r["elasticity"], elasticity_se=r["se"],
                   median_price=median_price.get(cat, np.nan))
        res["recommended_price"] = res["median_price"] * (1 + res["recommended_change"])
        rows.append(res)
    rec = pd.DataFrame(rows).set_index("category")

    # how much does the margin assumption matter?
    sens = pd.DataFrame({
        f"margin_{m:.0%}": [recommend(r["elasticity"], r["se"], m, oc["max_price_change"],
                                      oc["grid_step"], oc["n_draws"], rng)["recommended_change"]
                            for _, r in est.iterrows()]
        for m in (0.2, 0.3, 0.4, 0.5, 0.6)
    }, index=est.index)

    rec.round(4).to_csv(out / "price_recommendations.csv")
    sens.round(3).to_csv(out / "price_recommendations_margin_sensitivity.csv")
    plot_recommendations(rec, figs / "price_recommendations.png")
    plot_margin_sensitivity(rec, oc["gross_margin"], figs / "margin_sensitivity.png")

    acted = rec[rec["action"] == "change price"]
    log.info("%d of %d categories get a confident recommendation; mean change %.1f%%, "
             "expected profit lift %.1f%% (revenue %.1f%%)", len(acted), len(rec),
             100 * acted["recommended_change"].mean(), 100 * acted["expected_profit_lift"].mean(),
             100 * acted["expected_revenue_lift"].mean())
    for cat, r in rec.sort_values("elasticity").iterrows():
        log.info("  %-24s e=%5.2f  change %+4.0f%%  profit %+5.1f%%  P(gain) %3.0f%%  -> %s",
                 cat, r["elasticity"], 100 * r["recommended_change"],
                 100 * r["expected_profit_lift"], 100 * r["prob_profit_gain"], r["action"])

    log_run("optimize", oc, {"n_confident": int(len(acted)), "n_categories": int(len(rec)),
                             "mean_change_confident": float(acted["recommended_change"].mean()),
                             "mean_profit_lift_confident": float(acted["expected_profit_lift"].mean())},
            out)
    return rec
