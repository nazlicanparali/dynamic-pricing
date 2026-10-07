"""Designing a price experiment: geo-randomised test with CUPED.

Showing two customers different prices for the same product at the same time
is a bad idea (and they talk to each other), so price tests usually
randomise regions instead of users. Here a region is a 3-digit CEP (zip)
prefix: there are a few hundred with enough orders and none of them is more
than ~2% of revenue, unlike cities where Sao Paulo alone is ~15%.

Olist has no traffic data and no real price test, so:
1. A/A tests on the real data: random splits with no treatment. A design is
   only usable if it rejects ~5% of the time at alpha = 5%.
2. Noise ladder: how much each design choice (CUPED, winsorising, longer
   pre-period, longer test, a different metric) shrinks the noise.
3. Power: inject a known lift into the real data and count how often the
   test detects it.
4. One simulated run of the actual test: +10% price in treated regions, with
   demand reacting according to the estimated elasticity.

The main lesson (see README): at this scale the test cannot see a 1-2%
revenue effect, but it can measure how much *volume* drops, which gives the
elasticity, which is the number the pricing model actually needs.
"""
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import plotting
from .data import load_items
from .tracking import log_run

log = logging.getLogger(__name__)
plt = plotting.plt

DESIGNS = ["simple", "cuped", "paired_cuped"]
LABELS = {"simple": "random split, plain t-test", "cuped": "random split + CUPED",
          "paired_cuped": "matched pairs + CUPED"}


def winsorize_prices(items: pd.DataFrame, q: float | None) -> pd.DataFrame:
    """Cap item prices at the q-quantile so a few very expensive orders don't dominate."""
    if not q:
        return items
    return items.assign(price=items["price"].clip(upper=items["price"].quantile(q)))


def region_table(items: pd.DataFrame, unit: str, test_start: str, pre_weeks: int,
                 test_weeks: int, min_pre_orders: int) -> pd.DataFrame:
    start = pd.Timestamp(test_start)
    pre = items[(items["date"] >= start - pd.Timedelta(weeks=pre_weeks)) & (items["date"] < start)]
    test = items[(items["date"] >= start) & (items["date"] < start + pd.Timedelta(weeks=test_weeks))]

    def agg(df, prefix):
        return df.groupby(unit).agg(**{f"{prefix}_orders": ("order_id", "nunique"),
                                       f"{prefix}_units": ("order_item_id", "size"),
                                       f"{prefix}_revenue": ("price", "sum")})

    regions = agg(pre, "pre").join(agg(test, "test"), how="left").fillna(0)
    regions = regions[regions["pre_orders"] >= min_pre_orders]
    # put the pre-period on the same time scale as the test period
    for col in ["pre_orders", "pre_units", "pre_revenue"]:
        regions[col] *= test_weeks / pre_weeks
    return regions.sort_values("pre_revenue", ascending=False)


def make_pairs(pre_metric: np.ndarray) -> np.ndarray:
    """Pair regions with similar pre-period values. Returns (n_pairs, 2) indices."""
    order = np.argsort(-pre_metric)
    if len(order) % 2:
        order = order[:-1]  # drop the smallest region if the count is odd
    return order.reshape(-1, 2)


def assign(n: int, design: str, rng: np.random.Generator, pairs: np.ndarray | None = None):
    treat = np.zeros(n, dtype=bool)
    if design == "paired_cuped":
        flip = rng.random(len(pairs)) < 0.5
        treat[np.where(flip, pairs[:, 0], pairs[:, 1])] = True
    else:
        treat[rng.permutation(n)[: n // 2]] = True
    return treat


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y minus the part explained by the pre-period covariate x."""
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def analyse(y: np.ndarray, x: np.ndarray, treat: np.ndarray, design: str,
            pairs: np.ndarray | None = None, alpha: float = 0.05) -> dict:
    """Effect of treatment on the mean of y, absolute and relative to control."""
    y_adj = y if design == "simple" else cuped(y, x)
    control_mean = y[~treat].mean()

    if design == "paired_cuped":
        t_idx = np.where(treat[pairs[:, 0]], pairs[:, 0], pairs[:, 1])
        c_idx = np.where(treat[pairs[:, 0]], pairs[:, 1], pairs[:, 0])
        d = y_adj[t_idx] - y_adj[c_idx]
        effect, se, df = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1
    else:
        a, b = y_adj[treat], y_adj[~treat]
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        effect, se = a.mean() - b.mean(), np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))  # Welch

    p = 2 * stats.t.sf(abs(effect / se), df)
    half = stats.t.ppf(1 - alpha / 2, df) * se
    # relative numbers treat the control mean as fixed - fine at these sample sizes
    return {"effect": effect, "relative_effect": effect / control_mean,
            "ci_low": (effect - half) / control_mean, "ci_high": (effect + half) / control_mean,
            "p_value": p, "se": se}


def aa_tests(regions: pd.DataFrame, metric: str, n_sims: int, alpha: float, rng,
             designs=DESIGNS) -> pd.DataFrame:
    y, x = regions[f"test_{metric}"].to_numpy(), regions[f"pre_{metric}"].to_numpy()
    pairs = make_pairs(x)
    rows = []
    for _ in range(n_sims):
        for design in designs:
            res = analyse(y, x, assign(len(y), design, rng, pairs), design, pairs, alpha)
            rows.append({"design": design, "p_value": res["p_value"],
                         "relative_effect": res["relative_effect"]})
    return pd.DataFrame(rows)


def noise_ladder(items: pd.DataFrame, ac: dict, rng, n_sims: int = 1000) -> pd.DataFrame:
    """Std of the A/A estimate as design choices are added one at a time.

    The minimum detectable effect at 80% power / 5% alpha is about 2.8 x this std.
    """
    steps = [
        ("revenue, 4-week test, plain t-test", dict(pre=8, test=4, w=None, metric="revenue", d="simple")),
        ("+ CUPED (8-week pre-period)", dict(pre=8, test=4, w=None, metric="revenue", d="cuped")),
        ("+ winsorised prices", dict(pre=8, test=4, w=ac["winsorize_q"], metric="revenue", d="cuped")),
        ("+ 16-week pre-period", dict(pre=16, test=4, w=ac["winsorize_q"], metric="revenue", d="cuped")),
        ("+ 8-week test", dict(pre=16, test=8, w=ac["winsorize_q"], metric="revenue", d="cuped")),
        ("+ matched pairs", dict(pre=16, test=8, w=ac["winsorize_q"], metric="revenue",
                                 d="paired_cuped")),
        ("units instead of revenue", dict(pre=16, test=8, w=None, metric="units", d="paired_cuped")),
    ]
    rows = []
    for name, s in steps:
        start = ac["test_start"] if s["test"] == ac["test_weeks"] else \
            str((pd.Timestamp(ac["test_start"]) + pd.Timedelta(weeks=ac["test_weeks"] - s["test"])).date())
        regions = region_table(winsorize_prices(items, s["w"]), ac["unit"], start, s["pre"], s["test"],
                               ac["min_pre_orders"])
        aa = aa_tests(regions, s["metric"], n_sims, ac["alpha"], rng, designs=[s["d"]])
        sd = float(aa["relative_effect"].std())
        rows.append({"step": name, "n_regions": len(regions), "aa_std": sd, "approx_mde": 2.8 * sd})
    return pd.DataFrame(rows)


def power_curves(regions: pd.DataFrame, metric: str, lifts: list[float], n_sims: int,
                 alpha: float, rng) -> pd.DataFrame:
    """Share of simulated tests that detect a true lift multiplied into treated regions."""
    y, x = regions[f"test_{metric}"].to_numpy(), regions[f"pre_{metric}"].to_numpy()
    pairs = make_pairs(x)
    rows = []
    for lift in lifts:
        for design in DESIGNS:
            hits = 0
            for _ in range(n_sims):
                treat = assign(len(y), design, rng, pairs)
                res = analyse(np.where(treat, y * (1 + lift), y), x, treat, design, pairs, alpha)
                hits += (res["p_value"] < alpha) and (res["effect"] > 0)
            rows.append({"metric": metric, "lift": lift, "design": design, "power": hits / n_sims})
    return pd.DataFrame(rows)


def mde_at_80(power: pd.DataFrame) -> dict:
    """Smallest lift with 80% power, by linear interpolation of the power curve."""
    out = {}
    for (metric, design), g in power.groupby(["metric", "design"]):
        g = g.sort_values("lift")
        out[f"{metric}/{design}"] = (float(np.interp(0.8, g["power"], g["lift"]))
                                     if g["power"].max() >= 0.8 else None)
    return out


def simulate_price_test(items: pd.DataFrame, ac: dict, elasticity: float, margin: float,
                        rng) -> tuple[pd.DataFrame, dict]:
    """One run of the real experiment on top of the observed test-period orders.

    Treated regions: every price goes up by `price_change`, and each item is
    kept with probability (1 + price_change) ** elasticity, i.e. some buyers
    walk away. Cost per item = (1 - margin) * original price.
    """
    dp = ac["price_change"]
    capped = winsorize_prices(items, ac["winsorize_q"])
    regions = region_table(capped, ac["unit"], ac["test_start"], ac["pre_weeks"], ac["test_weeks"],
                           ac["min_pre_orders"])
    start = pd.Timestamp(ac["test_start"])
    test = capped[(capped["date"] >= start)
                  & (capped["date"] < start + pd.Timedelta(weeks=ac["test_weeks"]))
                  & capped[ac["unit"]].isin(regions.index)].copy()

    # plain random split + CUPED: matched pairs didn't reduce the noise any further (see ladder)
    treat = assign(len(regions), "cuped", rng)
    is_t = test[ac["unit"]].isin(set(regions.index[treat])).to_numpy()

    kept = ~is_t | (rng.random(len(test)) < (1 + dp) ** elasticity)
    test, is_t = test[kept].copy(), is_t[kept]
    test["profit"] = test["price"] * np.where(is_t, 1 + dp, 1) - (1 - margin) * test["price"]
    test.loc[is_t, "price"] *= 1 + dp

    agg = test.groupby(ac["unit"]).agg(units=("order_item_id", "size"), revenue=("price", "sum"),
                                       profit=("profit", "sum")).reindex(regions.index, fill_value=0)
    covariates = {"units": regions["pre_units"], "revenue": regions["pre_revenue"],
                  "profit": margin * regions["pre_revenue"]}
    results = {m: analyse(agg[m].to_numpy(), covariates[m].to_numpy(), treat, "cuped",
                          alpha=ac["alpha"]) for m in ["units", "revenue", "profit"]}
    table = pd.DataFrame(results).T
    table["expected_from_elasticity"] = {
        "units": (1 + dp) ** elasticity - 1,
        "revenue": (1 + dp) ** (1 + elasticity) - 1,
        "profit": (dp + margin) / margin * (1 + dp) ** elasticity - 1,
    }

    # what the test tells us about the elasticity itself
    u = table.loc["units"]
    implied = {k: float(np.log(1 + u[c]) / np.log(1 + dp))
               for k, c in [("estimate", "relative_effect"), ("ci_low", "ci_low"),
                            ("ci_high", "ci_high")]}
    implied["ci_low"], implied["ci_high"] = sorted([implied["ci_low"], implied["ci_high"]])

    # balance check: groups should look the same before the test
    x = regions["pre_units"].to_numpy()
    smd = (x[treat].mean() - x[~treat].mean()) / np.sqrt((x[treat].var() + x[~treat].var()) / 2)
    info = {"n_regions": int(len(regions)), "n_treated": int(treat.sum()),
            "pre_period_std_mean_diff": float(smd), "implied_elasticity": implied}
    return table, info


def plot_aa(aa: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, design in zip(axes, DESIGNS):
        p = aa.loc[aa["design"] == design, "p_value"]
        ax.hist(p, bins=20, range=(0, 1), color=plotting.BLUE, alpha=0.85)
        ax.axhline(len(p) / 20, color=plotting.ORANGE, ls="--", lw=1)
        ax.set_title(f"{LABELS[design]}\nfalse positive rate {np.mean(p < 0.05):.1%}", fontsize=9)
        ax.set_xlabel("p-value")
    axes[0].set_ylabel("number of A/A tests")
    fig.suptitle("A/A tests on real data (units, final design): p-values should be flat",
                 fontsize=11, weight="bold", y=1.04)
    plotting.save(fig, path)


def plot_ladder(ladder: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(ladder))[::-1]
    ax.barh(y, 100 * ladder["approx_mde"], color=plotting.BLUE, height=0.6)
    for yi, (_, r) in zip(y, ladder.iterrows()):
        ax.text(100 * r["approx_mde"] + 0.4, yi, f"{r['approx_mde']:.0%}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(ladder["step"], fontsize=8)
    ax.set_xlabel("smallest effect the test can reliably detect (%, 80% power)")
    ax.set_title("Making the geo test sensitive enough, one step at a time")
    plotting.save(fig, path)


def plot_power(power: pd.DataFrame, mde: dict, path):
    metrics = list(power["metric"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.7), sharey=True)
    colors = {"simple": plotting.GREY, "cuped": plotting.BLUE, "paired_cuped": plotting.GREEN}
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for design, g in power[power["metric"] == metric].groupby("design"):
            m = mde.get(f"{metric}/{design}")
            label = LABELS[design] + (f" (MDE {m:.0%})" if m else " (MDE > 20%)")
            ax.plot(100 * g["lift"], g["power"], marker="o", ms=4, color=colors[design], label=label)
        ax.axhline(0.8, color="black", lw=0.6, ls=":")
        ax.set_xlabel(f"true change in {metric} (%)")
        ax.set_title(f"metric: {metric}")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("power (alpha = 5%)")
    np.atleast_1d(axes)[0].set_ylim(0, 1.02)
    fig.suptitle("8-week geo test, 16-week pre-period", fontsize=11, weight="bold")
    plotting.save(fig, path)


def run(cfg: dict) -> dict:
    ac = cfg["ab_test"]
    out, figs = cfg["paths"]["outputs"], cfg["paths"]["figures"]
    rng = np.random.default_rng(cfg["seed"])

    items = load_items(cfg["data"]["raw_dir"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    capped = winsorize_prices(items, ac["winsorize_q"])
    regions = region_table(capped, ac["unit"], ac["test_start"], ac["pre_weeks"], ac["test_weeks"],
                           ac["min_pre_orders"])
    log.info("%d regions with >= %d orders in the pre-period", len(regions), ac["min_pre_orders"])

    ladder = noise_ladder(items, ac, rng)
    for _, r in ladder.iterrows():
        log.info("  %-38s %3d regions  A/A std %.3f  -> MDE ~%.0f%%", r["step"], r["n_regions"],
                 r["aa_std"], 100 * r["approx_mde"])

    aa = aa_tests(regions, "units", ac["n_aa_sims"], ac["alpha"], rng)
    fpr = aa.groupby("design")["p_value"].apply(lambda p: float((p < ac["alpha"]).mean())).to_dict()
    log.info("A/A false positive rates (units): %s", {k: round(v, 3) for k, v in fpr.items()})

    power = pd.concat([power_curves(regions, m, ac["lifts"], ac["n_power_sims"], ac["alpha"], rng)
                       for m in ["units", "revenue"]], ignore_index=True)
    mde = mde_at_80(power)
    log.info("MDE at 80%% power: %s", {k: (f"{v:.1%}" if v else "> 20%") for k, v in mde.items()})

    with open(out / "elasticity_summary.json") as f:
        mean_elasticity = json.load(f)["prior"]["mean"]
    sim, info = simulate_price_test(items, ac, mean_elasticity, cfg["optimization"]["gross_margin"],
                                    rng)
    log.info("simulated +%.0f%% price test (true elasticity %.2f):", 100 * ac["price_change"],
             mean_elasticity)
    for metric, r in sim.iterrows():
        log.info("  %-8s %+6.1f%% [%+.1f%%, %+.1f%%] p=%.3f (true effect %+.1f%%)", metric,
                 100 * r["relative_effect"], 100 * r["ci_low"], 100 * r["ci_high"], r["p_value"],
                 100 * r["expected_from_elasticity"])
    ie = info["implied_elasticity"]
    log.info("  -> elasticity implied by the units result: %.2f [%.2f, %.2f]",
             ie["estimate"], ie["ci_low"], ie["ci_high"])

    summary = {"n_regions": int(len(regions)), "aa_false_positive_rate_units": fpr,
               "noise_ladder": ladder.round(4).to_dict(orient="records"), "mde_80": mde,
               "simulated_test": {"true_elasticity": mean_elasticity, **info,
                                  "results": sim.round(5).to_dict(orient="index")}}
    ladder.round(4).to_csv(out / "ab_noise_ladder.csv", index=False)
    power.to_csv(out / "ab_power_curves.csv", index=False)
    sim.round(5).to_csv(out / "ab_simulated_test.csv")
    with open(out / "ab_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_aa(aa, figs / "ab_aa_tests.png")
    plot_ladder(ladder, figs / "ab_noise_ladder.png")
    plot_power(power, mde, figs / "ab_power.png")
    log_run("abtest", ac, {"fpr_units": fpr, "mde_80": {k: v or -1 for k, v in mde.items()},
                           "implied_elasticity": ie["estimate"]}, out)
    return summary
