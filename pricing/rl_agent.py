"""Tabular Q-learning for the stock-clearing problem in simulator.py.

State = (day of the season, units left), action = price multiplier.
The agent only sees rewards from the simulator, never the demand model.

The simulator is calibrated on real data: the weekday pattern and the
elasticity come from one category, the base demand from that category's
best-selling listing. Three policies are compared on the same simulated
demand:
- best static price (the best single price for the whole season)
- Q-learning
- the exact optimum from dynamic programming (the ceiling)
"""
import json
import logging

import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator, ScalarFormatter

from . import plotting
from .data import load_items
from .simulator import PricingEnv, evaluate_static, policy_value, simulate, solve_dp
from .tracking import log_run

log = logging.getLogger(__name__)
plt = plotting.plt


def train_q_learning(env: PricingEnv, episodes: int, alpha_power: float, eps_start: float,
                     eps_end: float, rng: np.random.Generator, eval_every: int = 2000,
                     eval_uniforms: np.ndarray | None = None):
    T, I, A = env.horizon, env.inventory, env.n_actions
    Q = np.zeros((T, I + 1, A))
    visits = np.zeros((T, I + 1, A))
    terminal = env.terminal_value(np.arange(I + 1))
    eps_schedule = eps_start * (eps_end / eps_start) ** np.linspace(0, 1, episodes)
    curve = []

    for ep in range(episodes):
        stock = I
        for t in range(T):
            if stock == 0:
                break  # sold out, every later reward is 0
            if rng.random() < eps_schedule[ep]:
                a = int(rng.integers(A))
            else:
                a = int(np.argmax(Q[t, stock]))
            new_stock, reward = env.step(t, stock, a, rng)
            if t == T - 1:
                future = terminal[new_stock]
            else:
                future = Q[t + 1, new_stock].max() if new_stock > 0 else 0.0
            # learning rate shrinks with the number of visits to this (state, action);
            # a fixed rate kept chasing the Poisson noise and never settled
            visits[t, stock, a] += 1
            alpha = 1 / visits[t, stock, a] ** alpha_power
            Q[t, stock, a] += alpha * (reward + future - Q[t, stock, a])
            stock = new_stock

        if eval_uniforms is not None and (ep + 1) % eval_every == 0:
            curve.append((ep + 1, simulate(env, Q.argmax(axis=2), eval_uniforms)["revenue"].mean()))
    return Q, visits, curve


def calibrate(items: pd.DataFrame, category: str, weeks: int = 26) -> tuple[float, np.ndarray]:
    """Base daily demand (best-selling listing) and weekday pattern (whole category)."""
    recent = items[items["date"] > items["date"].max() - pd.Timedelta(weeks=weeks)]
    sub = recent[recent["category"] == category]
    n_days = (recent["date"].max() - recent["date"].min()).days + 1
    base = sub.groupby(["product_id", "seller_id"]).size().max() / n_days

    daily = sub.groupby("date").size().reindex(
        pd.date_range(recent["date"].min(), recent["date"].max()), fill_value=0)
    weekday = daily.groupby(daily.index.dayofweek).mean()
    return float(base), (weekday / weekday.mean()).to_numpy()


def make_env(base: float, weekday: np.ndarray, elasticity: float, rc: dict) -> PricingEnv:
    expected = base * rc["horizon_days"]
    return PricingEnv(base_demand=base, weekday_factors=weekday, elasticity=elasticity,
                      multipliers=np.array(rc["price_multipliers"]), horizon=rc["horizon_days"],
                      inventory=max(1, int(round(rc["inventory_ratio"] * expected))),
                      salvage=rc["salvage_ratio"])


def compare(env: PricingEnv, rc: dict, rng: np.random.Generator) -> dict:
    V, dp_policy = solve_dp(env)
    static_values = [evaluate_static(env, a) for a in range(env.n_actions)]
    best_static = int(np.argmax(static_values))

    eval_u = rng.random((rc["eval_episodes"], env.horizon))
    Q, visits, curve = train_q_learning(env, rc["episodes"], rc["alpha_power"], rc["eps_start"],
                                        rc["eps_end"], rng, eval_uniforms=eval_u[:300])
    q_policy = Q.argmax(axis=2)
    static_policy = np.full_like(dp_policy, best_static)

    sims = {name: simulate(env, pol, eval_u) for name, pol in
            [("static", static_policy), ("q_learning", q_policy), ("dp_optimal", dp_policy)]}
    # exact expected values (no simulation noise) for the headline numbers
    exact = {"static": static_values[best_static], "q_learning": policy_value(env, q_policy),
             "dp_optimal": float(V[0, env.inventory])}
    diff = sims["q_learning"]["revenue"] - sims["static"]["revenue"]   # paired, same demand draws

    result = {
        "elasticity": env.elasticity,
        "inventory": env.inventory,
        "best_static_multiplier": float(env.multipliers[best_static]),
        "expected_revenue": exact,
        "q_learning_share_of_optimum": exact["q_learning"] / exact["dp_optimal"],
        "dp_gain_over_static": exact["dp_optimal"] / exact["static"] - 1,
        "q_learning_gain_over_static": exact["q_learning"] / exact["static"] - 1,
        # undefined when a static price is already optimal (nothing to capture)
        "share_of_possible_gain_captured": (
            (exact["q_learning"] - exact["static"]) / (exact["dp_optimal"] - exact["static"])
            if exact["dp_optimal"] - exact["static"] > 1e-6 else None),
        "simulated_q_minus_static": float(diff.mean()),
        "simulated_q_minus_static_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "sell_through": {k: float(1 - v["leftover"].mean() / env.inventory) for k, v in sims.items()},
    }
    return {"result": result, "curve": curve, "dp_policy": dp_policy, "q_policy": q_policy,
            "state_visits": visits.sum(axis=2), "sims": sims}


def scale_study(base: float, weekday: np.ndarray, elasticity: float, rc: dict) -> pd.DataFrame:
    """How much does dynamic pricing add over a static price as the business grows?"""
    rows = []
    for k in rc["scale_factors"]:
        env = make_env(base * k, weekday, elasticity, rc)
        V, _ = solve_dp(env)
        static = max(evaluate_static(env, a) for a in range(env.n_actions))
        rows.append({"scale": k, "inventory": env.inventory, "daily_demand": base * k,
                     "dp_gain_over_static": V[0, env.inventory] / static - 1})
    return pd.DataFrame(rows)


def plot_learning_curves(runs: dict, path):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.5))
    for ax, (label, run) in zip(np.atleast_1d(axes), runs.items()):
        res = run["result"]
        eps, vals = zip(*run["curve"])
        ax.plot(np.array(eps) / 1000, vals, color=plotting.BLUE, lw=1.3, label="Q-learning (greedy)")
        ax.axhline(res["expected_revenue"]["dp_optimal"], color=plotting.GREEN, ls="--",
                   label="optimum (DP)")
        ax.axhline(res["expected_revenue"]["static"], color=plotting.GREY, ls=":",
                   label="best static price")
        ax.set_title(f"elasticity {label}")
        ax.set_xlabel("training episodes (thousands)")
    np.atleast_1d(axes)[0].set_ylabel("revenue per season\n(300 fixed evaluation seasons)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plotting.save(fig, path)


def plot_policies(env: PricingEnv, run: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    vmin, vmax = env.multipliers.min(), env.multipliers.max()
    stocks = run["sims"]["dp_optimal"]["stocks"]
    cmap = plotting.matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f0f0f0")
    for ax, key, title in [(axes[0], "dp_policy", "optimal policy (DP)"),
                           (axes[1], "q_policy", "learned policy (Q-learning), grey = rarely visited")]:
        prices = env.multipliers[run[key]].astype(float)
        prices[:, 0] = np.nan  # no decision to make with zero stock
        if key == "q_policy":
            prices[run["state_visits"] < 30] = np.nan
        im = ax.imshow(prices.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[0.5, env.horizon + 0.5, -0.5, env.inventory + 0.5])
        ax.plot(np.arange(1, env.horizon + 1), np.percentile(stocks[:, :-1], 10, axis=0),
                color="black", lw=0.8, ls="--")
        ax.plot(np.arange(1, env.horizon + 1), np.percentile(stocks[:, :-1], 90, axis=0),
                color="black", lw=0.8, ls="--", label="stock levels actually reached (10-90%)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("day of season")
        ax.grid(False)
    axes[0].set_ylabel("units left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.26))
    fig.colorbar(im, ax=axes, label="price multiplier", shrink=0.9)
    fig.suptitle("Price by day and stock left: more stock than planned -> cheaper, less -> pricier",
                 fontsize=11, weight="bold", y=1.03)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scale(scale: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(scale["inventory"], 100 * scale["dp_gain_over_static"], marker="o", color=plotting.BLUE)
    ax.set_xscale("log")
    ax.set_xticks(scale["inventory"].tolist())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("units in stock for the season (log scale)")
    ax.set_ylabel("optimal dynamic vs best static (%)")
    ax.set_title("Dynamic pricing matters most for small stocks")
    plotting.save(fig, path)


def run(cfg: dict) -> dict:
    rc = cfg["rl"]
    out, figs = cfg["paths"]["outputs"], cfg["paths"]["figures"]
    rng = np.random.default_rng(cfg["seed"])

    items = load_items(cfg["data"]["raw_dir"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    base, weekday = calibrate(items, rc["category"])
    est = pd.read_csv(out / "elasticities.csv", index_col="category")
    cat_estimate = float(est.loc[rc["category"], "elasticity"])
    log.info("%s: base demand %.2f units/day, weekday factors %s, elasticity estimate %.2f",
             rc["category"], base, np.round(weekday, 2), cat_estimate)

    runs, rows = {}, []
    for scenario in rc["elasticity_scenarios"]:
        e = cat_estimate if scenario == "estimate" else float(scenario)
        label = f"{e:.2f}" + (" (estimate)" if scenario == "estimate" else "")
        runs[label] = compare(make_env(base, weekday, e, rc), rc, rng)
        r = runs[label]["result"]
        rows.append({"scenario": label, **{k: v for k, v in r.items() if not isinstance(v, dict)},
                     **{f"revenue_{k}": v for k, v in r["expected_revenue"].items()},
                     **{f"sell_through_{k}": v for k, v in r["sell_through"].items()}})
        log.info("elasticity %-16s static %.2f (x%.1f) | Q-learning %.2f | optimum %.2f | "
                 "Q = %.1f%% of optimum, %+.1f%% vs static (optimum %+.1f%%)",
                 label, r["expected_revenue"]["static"], r["best_static_multiplier"],
                 r["expected_revenue"]["q_learning"], r["expected_revenue"]["dp_optimal"],
                 100 * r["q_learning_share_of_optimum"], 100 * r["q_learning_gain_over_static"],
                 100 * r["dp_gain_over_static"])

    scale = scale_study(base, weekday, cat_estimate, rc)
    for _, s in scale.iterrows():
        log.info("  stock %4d units: optimal dynamic pricing %+.2f%% over static", s["inventory"],
                 100 * s["dp_gain_over_static"])

    pd.DataFrame(rows).round(4).to_csv(out / "rl_results.csv", index=False)
    scale.round(4).to_csv(out / "rl_scale_study.csv", index=False)
    with open(out / "rl_results.json", "w") as f:
        json.dump({"category": rc["category"], "base_daily_demand": base,
                   "weekday_factors": weekday.round(3).tolist(),
                   "scenarios": [r["result"] for r in runs.values()]}, f, indent=2)

    main_label = next(k for k in runs if "estimate" in k)
    main_env = make_env(base, weekday, cat_estimate, rc)
    np.savetxt(out / "rl_q_policy.csv", main_env.multipliers[runs[main_label]["q_policy"]],
               delimiter=",", fmt="%.2f",
               header="price multiplier by day (rows) and units left (columns 0..inventory)")
    plot_learning_curves(runs, figs / "rl_learning_curves.png")
    plot_policies(main_env, runs[main_label], figs / "rl_policies.png")
    plot_scale(scale, figs / "rl_scale_study.png")

    main = runs[main_label]["result"]
    log_run("rl", rc, {"q_share_of_optimum": main["q_learning_share_of_optimum"],
                       "q_gain_over_static": main["q_learning_gain_over_static"],
                       "dp_gain_over_static": main["dp_gain_over_static"]}, out)
    return main
