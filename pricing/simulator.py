"""Selling a fixed stock over a fixed season: simulator + exact solution.

Setup (prices are relative to a base price of 1):
- the seller has `inventory` units and `horizon` days to sell them
- every day they pick a price multiplier from a small menu
- demand that day ~ Poisson(base * weekday_factor * multiplier ** elasticity)
- sales = min(demand, stock left); reward = multiplier * sales
- whatever is left at the end is cleared at `salvage` per unit

Because the demand model is known here, the optimal policy can be computed
exactly with dynamic programming. The RL agent never sees the model, it only
interacts with `step()`, so DP gives a ceiling to measure the agent against.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PricingEnv:
    base_demand: float
    weekday_factors: np.ndarray
    elasticity: float
    multipliers: np.ndarray
    horizon: int
    inventory: int
    salvage: float
    start_dow: int = 0  # 0 = the season starts on a Monday

    def __post_init__(self):
        self.multipliers = np.asarray(self.multipliers, dtype=float)
        self.weekday_factors = np.asarray(self.weekday_factors, dtype=float)
        dows = (self.start_dow + np.arange(self.horizon)) % 7
        # expected demand for every (day, action), shape (horizon, n_actions)
        self.lam = (self.base_demand * self.weekday_factors[dows][:, None]
                    * self.multipliers[None, :] ** self.elasticity)

    @property
    def n_actions(self) -> int:
        return len(self.multipliers)

    def step(self, t: int, stock: int, action: int, rng: np.random.Generator) -> tuple[int, float]:
        sales = min(rng.poisson(self.lam[t, action]), stock)
        return stock - sales, self.multipliers[action] * sales

    def terminal_value(self, stock):
        return self.salvage * stock


def solve_dp(env: PricingEnv) -> tuple[np.ndarray, np.ndarray]:
    """Backward induction. Returns V (horizon+1, inventory+1) and the policy (horizon, inventory+1)."""
    T, I, A = env.horizon, env.inventory, env.n_actions
    V = np.zeros((T + 1, I + 1))
    V[T] = env.terminal_value(np.arange(I + 1))
    policy = np.zeros((T, I + 1), dtype=int)
    k = np.arange(I + 1)

    for t in range(T - 1, -1, -1):
        q = np.empty((I + 1, A))
        for a in range(A):
            pmf = stats.poisson.pmf(k, env.lam[t, a])
            sf = stats.poisson.sf(k - 1, env.lam[t, a])        # P(D >= k)
            price = env.multipliers[a]
            for x in range(I + 1):
                # demand below stock: sell k, keep x - k. Demand >= stock: sell out.
                below = pmf[:x] @ (price * k[:x] + V[t + 1, x - k[:x]])
                q[x, a] = below + sf[x] * (price * x + V[t + 1, 0])
        V[t] = q.max(axis=1)
        policy[t] = q.argmax(axis=1)
    return V, policy


def evaluate_static(env: PricingEnv, action: int) -> float:
    """Exact expected revenue of always using the same price."""
    fixed = np.full((env.horizon, env.inventory + 1), action)
    return policy_value(env, fixed)


def policy_value(env: PricingEnv, policy: np.ndarray) -> float:
    """Exact expected revenue of a (day, stock) -> action table, from the start state."""
    T, I = env.horizon, env.inventory
    V = env.terminal_value(np.arange(I + 1)).astype(float)
    k = np.arange(I + 1)
    for t in range(T - 1, -1, -1):
        new = np.empty(I + 1)
        for x in range(I + 1):
            a = policy[t, x]
            lam, price = env.lam[t, a], env.multipliers[a]
            pmf = stats.poisson.pmf(k[:x], lam)
            new[x] = pmf @ (price * k[:x] + V[x - k[:x]]) + stats.poisson.sf(x - 1, lam) * (price * x + V[0])
        V = new
    return float(V[I])


def simulate(env: PricingEnv, policy: np.ndarray, uniforms: np.ndarray) -> dict:
    """Run a policy table on many episodes at once.

    `uniforms` (episodes x horizon) are turned into Poisson demand with the
    inverse CDF, so every policy sees the same "luck" (common random numbers)
    and the comparison between policies is much less noisy.
    """
    uniforms = np.clip(uniforms, 1e-12, 1 - 1e-12)   # ppf(0) would be -1
    n = uniforms.shape[0]
    stock = np.full(n, env.inventory)
    revenue = np.zeros(n)
    prices = np.zeros((n, env.horizon))
    stocks = np.zeros((n, env.horizon + 1), dtype=int)
    stocks[:, 0] = stock
    for t in range(env.horizon):
        a = policy[t, stock]
        demand = stats.poisson.ppf(uniforms[:, t], env.lam[t, a]).astype(int)
        sales = np.minimum(demand, stock)
        revenue += env.multipliers[a] * sales
        stock = stock - sales
        prices[:, t] = env.multipliers[a]
        stocks[:, t + 1] = stock
    revenue += env.terminal_value(stock)
    return {"revenue": revenue, "prices": prices, "stocks": stocks, "leftover": stock}
