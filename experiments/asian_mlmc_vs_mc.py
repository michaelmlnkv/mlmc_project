from mlmc.mc import mlmc_asian, warmup
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def _asian_mc_sum_sumsq(S0, mu, sigma, T, n_steps, n_paths, strike_price, r):
    """Return (sum, sumsq) of discounted Asian call payoffs without storing full paths."""
    dt = T / n_steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * T)

    total = 0.0
    total_sq = 0.0

    for p in range(n_paths):
        S = S0
        # include S0 in the average to match your payoff convention
        running_sum = S

        for _ in range(n_steps):
            z = np.random.normal()
            S = S * np.exp(drift + vol * z)
            running_sum += S

        avg_price = running_sum / (n_steps + 1)
        payoff = 0.0
        if avg_price > strike_price:
            payoff = avg_price - strike_price

        payoff *= disc
        total += payoff
        total_sq += payoff * payoff

    return total, total_sq

def asian_price_mc(S0, mu, sigma, T, n_steps, n_paths, strike_price, r):
    """Plain Monte Carlo price for an arithmetic Asian call.

    This implementation is memory-safe for very large `n_paths` because it does NOT
    allocate a (n_paths, n_steps+1) path array.

    Returns:
      price: discounted MC estimate
      se:    standard error of the estimator
    """
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    if n_paths <= 1:
        # Degenerate case; avoid division by zero in variance
        total, _ = _asian_mc_sum_sumsq(S0, mu, sigma, T, n_steps, max(n_paths, 1), strike_price, r)
        return total / max(n_paths, 1), float('inf')

    total, total_sq = _asian_mc_sum_sumsq(S0, mu, sigma, T, n_steps, n_paths, strike_price, r)

    mean = total / n_paths
    # Unbiased sample variance of discounted payoffs
    var = (total_sq - n_paths * mean * mean) / (n_paths - 1)
    se = np.sqrt(var / n_paths)

    return mean, se

def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    mu = r
    max_level = 10
    epsilons = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    n_paths0 = 20000
    warmup()
    _asian_mc_sum_sumsq(S0, mu, sigma, T, 8, 4, K, r)
    mlmc_res = {"costs": [], "prices": [], "ses": []}
    mc_res = {"costs": [], "prices": [], "ses": []}

    for idx, eps in enumerate(epsilons):
        #MLMC benchmark
        start =  time.perf_counter_ns()
        price, se = mlmc_asian(S0, mu, sigma, max_level, K, r, T, eps)
        end = time.perf_counter_ns()
        mlmc_res["costs"].append(end-start)
        mlmc_res["prices"].append(price)
        mlmc_res["ses"].append(se)
        print(mlmc_res)

        #MC benchmark
        n_steps = int(np.ceil(2 * T / eps))
        start = time.perf_counter_ns()
        mean0, se0 = asian_price_mc(S0, mu, sigma, T, n_steps, n_paths0, K, r)
        n_paths = int(np.ceil(n_paths0 * (2*se0/eps)**2))
        price1, se1 = asian_price_mc(S0, mu, sigma, T, n_steps, n_paths, K, r)
        end = time.perf_counter_ns()
        mc_res["costs"].append(end - start)
        mc_res["prices"].append(price1)
        mc_res["ses"].append(se1)
        print(mc_res)

    mlmc_costs = np.array(mlmc_res["costs"]) * 1e-9
    mc_costs = np.array(mc_res["costs"]) * 1e-9
    eps_arr = np.array(epsilons)

    # Reference slopes
    c_mc = mc_costs[0] * eps_arr[0] ** 3
    c_mlmc = mlmc_costs[0] * eps_arr[0] ** 2

    plt.figure(figsize=(7, 5))

    plt.loglog(eps_arr, mlmc_costs, "o-", label="MLMC")
    plt.loglog(eps_arr, mc_costs, "o-", label="Standard MC")

    plt.loglog(
        eps_arr, c_mlmc * eps_arr ** (-2),
        "--", label=r"$O(\varepsilon^{-2})$"
    )

    plt.loglog(
        eps_arr, c_mc * eps_arr ** (-3),
        "--", label=r"$O(\varepsilon^{-3})$"
    )

    plt.xlabel(r"Target accuracy $\varepsilon$")
    plt.ylabel("Runtime (seconds)")
    plt.title("Cost vs Accuracy (Asian Call)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
