from mlmc.mc import mlmc_barrier, warmup, barrier_price_mc, _barrier_mc_sum_sumsq
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt



def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    mu = r
    max_level = 10
    epsilons = [0.2, 0.1, 0.05, 0.025, 0.0125]
    n_paths0 = 20000
    barrier = 120
    warmup()
    _barrier_mc_sum_sumsq(S0, mu, sigma, 8, 4, K, barrier, r, T)
    mlmc_res = {"costs": [], "prices": [], "ses": []}
    mc_res = {"costs": [], "prices": [], "ses": []}

    for idx, eps in enumerate(epsilons):
        #MLMC benchmark
        max_level = int(np.ceil(np.log2(4*T/eps**2)))
        print(max_level)
        start =  time.perf_counter_ns()
        price, se = mlmc_barrier(S0, mu, sigma, max_level, K, barrier, r, T, eps)
        end = time.perf_counter_ns()
        mlmc_res["costs"].append(end-start)
        mlmc_res["prices"].append(price)
        mlmc_res["ses"].append(se)
        print(mlmc_res)

        #MC benchmark
        if idx != 6:
            n_steps = int(np.ceil(4*T / eps**2))
            start = time.perf_counter_ns()
            mean0, se0 = barrier_price_mc(S0, mu, sigma, n_steps, n_paths0, K, barrier, r, T)
            n_paths = int(np.ceil(n_paths0 * (2*se0/eps)**2))
            price1, se1 = barrier_price_mc(S0, mu, sigma, n_steps, n_paths, K, barrier, r, T)
            end = time.perf_counter_ns()
            mc_res["costs"].append(end - start)
            mc_res["prices"].append(price1)
            mc_res["ses"].append(se1)
            print(mc_res)

    mlmc_costs = np.array(mlmc_res["costs"]) * 1e-9
    mc_costs = np.array(mc_res["costs"]) * 1e-9
    eps_arr = np.array(epsilons)

    # Reference slopes
    c_mc = mc_costs[0] * eps_arr[0] ** 4
    c_mlmc = mlmc_costs[1] * eps_arr[1] ** 3

    start = 1  # skip the first 2 eps points
    x = np.log(eps_arr[start:])
    y = np.log(mlmc_costs[start:])

    slope, intercept = np.polyfit(x, y, 1)

    print("log-log slope (cost vs eps):", slope)

    plt.figure(figsize=(7, 5))

    plt.loglog(eps_arr, mlmc_costs, "o-", label="MLMC")
    plt.loglog(eps_arr, mc_costs, "o-", label="Standard MC")

    plt.loglog(
        eps_arr, c_mlmc * eps_arr ** (-3),
        "--", label=r"$O(\varepsilon^{-3})$"
    )

    plt.loglog(
        eps_arr, c_mc * eps_arr ** (-4),
        "--", label=r"$O(\varepsilon^{-4})$"
    )

    plt.xlabel(r"Target accuracy $\varepsilon$")
    plt.ylabel("Runtime (seconds)")
    plt.title("Cost vs Accuracy (Barrier Up-and-Out Call)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
