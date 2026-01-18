from mlmc.mc import mlmc_asian, warmup, asian_price_mc, _asian_mc_sum_sumsq
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from mlmc.max_level import adaptive_max_level_bias_test



def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    mu = r
    max_level_cap = 14  # maximum level allowed for adaptive bias test
    epsilons = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    n_paths0 = 20000
    warmup()
    _asian_mc_sum_sumsq(S0, mu, sigma, T, 8, 4, K, r)
    mlmc_res = {"costs": [], "prices": [], "ses": [], "Ls": []}
    mc_res = {"costs": [], "prices": [], "ses": []}

    for idx, eps in enumerate(epsilons):
        # MLMC benchmark (choose L adaptively to control bias)
        # L = adaptive_max_level_bias_test(
        #     S0=S0,
        #     mu=mu,
        #     sigma=sigma,
        #     strike_price=K,
        #     r=r,
        #     T=T,
        #     eps=eps,
        #     L_min=2,
        #     L_max=max_level_cap,
        #     alpha=1.0,
        #     safety_se_factor=4.0,
        # )

        # To save computation let L = 10
        L = 10

        start = time.perf_counter_ns()
        price, se = mlmc_asian(S0, mu, sigma, L, K, r, T, eps)
        end = time.perf_counter_ns()
        mlmc_res["costs"].append(end-start)
        mlmc_res["prices"].append(price)
        mlmc_res["ses"].append(se)
        mlmc_res["Ls"].append(L)
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
