import mlmc.mc
import numpy as np
from numba import njit

def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    mu = r
    n_steps = 2**10
    n_paths0 = 20000

    # mean, vars, costs, corr_sum = mlmc.mc._variance_estimator_asian(S0, mu, sigma, 10, 5000, K, r, T)
    # print('estimated vars:', vars)
    # print('estimated cost per path:', costs)
    # path_five = mlmc.mc._per_level_path_calc_asian(10, 0, vars, costs, 0.1)
    # print(path_five)
    price, se = mlmc.mc.mlmc_asian(S0, mu, sigma, 10, K, r, T, 0.1)
    print('price, se:', price, se)

    # mean0, se0 = mlmc.mc.asian_price_mc(S0, mu, sigma, n_steps, n_paths0, K, r, T)
    # n_paths = int(np.ceil(n_paths0 * (2 * se0 / 0.1) ** 2))
    # print(n_paths)
    # mean, se1 = mlmc.mc.asian_price_mc(S0, mu, sigma, n_steps, n_paths, K, r, T)
    # print(se1)

if __name__ == "__main__":
    main()
