from mlmc.sde import simulate_gbm_paths_recursive, simulate_gbm_coupled_paths
import numpy as np

def test_single_level():
    S0, mu, sigma, T = 100, 0.05, 0.2, 1.0
    paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, 256, 20000)
    ST = paths[:, -1]
    print("Single-level mean (should be ~105.13):", ST.mean())

def test_coupled():
    S0, mu, sigma, T = 100, 0.05, 0.2, 5.0
    level = 1
    S_f, S_c = simulate_gbm_coupled_paths(S0, mu, sigma, T, level, 10000)
    print("Fine mean :", S_f[:, -1].mean())
    print("Coarse mean:", S_c[:, -1].mean())
    print("Difference mean:", (S_f[:, -1] - S_c[:, -1]).mean())
    print(np.shape(S_c))
    print(np.shape(S_f))

if __name__ == "__main__":
    test_single_level()
    test_coupled()