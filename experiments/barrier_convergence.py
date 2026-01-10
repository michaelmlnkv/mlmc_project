import numpy as np
import matplotlib.pyplot as plt
from mlmc.mc import barrier_price_mc
np.random.seed(42)
def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    barrier = 120
    mu = r

    n_paths_ref = 300000
    n_steps_ref = 4096
    print("Starting reference calculation")
    price_ref, se_ref = barrier_price_mc(S0, mu, sigma, n_steps_ref, n_paths_ref, K, barrier, r, T)
    print("Barrier reference:", price_ref, "SE:", se_ref)

    test_steps = [16, 32, 64, 128, 256, 512]
    bias = np.array([])
    hs = np.array([])
    ses = np.array([])
    prices = np.array([])
    n_paths_test = 50000
    for idx, step in enumerate(test_steps):
        np.random.seed(42)
        price, se = barrier_price_mc(S0, mu, sigma, step, n_paths_test, K, barrier, r, T)
        bias = np.append(bias, abs(price - price_ref))
        hs = np.append(hs, T / float(step))
        ses = np.append(ses, se)
        prices = np.append(prices, price)
        print(f"steps={step:4d}  h={hs[idx]:.6f}  price={price:.6f}  |bias|≈{bias[idx]:.6f}  SE={se:.6f}")

    plt.figure()
    plt.loglog(hs, bias, marker="o")
    #plt.gca().invert_xaxis()
    plt.xlabel("time step h = T / n_steps, smaller is better")
    plt.ylabel("|price - reference|")
    plt.title("Barrier call: discretization bias vs time step")
    plt.grid(True, which="both")
    plt.show()

    plt.figure()
    plt.errorbar(test_steps, prices, yerr=ses, fmt="o")
    plt.axhline(price_ref, linestyle="--")
    plt.xlabel("n_steps")
    plt.ylabel("price")
    plt.title("Barrier call: price vs n_steps (with MC SE)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

