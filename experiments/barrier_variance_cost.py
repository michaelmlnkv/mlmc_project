from mlmc.mc import _single_level_calc_barrier
import matplotlib.pyplot as plt
import numpy as np

def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    mu = r
    L = 10
    barrier = 120
    levels = np.array(list(range(L+1)))
    print("levels:", levels)


    vars = np.array([])
    hs = np.array([])
    costs = np.array([])

    for idx, level in enumerate(levels):
        mean, var, cost = _single_level_calc_barrier(S0, mu, sigma, level, 50000, K, barrier, r, T)
        vars = np.append(vars, var)
        hs = np.append(hs, T / (2**level))
        costs = np.append(costs, cost)
        print(f"level={level:4d}  h={hs[idx]:.6f} mean correction={mean:.6f}  cost≈{costs[idx]:.6f}, sample var={var:.6f}")

    plt.figure()
    plt.loglog(hs, vars, marker="o")
    # plt.gca().invert_xaxis()
    plt.xlabel("time step h = T / n_steps, smaller is better")
    plt.ylabel("Correction sample variance")
    plt.title("Barrier call: Correction sample variance decay")
    plt.grid(True, which="both")

    start = 2
    beta, logC = np.polyfit(np.log(hs[start:]), np.log(vars[start:]), 1)
    print("Estimated beta:", beta)
    anchor_idx = 4
    C2 = vars[anchor_idx] / (hs[anchor_idx])
    plt.loglog(hs, C2 * hs**beta, linestyle="--", label="O(h^0.458) reference")
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(levels, costs, marker="o")
    # plt.gca().invert_xaxis()
    plt.xlabel("level, larger is better")
    plt.ylabel("cost (s)")
    plt.title("Barrier call: Cost per level (s)")
    plt.grid(True, which="both")
    c = costs[anchor_idx] / (2 ** levels[anchor_idx])
    plt.semilogy(levels, c*2**levels, linestyle="--", label="O(2^l) reference")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()