from mlmc.mc import mlmc_asian, mlmc_barrier
import numpy as np

def _pick_L_from_eps(T, eps, alpha = 1.0):
    """Heuristic starting level from weak order alpha: h_L ~ eps."""
    # Use bias target eps/2 => h_L ~ (eps/2)^(1/alpha)
    h_target = (eps / 2.0) ** (1.0 / alpha)
    return int(np.ceil(np.log2(T / h_target)))


def adaptive_max_level_bias_test(S0, mu, sigma, strike_price, barrier, r, T, eps, option_type, L_min=2, L_max=14, alpha=1.0, safety_se_factor=4.0):
    """Choose max MLMC level L so estimated bias is <= eps/2.

    We estimate bias using successive-level differences:
        bias(L) ~ |E[P_{L+1}] - E[P_L]|.

    To make this difference meaningful, we run MLMC at levels L and L+1
    with a tighter sampling error so the Monte Carlo noise doesn't dominate.

    Args:
      eps: total accuracy target (RMSE-style)
      safety_se_factor: we set MLMC sampling target to eps/safety_se_factor
                        for the bias test runs.

    Returns:
      Chosen L.
    """
    # Start from theory-based guess, but clamp to [L_min, L_max-1]
    L = _pick_L_from_eps(T, eps, alpha=alpha)
    L = max(L_min, min(L, L_max - 1))

    # Use a tighter SE for the bias test so |price(L+1)-price(L)| is informative
    eps_bias_se = eps / safety_se_factor

    while True:
        if option_type == "Asian":
            pL, seL = mlmc_asian(S0, mu, sigma, L, strike_price, r, T, eps_bias_se)
            pLp1, seLp1 = mlmc_asian(S0, mu, sigma, L + 1, strike_price, r, T, eps_bias_se)
        elif option_type == "Barrier":
            pL, seL = mlmc_barrier(S0, mu, sigma, L, strike_price, barrier, r, T, eps_bias_se)
            pLp1, seLp1 = mlmc_barrier(S0, mu, sigma, L + 1, strike_price, barrier, r, T, eps_bias_se)
        diff = abs(pLp1 - pL)

        # Accept if successive-level difference is within bias budget
        if diff <= eps / 2.0:
            return L

        L += 1
        if L >= L_max:
            # Give up at L_max; return the max allowed
            return L_max