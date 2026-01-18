import numpy as np
from numba import njit
from mlmc.sde import simulate_gbm_coupled_paths
from mlmc.sde import _brownian_bridge_calc

'''
Asian payoffs
'''
@njit
def asian_payoff_single_path(path, strike_price):
    avg_price = np.mean(path)
    return 0 if avg_price - strike_price < 0 else avg_price - strike_price

@njit
def asian_payoff_per_path(paths, strike_price):
    n_paths, n_steps = np.shape(paths)
    payoffs = np.zeros(n_paths)
    for idx, path in enumerate(paths):
        payoffs[idx] = asian_payoff_single_path(path, strike_price)
    return payoffs

'''
Undiscounted!!!
'''
@njit
def _asian_payoff_coupled_paths(fine_paths, coarse_paths, strike_price):
    n_paths_fine, n_steps_fine = np.shape(fine_paths)
    n_paths_coarse, n_steps_coarse = np.shape(coarse_paths)
    assert(n_paths_fine == n_paths_coarse)
    payoffs_fine, payoffs_coarse = np.zeros(n_paths_fine), np.zeros(n_paths_coarse)
    for idx in range(n_paths_fine):
        payoffs_fine[idx] = asian_payoff_single_path(fine_paths[idx], strike_price)
        payoffs_coarse[idx] = asian_payoff_single_path(coarse_paths[idx], strike_price)
    return payoffs_fine, payoffs_coarse

@njit
def asian_corrections(fine_paths, coarse_paths, strike_price):
    payoffs_fine, payoffs_coarse = _asian_payoff_coupled_paths(fine_paths, coarse_paths, strike_price)
    return payoffs_fine - payoffs_coarse


'''
Barrier call (up-and-out) payoffs
'''

@njit
def barrier_payoff_single_path(path, strike_price, barrier, h, sigma, bridge=False):
    for idx in range(1, len(path)):
        price = path[idx]
        if price >= barrier:
            return 0.0
        if bridge:
            prob = _brownian_bridge_calc(np.log(path[idx-1]), np.log(price), h, np.log(barrier) , sigma)
            U = np.random.random()
            if prob > U:
                return 0.0


    return 0 if path[-1] - strike_price < 0 else path[-1] - strike_price

@njit
def barrier_payoff_per_path(paths, strike_price, barrier, h, sigma, bridge=False):
    n_paths, n_steps = np.shape(paths)
    payoffs = np.zeros(n_paths)
    for idx, path in enumerate(paths):
        payoffs[idx] = barrier_payoff_single_path(path, strike_price, barrier, h, sigma, bridge=bridge)
    return payoffs

'''
Undiscounted!!!
'''
@njit
def _barrier_payoff_coupled_paths(fine_paths, coarse_paths, strike_price, barrier, h_fine, sigma, bridge=False):
    n_paths_fine, n_steps_fine = np.shape(fine_paths)
    n_paths_coarse, n_steps_coarse = np.shape(coarse_paths)
    assert n_paths_fine == n_paths_coarse

    payoffs_fine = np.zeros(n_paths_fine)
    payoffs_coarse = np.zeros(n_paths_coarse)

    b = np.log(barrier)
    h_coarse = 2.0 * h_fine

    for i in range(n_paths_fine):
        knocked_f = False
        knocked_c = False

        # quick initial check
        if fine_paths[i, 0] >= barrier:
            knocked_f = True
        if coarse_paths[i, 0] >= barrier:
            knocked_c = True

        # Walk over coarse intervals. coarse index j corresponds to fine indices 2j.
        for j in range(1, n_steps_coarse):
            if knocked_f and knocked_c:
                break

            # Endpoints
            Sc0 = coarse_paths[i, j - 1]
            Sc1 = coarse_paths[i, j]

            f_idx1 = 2 * j - 1
            f_idx2 = 2 * j
            Sf0 = fine_paths[i, f_idx1 - 1]
            Sf1 = fine_paths[i, f_idx1]
            Sf2 = fine_paths[i, f_idx2]

            # Discrete endpoint barrier checks
            if not knocked_c and Sc1 >= barrier:
                knocked_c = True
            if not knocked_f and (Sf1 >= barrier or Sf2 >= barrier):
                knocked_f = True

            if bridge:
                # Use two independent uniforms for the two fine sub-interval bridge tests.
                # Then derive a *uniform* Uc from their minimum for the coarse bridge test.
                U1 = np.random.random()
                U2 = np.random.random()
                Umin = U1
                if U2 < Umin:
                    Umin = U2
                # Probability integral transform: F_min(u)=1-(1-u)^2, so Uc=F_min(Umin) ~ Unif(0,1)
                one_minus = 1.0 - Umin
                Uc = 1.0 - one_minus * one_minus

                # Coarse bridge prob on [t, t+2h]
                if not knocked_c:
                    pc = _brownian_bridge_calc(np.log(Sc0), np.log(Sc1), h_coarse, b, sigma)
                    if Uc < pc:
                        knocked_c = True

                # Fine bridge probs on the two sub-intervals (independent uniforms)
                if not knocked_f:
                    pf1 = _brownian_bridge_calc(np.log(Sf0), np.log(Sf1), h_fine, b, sigma)
                    if U1 < pf1:
                        knocked_f = True
                    else:
                        pf2 = _brownian_bridge_calc(np.log(Sf1), np.log(Sf2), h_fine, b, sigma)
                        if U2 < pf2:
                            knocked_f = True

        # Terminal payoff if not knocked out
        if not knocked_f:
            STf = fine_paths[i, -1]
            if STf > strike_price:
                payoffs_fine[i] = STf - strike_price

        if not knocked_c:
            STc = coarse_paths[i, -1]
            if STc > strike_price:
                payoffs_coarse[i] = STc - strike_price

    return payoffs_fine, payoffs_coarse

@njit
def barrier_corrections(fine_paths, coarse_paths, strike_price, barrier, h_fine, sigma, bridge=False):
    payoffs_fine, payoffs_coarse = _barrier_payoff_coupled_paths(fine_paths, coarse_paths, strike_price, barrier, h_fine, sigma, bridge=bridge)
    return payoffs_fine - payoffs_coarse

