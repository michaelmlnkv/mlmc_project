import numpy as np
from numba import njit
from mlmc.sde import simulate_gbm_coupled_paths

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
def barrier_payoff_single_path(path, strike_price, barrier):
    for price in path:
        if price >= barrier:
            return 0.0

    return 0 if path[-1] - strike_price < 0 else path[-1] - strike_price

@njit
def barrier_payoff_per_path(paths, strike_price, barrier):
    n_paths, n_steps = np.shape(paths)
    payoffs = np.zeros(n_paths)
    for idx, path in enumerate(paths):
        payoffs[idx] = barrier_payoff_single_path(path, strike_price, barrier)
    return payoffs

'''
Undiscounted!!!
'''
@njit
def _barrier_payoff_coupled_paths(fine_paths, coarse_paths, strike_price, barrier):
    n_paths_fine, n_steps_fine = np.shape(fine_paths)
    n_paths_coarse, n_steps_coarse = np.shape(coarse_paths)
    assert(n_paths_fine == n_paths_coarse)
    payoffs_fine, payoffs_coarse = np.zeros(n_paths_fine), np.zeros(n_paths_coarse)
    for idx in range(n_paths_fine):
        payoffs_fine[idx] = barrier_payoff_single_path(fine_paths[idx], strike_price, barrier)
        payoffs_coarse[idx] = barrier_payoff_single_path(coarse_paths[idx], strike_price, barrier)
    return payoffs_fine, payoffs_coarse

@njit
def barrier_corrections(fine_paths, coarse_paths, strike_price, barrier):
    payoffs_fine, payoffs_coarse = _barrier_payoff_coupled_paths(fine_paths, coarse_paths, strike_price, barrier)
    return payoffs_fine - payoffs_coarse

