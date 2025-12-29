import numpy as np
from numba import njit
from sde import simulate_gbm_paths_recursive

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
Barrier call (up-and-out) payoffs
'''

@njit
def barrier_payoff_single_path(path, strike_price, barrier):
    for price in path:
        if price >= barrier:
            return 0.0

    return 0 if path[-1] - strike_price < 0 else path[-1] - strike_price

def barrier_payoff_per_path(paths, strike_price, barrier):
    n_paths, n_steps = np.shape(paths)
    payoffs = np.zeros(n_paths)
    for idx, path in enumerate(paths):
        payoffs[idx] = barrier_payoff_single_path(path, strike_price, barrier)
    return payoffs


