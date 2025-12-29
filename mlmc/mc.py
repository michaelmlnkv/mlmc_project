import numpy as np
from numba import njit
from mlmc.payoffs import barrier_payoff_per_path
from sde import simulate_gbm_paths_recursive
from mlmc.payoffs import asian_payoff_per_path

@njit
def asian_price_mc(paths, strike_price, r, T):
    payoffs = asian_payoff_per_path(paths, strike_price)
    return np.exp(-r*T) * np.mean(payoffs)

@njit
def barrier_price_mc(paths, strike_price, barrier, r, T):
    payoffs = barrier_payoff_per_path(paths, strike_price, barrier)
    return np.exp(-r*T) * np.mean(payoffs)
