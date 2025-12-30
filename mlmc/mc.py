import numpy as np
from numba import njit
from mlmc.payoffs import barrier_payoff_per_path
from mlmc.sde import simulate_gbm_paths_recursive
from mlmc.payoffs import asian_payoff_per_path

@njit
def _mean_and_var(payoffs_p_path):
    mean = np.mean(payoffs_p_path)
    var = np.var(payoffs_p_path)
    return mean, var
@njit
def _inner_asian_price_mc(paths, strike_price, r, T):
    payoffs = asian_payoff_per_path(paths, strike_price)
    num_paths = np.shape(payoffs)[0]
    mean, var = _mean_and_var(payoffs)
    mean *= np.exp(-r*T)
    var *= np.exp(-r * T)
    se = np.sqrt(var / num_paths)
    return mean, se

@njit
def _inner_barrier_price_mc(paths, strike_price, barrier, r, T):
    payoffs = barrier_payoff_per_path(paths, strike_price, barrier)
    num_paths = np.shape(payoffs)[0]
    mean, var = _mean_and_var(payoffs)
    mean *= np.exp(-r * T)
    var *= np.exp(-r * T)
    se = np.sqrt(var / num_paths)
    return mean, se

def asian_price_mc(S0, mu, sigma, n_steps, n_paths, strike_price, r, T):
    paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths)
    mean, se = _inner_asian_price_mc(paths, strike_price, r, T)
    return mean, se

def barrier_price_mc(S0, mu, sigma, n_steps, n_paths, strike_price, barrier, r, T):
    paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths)
    mean, se = _inner_barrier_price_mc(paths, strike_price, barrier, r, T)
    return mean, se

