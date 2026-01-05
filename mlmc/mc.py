import numpy as np
from numba import njit
from mlmc.payoffs import barrier_payoff_per_path
from mlmc.sde import simulate_gbm_paths_recursive, simulate_gbm_coupled_paths
from mlmc.payoffs import asian_payoff_per_path, barrier_corrections, asian_corrections
import time

def warmup():
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    r, K, B = 0.05, 100.0, 120.0

    n_paths = 2
    level = 1
    n_steps = 2

    # path sims
    simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths)
    simulate_gbm_coupled_paths(S0, mu, sigma, T, level, n_paths)

    # payoffs
    paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths)
    asian_payoff_per_path(paths, K)
    barrier_payoff_per_path(paths, K, B)

    fine, coarse = simulate_gbm_coupled_paths(S0, mu, sigma, T, level, n_paths)
    asian_corrections(fine, coarse, K)
    barrier_corrections(fine, coarse, K, B)

    # estimators
    _variance_estimator_asian(S0, mu, sigma, 10, n_paths, K, r, T)
    #mlmc_asian(S0, mu, sigma, 5, K, r, T, 0.2)

    # stats
    x = np.array([1.0, 2.0])
    _mean_and_var(x)

@njit
def _mean_and_var(arr):
    n = np.shape(arr)[0]
    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n

    # unbiased variance (ddof = 1)
    var = 0.0
    for i in range(n):
        diff = arr[i] - mean
        var += diff * diff
    var /= (n - 1)

    return mean, var

'''
Asian + Barrier MC pricing
'''
@njit
def _inner_asian_price_mc(paths, strike_price, r, T):
    payoffs = asian_payoff_per_path(paths, strike_price)
    num_paths = np.shape(payoffs)[0]
    mean, var = _mean_and_var(payoffs)
    mean *= np.exp(-r*T)
    var *= np.exp(-2 * r * T)
    se = np.sqrt(var / num_paths)
    return mean, se, var

@njit
def _inner_barrier_price_mc(paths, strike_price, barrier, r, T):
    payoffs = barrier_payoff_per_path(paths, strike_price, barrier)
    num_paths = np.shape(payoffs)[0]
    mean, var = _mean_and_var(payoffs)
    mean *= np.exp(-r * T)
    var *= np.exp(-2 * r * T)
    se = np.sqrt(var / num_paths)
    return mean, se, var

@njit
def asian_price_mc(S0, mu, sigma, n_steps, n_paths, strike_price, r, T):
    paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths)
    mean, se, _ = _inner_asian_price_mc(paths, strike_price, r, T)
    return mean, se

@njit
def barrier_price_mc(S0, mu, sigma, n_steps, n_paths, strike_price, barrier, r, T):
    paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths)
    mean, se, _ = _inner_barrier_price_mc(paths, strike_price, barrier, r, T)
    return mean, se

'''
Single level correction, sample variance, and cost calculations
'''
#Asian
def _single_level_calc_asian(S0, mu, sigma, level, n_paths, strike_price, r, T, return_correction_sum=False):
    start = time.perf_counter_ns()
    if level != 0:
        fine_paths, coarse_paths = simulate_gbm_coupled_paths(S0, mu, sigma, T, level, n_paths)
        corrections = asian_corrections(fine_paths, coarse_paths, strike_price)
        mean, var = _mean_and_var(corrections)
        disc_corr = corrections * np.exp(-r * T)
        correction_sum = np.sum(disc_corr)
        correction_sumsq = np.sum(disc_corr * disc_corr)
        mean *= np.exp(-r * T)
        var *= np.exp(-2 * r * T)
        se = np.sqrt(var / n_paths)

    #level 0, just use regular MC estimate of P_0
    else:
        paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, 1, n_paths)
        mean, se, var = _inner_asian_price_mc(paths, strike_price, r, T) #already discounted
        payoffs0 = asian_payoff_per_path(paths, strike_price)
        disc_payoffs0 = payoffs0 * np.exp(-r * T)
        correction_sum = np.sum(disc_payoffs0)
        correction_sumsq = np.sum(disc_payoffs0 * disc_payoffs0)

    end  = time.perf_counter_ns()
    cost = end - start
    if return_correction_sum:
        return mean, var, cost/n_paths, correction_sum, correction_sumsq
    return mean, var, cost / n_paths

#Barrier
def _single_level_calc_barrier(S0, mu, sigma, level, n_paths, strike_price, barrier, r, T):
    start = time.perf_counter()
    if level != 0:
        fine_paths, coarse_paths = simulate_gbm_coupled_paths(S0, mu, sigma, T, level, n_paths)
        corrections = barrier_corrections(fine_paths, coarse_paths, strike_price, barrier)
        mean, var = _mean_and_var(corrections)
        mean *= np.exp(-r * T)
        var *= np.exp(-2 * r * T)
        se = np.sqrt(var / n_paths)

    #level 0, just use regular MC estimate of P_0
    else:
        paths = simulate_gbm_paths_recursive(S0, mu, sigma, T, 1, n_paths)
        mean, se, var = _inner_barrier_price_mc(paths, strike_price, barrier, r, T) #already discounted

    end  = time.perf_counter()
    cost = end - start
    return mean, var, cost / n_paths

'''
MLMC Estimator (Asian)
'''
def _asian_mlmc_estimate(S0, mu, sigma, max_levels, paths_p_level, strike_price, r, T):
    price = 0.0
    se = 0.0
    level_contributions = []
    corr_sum_p_level = []
    corr_sumsq_p_level = []
    assert(max_levels+1 == len(paths_p_level))
    for level in range(max_levels+1):
        n_paths = paths_p_level[level]
        if n_paths != 0:
            curr_correction, curr_var, curr_cost, corr_sum, corr_sumsq = _single_level_calc_asian(S0, mu, sigma, level, n_paths, strike_price, r, T, return_correction_sum=True)
            price += curr_correction
            level_contributions.append(curr_correction)
            corr_sum_p_level.append(corr_sum)
            corr_sumsq_p_level.append(corr_sumsq)
        else:
            level_contributions.append(0)
            corr_sum_p_level.append(0)
            corr_sumsq_p_level.append(0)
    return price, se, np.array(level_contributions), np.array(corr_sum_p_level), np.array(corr_sumsq_p_level)

def _variance_estimator_asian(S0, mu, sigma, max_level, n_paths, strike_price, r, T):
    est_var = []
    est_cost = []
    means = []
    corrections_p_lvl = []
    corrections_sumsq_p_lvl = []
    for level in range(max_level+1):
        mean, var, cost_p_path, correction_sum, correction_sumsq = _single_level_calc_asian(S0, mu, sigma, level, n_paths, strike_price, r, T, return_correction_sum=True)
        est_var.append(var)
        est_cost.append(cost_p_path)
        means.append(mean)
        corrections_p_lvl.append(correction_sum)
        corrections_sumsq_p_lvl.append(correction_sumsq)
    return np.array(means), np.array(est_var), np.array(est_cost), np.array(corrections_p_lvl), np.array(corrections_sumsq_p_lvl)

def _per_level_path_calc_asian(max_level, level, vars, cost, epsilon):
    ir = 0.0
    for lvl in range(max_level+1):
        ir += np.sqrt(vars[lvl]*cost[lvl])
    paths = (2/epsilon)**2 * ir * np.sqrt(vars[level]/cost[level])
    return int(np.ceil(paths))

def _path_number_calculation_asian(S0, mu, sigma, max_levels, strike_price, r, T, epsilon):
    add_paths_p_lvl = []
    n_pilot = 500
    est_mean, est_vars, est_costs, pilot_corrections_sum, pilot_corrections_sumsq = _variance_estimator_asian(S0, mu, sigma, max_levels, n_pilot, strike_price, r, T)
    #print("est cost 2:", est_costs)
    for level in range(max_levels+1):
        paths_total = _per_level_path_calc_asian(max_levels, level, est_vars, est_costs, epsilon)
        paths_add = max(paths_total-n_pilot, 0)
        add_paths_p_lvl.append(paths_add)
    return np.array(add_paths_p_lvl), pilot_corrections_sum, pilot_corrections_sumsq, est_vars

def mlmc_asian(S0, mu, sigma, max_level, strike_price, r, T, epsilon):
    warmup()
    n_pilot = 500
    paths_p_level, pilot_corrections_sum, pilot_corrections_sumsq, est_vars = _path_number_calculation_asian(S0, mu, sigma, max_level, strike_price, r, T, epsilon)
    #print("add paths p level:", paths_p_level)
    _, _, _, add_corr_p_level, add_corr_sq_p_level = _asian_mlmc_estimate(S0, mu, sigma, max_level, paths_p_level, strike_price, r, T)
    price = 0.0
    se = 0.0
    for level in range(max_level+1):
        n_total = n_pilot + paths_p_level[level]
        total_sum = pilot_corrections_sum[level] + add_corr_p_level[level]
        total_corr = total_sum / n_total
        total_sumsq = pilot_corrections_sumsq[level] + add_corr_sq_p_level[level]
        price += total_corr

        if n_total > 1:
            v_l = (total_sumsq - n_total * total_corr * total_corr) / (n_total - 1)
        else:
            v_l = 0.0

        se += v_l / n_total
    se = np.sqrt(se)
    return price, se





