import numpy as np
from numba import njit


@njit

def simulate_gbm_paths_closed_form(S0, mu, sigma, n_steps, n_paths):
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            paths[i, t] = S0 * np.exp((mu - 0.5 * sigma**2)*t + sigma * np.random.randn(t**0.5))

    return paths

@njit
def simulate_gbm_paths_recursive(S0, mu, sigma, T, n_steps, n_paths):
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    dt = T / n_steps
    sqrt_dt = dt**0.5

    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            Z = np.random.normal()
            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*sqrt_dt*Z)

    return paths

# Maybe make a version with variable level difference for fine (n) and coarse (m), n >= m
@njit
def simulate_gbm_coupled_paths(S0, mu, sigma, T, level, n_paths):
    n_fine = 2**level
    n_coarse = 2**(level-1)

    dt_fine = T / n_fine
    sqrt_dt_fine = dt_fine**0.5
    dt_coarse = T / n_coarse
    sqrt_dt_coarse = dt_coarse**0.5

    fine_paths = np.zeros((n_paths, n_fine + 1))
    coarse_paths = np.zeros((n_paths, n_coarse + 1))

    coarse_paths[:, 0] = S0
    fine_paths[:, 0] = S0
    #fine_paths[:, 1] = S0

    for path in range(n_paths):

        for coarse_step in range(1, n_coarse+1):
            fine_step_1 = coarse_step * 2 - 1
            fine_step_2 = fine_step_1 + 1

            Z1 = np.random.normal()
            Z2 = np.random.normal()
            ZC = (Z1 + Z2)/(2**0.5)

            coarse_paths[path, coarse_step] = (coarse_paths[path, coarse_step-1] *
                                               np.exp((mu - 0.5*sigma**2)*dt_coarse + sigma*sqrt_dt_coarse*ZC))

            fine_paths[path, fine_step_1] = (fine_paths[path, fine_step_1-1] *
                                               np.exp((mu - 0.5*sigma**2)*dt_fine + sigma*sqrt_dt_fine*Z1))

            fine_paths[path, fine_step_2] = (fine_paths[path, fine_step_2 - 1] *
                                             np.exp((mu - 0.5 * sigma ** 2) * dt_fine + sigma * sqrt_dt_fine * Z2))

    return fine_paths, coarse_paths



