import mlmc.mc

def main():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100.0
    mu = r

    mean, vars, costs = mlmc.mc._variance_estimator_asian(S0, mu, sigma, 10, 5000, K, r, T)
    print('estimated vars:', vars)
    print('estimated cost per path:', costs)
    path_five = mlmc.mc._per_level_path_calc_asian(10, 9, vars, costs, 0.1)
    print(path_five)

if __name__ == "__main__":
    main()
