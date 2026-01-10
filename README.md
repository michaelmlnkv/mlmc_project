# Multi-level Monte Carlo vs Classic Monte Carlo for Asian and Barrier call option pricing
## Introduction
Efficiently and correctly pricing options is important. However, pricing methods utilizing classic Monte Carlo methods can be inefficient, especially given a small $\varepsilon$ due to the large number of paths (to minimize variance),
as well as a high required discretization density (bias). To solve this, we implement a [Multi-level Monte Carlo method](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf) (introduced by Michael B. Giles, University
of Oxford). The core idea behind this improvement over classic MC
is to use the identity $\mathbb{E}[V_L] = \mathbb{E}[V_0] + \sum_{l=1}^L \mathbb{E}[V_l] - \mathbb{E}[V_{l-1}]$, and computing $\Delta_l \approx \mathbb{E}[V_l] - \mathbb{E}[V_{l-1}]$ using **coupled** paths, with 
$2^l$ (fine path) and $2^{l-1}$ (coarse path) points per path.
This ensures that as $l \uparrow$, $Var(\Delta_l) \to 0$, making the required number of samples shrink significantly, only requiring a very small number of samples at the highest level, thus significantly reducing computational cost.
## Motivation
We can break the error of our MC's estimate $\hat{V}_h$ w.r.t the true option price $V$ from $|\hat{V}_h - V| \leq |\hat{V}_h - V_h| + |V_h - V| \leq \varepsilon$ (with $V_h$ being the true discretized price). Assuming we allocate our error budget equally, we
have $\varepsilon/2$ for the variance error ($|\hat{V}_h - V|$), and $\varepsilon/2$ for the bias error ($|V_h - V|$). To control the variance error, we want our SE to be less than $\varepsilon/\sqrt{N}$, implying that 
our number of samples $N = O(\varepsilon^{-2})$ (by CLT, the SE of classic MC is $\sigma / \sqrt{N}$). Similarly, the discretization bias for Asian and Barrier call options is $O(h)$, and since $h = T/n$, we can 
conclude that the number of points per sample $n = O(\varepsilon^{-1})$. Thus, the combined computational cost for classic MC is $O(\varepsilon^{-2} \times O(\varepsilon^{-1}) = O(\varepsilon^{-3})$, i.e., if you want to 
halve the error, you must perform 8x the computational cost.

Multi-level MC solves this issue by using the telescoping sum mentioned above $\mathbb{E}[V_L] = \mathbb{E}[V_0] + \sum_{l=1}^L \mathbb{E}[V_l] - \mathbb{E}[V_{l-1}]$. The total number of required paths still remains
$N = O(\varepsilon^{-2})$, but the average cost per path becomes $O(1)$, due to the significantly lowered number of samples required at higher levels. Thus, the total computational cost is $O(\varepsilon^{-2})$, a clear
improvement over classic MC.

## Multilevel Monte Carlo (MLMC)
Multilevel Monte Carlo (MLMC) is a variance–reduction technique that accelerates Monte Carlo estimation by exploiting a hierarchy of discretizations. Rather than estimating an expectation using only the finest time step, MLMC combines information from multiple levels of resolution in a way that minimizes total computational cost.
Let $P_l$ denote the payoff computed using a time step $h_l = T / 2^l$. MLMC is based on the telescoping identity
$\mathbb{E}[P_L]=\mathbb{E}[P_0]+\sum_{l=1}^L \mathbb{E}[P_l - P_{l-1}],$
which expresses the finest–level expectation as a sum of expectations of differences between successive levels. Each term in this sum is estimated independently using Monte Carlo sampling.

The effectiveness of MLMC relies on coupling the simulations at levels $l$ and $l-1$ so that the variance of the correction $P_l - P_{l-1}$ decays rapidly as the time step is refined. In practice, this is achieved by constructing the fine and coarse paths from shared underlying random increments. When the payoff is sufficiently regular, this coupling leads to a strong correlation between $P_l$ and $P_{l-1}$, resulting in a much smaller variance for their difference than for either payoff individually.

Because the variance of the level corrections decreases with $l$ while the cost per sample increases, MLMC allocates many samples to coarse, inexpensive levels and progressively fewer samples to finer, more expensive levels. An optimal allocation balances variance and cost across levels, yielding a Monte Carlo estimator whose total variance is controlled while minimizing computational effort.

Under suitable conditions on the payoff regularity and discretization scheme, MLMC achieves an overall computational complexity of $O(\varepsilon^{-2})$ for a target RMSE $\varepsilon$, representing a significant improvement over single–level Monte Carlo. This project applies MLMC to option pricing problems and demonstrates its efficiency gains empirically.

## Options
Let $K$ be strike price of the option, $r$ the risk-free rate, and $T$ be exercising time.
### Asian Option
Let $A(0,T)$ denote the average price of the underlying over time period $[0,T]$. The payoff $P(T) = max(0, A(0,T) - K)$. Thus, the fair price is the discounted payoff $V = e^{-rT}P(T)$.
### Barrier Option (Up-and-out)
Let $B$ be a barrier, i.e., if the price of the underlying rises over $B$ at any time in $[0,T], the payoff becomes 0 ("the option gets knocked out"). If the price at time $T$ is $S(T)$, then the payoff $P(T)$ is $max(0, S(T)-K)$ if the price never broke the barrier, and 0 otherwise. Similarly, the fair price $V$ is $P(T)e^{-rT}$.

## Single-level MC Experiments
For the following experiments we set parameters $S(0) = 100, r = \mu = 0.05, \sigma = 0.2, T = 1.0, K = 100, B = 120$.
### Asian Convergence 
First, we estimate a baseline with 4096 steps and 200,000 paths to minimize both the discretization bias as well as MC variance. Then, we compute error for step numbers in $[16, 32, 64, 128, 256, 512]$, each with 50,000 paths.

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/2762ad18-ed04-484b-9617-c8654adb395c" /> <img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/1f04dff0-b274-4ac2-a672-aa32c8b3d1b1" />
We can see that the bias doesn't decay cleanly as $h$ gets smaller (we expect it to decay as $O(h)$, i.e., a straight line with slope 1 in a log-log plot). This tells us that the MC variance dominates the error (y-axis on the left graph), so we have to find a way to reduce this variance. On the right, we plot the price vs number of steps, including the MC standard error interval. It is a good sign that most of the SE intervals contain the true price, but it is slightly concerning to see the SE interval with the highest number of steps not include the baseline.

### Barrier Convergence 
We follow the same procedure as above.

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/85fa1fe2-ef39-412c-bee7-f3daadaa856c" /> <img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/3c685120-d4b0-4b9d-9d2f-2c7ae82963fc" />
For the barrier option we can clearly see the bias decrease as $h$ gets smaller. This tells us that for barrier options, the discretization bias dominates the error, so increasing the number of steps (making $h$ smaller) yields a clear improvement. We can also see the price of the option decrease as the number of steps grows, because more knockouts get detected, so the price drops toward the "true", more finely monitored, value.



