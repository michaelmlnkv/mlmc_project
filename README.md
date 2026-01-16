# Multi-level Monte Carlo vs Classic Monte Carlo for Asian and Barrier Call Option Pricing
## Introduction
Efficiently and correctly pricing options is important. However, pricing methods utilizing classic Monte Carlo methods can be inefficient, especially given a small $\varepsilon$ due to the large number of paths (to minimize variance),
as well as a high required discretization density (bias). To solve this, we implement a [Multi-level Monte Carlo method](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf) (introduced by Michael B. Giles, University
of Oxford). The core idea behind this improvement over classic MC
is to use the identity $\mathbb{E}[V_L] = \mathbb{E}[V_0] + \sum_{l=1}^L \mathbb{E}[V_l] - \mathbb{E}[V_{l-1}]$, and computing $\Delta_l \approx \mathbb{E}[V_l] - \mathbb{E}[V_{l-1}]$ using **coupled** paths, with 
$2^l$ (fine path) and $2^{l-1}$ (coarse path) points per path. If we denote $\Delta_l = V_l - V_{l-1}$ a **correction**, then
this ensures that as $l \uparrow$, $Var(\Delta_l) \to 0$, making the required number of samples shrink significantly, only requiring a very small number of samples at the highest level, thus significantly reducing computational cost.
## Motivation
We can break the error of our MC's estimate $\hat{V}_h$ w.r.t the true option price $V$ from $|\hat{V}_h - V| \leq |\hat{V}_h - V_h| + |V_h - V| \leq \varepsilon$ (with $V_h$ being the true discretized price). Assuming we allocate our error budget equally, we
have $\varepsilon/2$ for the variance error ($|\hat{V}_h - V|$), and $\varepsilon/2$ for the bias error ($|V_h - V|$). To control the variance error, we want our SE to be less than $\varepsilon/2$, implying that 
our number of samples $N = O(\varepsilon^{-2})$ (by CLT, the SE of classic MC is $\sigma / \sqrt{N}$, so $\sigma / \sqrt{N} \leq \varepsilon / 2 \implies N \geq 4\sigma^2 / \varepsilon^2 = O(\varepsilon^{-2})$ ). Similarly, the discretization bias for Asian and Barrier call options is $O(h)$ and $O(\sqrt{h})$, respectively, and since $h = T/n$, we can 
conclude the necessary number of points per sample is $n = O(\varepsilon^{-1})$ (Asian) or $O(\varepsilon^{-2})$ (Barrier). Thus, the combined computational cost for classic MC is $O(\varepsilon^{-2}) \times O(\varepsilon^{-1}) = O(\varepsilon^{-3})$, i.e., if you want to 
halve the error, you must perform 8x the computational cost (for Asian options). For barrier options, this cost is $O(\varepsilon^{-4})$, increasing the error by a factor of 16 to halve the error.

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

The standard error for MLMC is defined as such: $SE_{MLMC} = \sqrt{\sum_{l=0}^L \frac{Var(\Delta_l)}{N_l}}$, where $N_l$ is the number of paths allocated to a specific level $l$ (the allocation itself is discussed later).
The MLMC estimate is $\hat V_L = \sum_{l=0}^L Y_l$, with $Y_l = \frac{1}{N_l} \sum_{i = 0}^{N_l} \Delta_i$.

## Options
Let $K$ be strike price of the option, $r$ the risk-free rate, and $T$ be exercising time.
### Asian Option
Let $A(0,T)$ denote the average price of the underlying over time period $[0,T]$. The payoff $P(T) = max(0, A(0,T) - K)$. Thus, the fair price is the discounted payoff $V = e^{-rT}P(T)$.
### Barrier Option (Up-and-out)
Let $B$ be a barrier, i.e., if the price of the underlying rises over $B$ at any time in $[0,T]$, the payoff becomes 0 ("the option gets knocked out"). If the price at time $T$ is $S(T)$, then the payoff $P(T)$ is $max(0, S(T)-K)$ if the price never broke the barrier, and 0 otherwise. Similarly, the fair price $V$ is $P(T)e^{-rT}$.

## Single-level MC Experiments
For the following experiments we set parameters $S(0) = 100, r = \mu = 0.05, \sigma = 0.2, T = 1.0, K = 100, B = 120$.
### Asian Convergence 
First, we estimate a baseline with 4096 steps and 200,000 paths to minimize both the discretization bias as well as MC variance. Then, we compute error for step numbers in $[16, 32, 64, 128, 256, 512]$, each with 50,000 paths.

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/2762ad18-ed04-484b-9617-c8654adb395c" /> <img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/1f04dff0-b274-4ac2-a672-aa32c8b3d1b1" />

We can see that the bias doesn't decay cleanly as $h$ gets smaller (we expect it to decay as $O(h)$, i.e., a straight line with slope 1 in a log-log plot). This tells us that the MC variance dominates the error (y-axis on the left graph), so we have to find a way to reduce this variance. On the right, we plot the price vs number of steps, including the MC standard error interval. It is a good sign that most of the SE intervals contain the true price, but it is slightly concerning to see the SE interval with the highest number of steps not include the baseline.

### Barrier Convergence 
We follow the same procedure as above.

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/85fa1fe2-ef39-412c-bee7-f3daadaa856c" /> <img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/3c685120-d4b0-4b9d-9d2f-2c7ae82963fc" />

For the barrier option we can clearly see the bias decrease as $h$ gets smaller. This tells us that for barrier options, the discretization bias dominates the error, so increasing the number of steps (making $h$ smaller) yields a clear improvement. We can also see the price of the option decrease as the number of steps grows, which we can attribute to more knockouts being detected, making the price drop toward the "true", more finely monitored, value.

## Variance Decay
From the above experiments we can see that we need a way of decreasing variance without paying the $1/ \sqrt{N}$ computational cost. As a first step towards this goal, we first observe the variance decay of the corrections when using coupled fine/coarse paths.
We use the same parameters as above, and calculate cost per sample $C_l$ by timing the calculations for a single level and dividing by the number of samples (here, number of paths). We expect this to follow $C_l = O(2^l)$ since we use $2^l$ points per (fine) path. The coarse path using $2^{l-1}$ points is also $O(2^l)$.
### Asian Options
Since our payoff function (taking an average over a path) is sufficiently smooth, we can conclude that our fair price is also sufficiently smooth (product of 2 "sufficiently smooth" functions, e.g. Lipschitz continuous). We know that exact GBM updates have strong order $\alpha \approx 1$, so $\mathbb{E}[|V_l - V|^2]^{1/2} = O(h_l^\alpha) = O(h_l)$, implying that due to the fine and coarse paths being coupled, the difference $|\Delta_l| = |V_l - V_{l-1}| = |V_l - V + V - V_{l-1}| \leq |V_l - V| + |V_{l-1} - V| = O(h_l) + O(h_{l-1}) = O(h_l)$. Then, it clearly follows that $\Delta_l^2 = O(h^2_l)$. It is important to know that $\mathbb{E}[\Delta_l] << \Delta_l$, so we can make the approximation $Var(\Delta_l) = \mathbb{E}[(\Delta_l - \mathbb{E}[\Delta_L])^2] \approx \mathbb{E}[\Delta_l^2] = O(h^2_l)$. Thus, we expect to see a straight line with slope 2 on a loglog plot of $Var(\Delta_l)$ vs $h_l$. 
We also plot the cost per sample $C_l$ vs level $l$, and expect to see a straight line of slope 2 on a semilog ($C_l$ with log scale) plot.

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/5ee860c3-c0f9-4301-80fa-484d0eb392be" /> <img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/ca5bef90-5e22-4368-91a6-560bb42cd1ce" />

We can see that the cost the per sample behaves exactly as expected. The variance decays slightly worse than we would like, with a fitted slope of around 1.9. However, this is still close enough to verify that the variance decays approximately as we expect.
### Barrier Options
In this case, our payoff function is not nearly as smooth as taking an average over a path. Intuitively, the discontinuities are introduced due to the fact that more points corresponds to more opportunities for $S$ to cross the barrier $B$, and the option to get knocked out. Thus, even with coupled paths, the coarse path might not break the barrier, but the finer path might. We observe the significantly slower variance decay in the following plot, as well as the expected $O(2^l)$ cost per sample.

<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/f1c64adf-160a-4663-8dd3-bae59a3f6370" /> <img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/1c4a971d-96b0-4338-b230-f755ca6c448e" />

The fitted slope is around 0.488, significantly less than the Asian option's 1.9, confirming our expectation that the Asian option's smoother payoff function results in a stronger variance decay.

## Multi-level MC vs Classic MC
In this experiment we compare the cost against error $\varepsilon$ of MLMC and classic MC. For both MLMC and MC we split the error between discretization bias and MC variance evenly. 
### Asian
To start, let $\varepsilon > 0$.
* Classic MC
    * Number of steps $n = \lceil \frac{2T}{\varepsilon} \rceil$ makes the discretization error $\leq \varepsilon/2$
    * For the number of paths $N$ we first let $N_0$ be an arbitrary number of paths for a test/pilot run, here 20 000 is used. After we compute $SE_0$ from the pilot run, we then let $N = \lceil N_0 \cdot \left( \frac{2SE_0}{\varepsilon}\right) ^2 \rceil$. This ensures that the number of paths is appropriately scaled so that the variance (SE) is less than $\varepsilon/2$.
* MLMC
    * We let the max level $L = 10$. However, there is an option to use an adaptive max level that keeps incrementing $L$ and testing whether $|\hat V_{L+1} - \hat V_L| \leq \varepsilon / 2$: if this condition is fulfilled accept that $L$. This controls the bias of our MLMC estimate.
    * We want to make $SE_{MLMC} \leq \varepsilon / 2$ while minimizing the total cost $C = \sum_{l=0}^L C_lN_l$, where $C_l$ is the cost per sample on level $l$. The solution to this optimization problem is $N_k = \left( \frac{2}{\varepsilon} \right)^2 \cdot \sum_{l=0}^L \sqrt{Var(\Delta_l) \cdot C_l} \cdot \sqrt{\frac{Var(\Delta_k)}{C_k}}$. Clearly, neither $Var(\Delta_l)$, nor $C_l$ is a known quantity, so we estimate it using a pilot run of 500 samples per level, keeping track of the corrections summed up over the pilot runs. We then calculate the number of additional samples needed per level, finally estimating $Y_l$ for each level to find our estimate $\hat V_L$ and $SE_{MLMC}$.
  
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/7d2b78a1-a5e5-41c0-a430-4552b5672f7a" />

We can clearly see that as $\varepsilon$ gets smaller, the runtime vs accuracy loglog plot of MC runs parallel to $O(\varepsilon^{-3})$, MLMC runs parallel to $(\varepsilon^{-2})$, and that the cost of MLMC becomes significantly less than MC. The flatter part of the MLMC graph towards the right side of the plot can be attributed to the pilot runs doing more work than was necessary for larger $\varepsilon$. This plot shows us that for Asian options, MLMC is much more efficient than classic MC.

### Barrier (Naive)
Similarly, let $\varepsilon > 0$.
* Classic MC
    * Since discretization error for barrier options is $O(\sqrt{h})$, we let the number of steps $n = \lceil \frac{4T}{\varepsilon^2} \rceil$.
    * We compute the number of paths $N$ in similar fashion described above.
* MLMC
    * Similar to the Asian option, we have the option of finding the adaptive max level, but we use $L=12$ to simplify computation.
    * We follow the same procedure described in the Asian option section above.

If we recall the variance decay of the barrier option, we can see that the fitted slope is around 0.488, implying that $Var(\Delta_l) = O(h_l^{0.488}) \approx O(h_l^{0.5})$. This is significantly worse than the Asian option, and we expect both MC and MLMC to perform as such. For classic MC, we expect to see cost $C_{MC} = O(\varepsilon^{-4})$ primarily due to the a higher number of steps being required per sample, scaling as $n = O(\varepsilon^{-2})$. Similarly, we expect MLMC to perform worse than the $O(\varepsilon^{-2})$ scaling we saw with the Asian option. This can be attributed to the rate of variance decay $\approx O(h_l^{0.5})$ being less than the cost growth per sample $O(h^{-1})$ (in absolute values). Overall, we anticipate $C_{MLMC} \approx O(\varepsilon^{-3})$.

### Barrier (with Brownian bridge)

  






