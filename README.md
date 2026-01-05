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
our number of samples $N = O(\varepsilon^{-2}$ (by CLT, the SE of classic MC is $\sigma / \sqrt{N}$). Similarly, the discretization bias for Asian and Barrier call options is $O(h)$, and since $h = T/n$, we can 
conclude that the number of points per sample $n = O(\varepsilon^{-1})$. Thus, the combined computational cost for classic MC is $O(\varepsilon^{-2} \times O(\varepsilon^{-1}) = O(\varepsilon^{-3})$, i.e., if you want to 
halve the error, you must perform 8x the computational cost.

Multi-level MC solves this issue by using the telescoping sum mentioned above $\mathbb{E}[V_L] = \mathbb{E}[V_0] + \sum_{l=1}^L \mathbb{E}[V_l] - \mathbb{E}[V_{l-1}]$. The total number of required paths still remains
$N = O(\varepsilon^{-2}$, but the average cost per path becomes $O(1)$, due to the significantly lowered number of samples required at higher levels. Thus, the total computational cost is $O(\varepsilon^{-2}$, a clear
improvement over classic MC.

## Multilevel Monte Carlo (MLMC)
Multilevel Monte Carlo (MLMC) is a variance–reduction technique that accelerates Monte Carlo estimation by exploiting a hierarchy of discretizations. Rather than estimating an expectation using only the finest time step, MLMC combines information from multiple levels of resolution in a way that minimizes total computational cost.
Let \(P_\ell\) denote the payoff computed using a time step \(h_\ell = T / 2^\ell\). MLMC is based on the telescoping identity
\[
\mathbb{E}[P_L]
=\mathbb{E}[P_0]
+
\sum_{\ell=1}^L \mathbb{E}[P_\ell - P_{\ell-1}],
\]
which expresses the finest–level expectation as a sum of expectations of differences between successive levels. Each term in this sum is estimated independently using Monte Carlo sampling.

The effectiveness of MLMC relies on coupling the simulations at levels \(\ell\) and \(\ell-1\) so that the variance of the correction \(P_\ell - P_{\ell-1}\) decays rapidly as the time step is refined. In practice, this is achieved by constructing the fine and coarse paths from shared underlying random increments. When the payoff is sufficiently regular, this coupling leads to a strong correlation between \(P_\ell\) and \(P_{\ell-1}\), resulting in a much smaller variance for their difference than for either payoff individually.

Because the variance of the level corrections decreases with \(\ell\) while the cost per sample increases, MLMC allocates many samples to coarse, inexpensive levels and progressively fewer samples to finer, more expensive levels. An optimal allocation balances variance and cost across levels, yielding a Monte Carlo estimator whose total variance is controlled while minimizing computational effort.

Under suitable conditions on the payoff regularity and discretization scheme, MLMC achieves an overall computational complexity of \(O(\varepsilon^{-2})\) for a target RMSE \(\varepsilon\), representing a significant improvement over single–level Monte Carlo. This project applies MLMC to option pricing problems and demonstrates its efficiency gains empirically.
