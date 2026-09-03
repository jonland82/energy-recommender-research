# Worked Example: Neighborhood Calibration and the Refinement Bound

This example shows how a propensity neighborhood changes a recommendation, how adding coordinates makes neighborhoods thinner, and how that thinning affects the finite-sample bound.

## 1. Begin with the base propensity scores

Suppose a target customer $x$ is eligible for two programs, $A$ and $B$. The fitted propensity model gives

$$
\widetilde p_A(x)=0.58,
\qquad
\widetilde p_B(x)=0.49.
$$

The base model therefore recommends program $A$.

For illustration, suppose the true—but ordinarily unknown—propensities are

$$
p_A(x)=0.47,
\qquad
p_B(x)=0.62.
$$

The target customer's true propensity-model residuals are consequently

$$
\rho_A(x)=p_A(x)-\widetilde p_A(x)=0.47-0.58=-0.11,
$$

and

$$
\rho_B(x)=p_B(x)-\widetilde p_B(x)=0.62-0.49=0.13.
$$

The base model overestimates program $A$ and underestimates program $B$. Neighborhood calibration attempts to identify this pattern using similar historical customers.

## 2. Construct nested neighborhoods

Let the distance between the target and historical customer $x_i$ be built from cumulative coordinate blocks:

$$
d_m(x,x_i)=\sum_{j=1}^{m}\delta_j(x,x_i),
\qquad
\delta_j(x,x_i)\geq 0.
$$

Because every added contribution is nonnegative,

$$
d_{m+1}(x,x_i)\geq d_m(x,x_i).
$$

At a fixed radius $r$, define the level-$m$ neighborhood by

$$
\mathcal N_m(x;r)=\{i:d_m(x,x_i)\leq r\},
\qquad
n_m=|\mathcal N_m(x;r)|.
$$

It follows immediately that

$$
\mathcal N_{m+1}(x;r)\subseteq\mathcal N_m(x;r),
\qquad
n_{m+1}\leq n_m.
$$

Thus, adding coordinates can only retain or remove peers; it cannot add new peers to a fixed-radius neighborhood.

For example, take $r=0.20$. One historical customer might have cumulative distances

$$
d_1=0.06,
\qquad
d_2=0.06+0.08=0.14,
\qquad
d_3=0.14+0.09=0.23.
$$

This customer belongs to levels 1 and 2 but is removed at level 3 because $0.23>0.20$.

Suppose the full historical population produces the following nested neighborhoods:

| Level | Coordinates used | Number of peers $n_m$ |
|---|---|---:|
| 1 | Basic customer attributes | 5,000 |
| 2 | Attributes and propensity coordinates | 1,000 |
| 3 | Attributes, propensity, and detailed behavior | 100 |

The additional coordinates make the neighborhoods progressively thinner:

$$
\mathcal N_3(x;r)
\subseteq
\mathcal N_2(x;r)
\subseteq
\mathcal N_1(x;r).
$$

## 3. Calculate the neighborhood residual corrections

For each historical peer and program, calculate an out-of-fold residual

$$
R_{ia}=Y_{ia}-\widetilde p_a(x_i).
$$

At level $m$, the local residual correction for program $a$ is

$$
\overline R_{a,m}(x)
=
\frac{1}{n_m}
\sum_{i\in\mathcal N_m(x;r)}R_{ia}.
$$

Suppose the neighborhood averages are

| Level | Mean residual for $A$ | Mean residual for $B$ |
|---|---:|---:|
| 1 | $-0.035$ | $+0.045$ |
| 2 | $-0.105$ | $+0.125$ |
| 3 | $-0.110$ | $+0.130$ |

The broad level-1 neighborhood detects only a weak calibration error. The thinner neighborhoods contain peers whose residual patterns are much closer to the target residuals $(-0.11,+0.13)$.

## 4. Correct the target scores

The neighborhood-calibrated estimate is

$$
\widehat p_{a,m}(x)
=
\widetilde p_a(x)
+
\overline R_{a,m}(x).
$$

The corrected scores are therefore

| Level | $\widehat p_{A,m}(x)$ | $\widehat p_{B,m}(x)$ | Recommendation |
|---|---:|---:|---|
| Base | $0.580$ | $0.490$ | $A$ |
| 1 | $0.580-0.035=0.545$ | $0.490+0.045=0.535$ | $A$ |
| 2 | $0.580-0.105=0.475$ | $0.490+0.125=0.615$ | $B$ |
| 3 | $0.580-0.110=0.470$ | $0.490+0.130=0.620$ | $B$ |

Level 2 changes the ranking from the wrong program, $A$, to the correct program, $B$.

## 5. See how thinning changes both terms in the bound

Let the worst residual mismatch within a neighborhood be

$$
\varepsilon_m^\rho(x)
=
\max_{i\in\mathcal N_m(x;r)}
\|\rho(x_i)-\rho(x)\|_\infty.
$$

Because each new neighborhood is a subset of the preceding neighborhood, removing peers cannot increase this maximum:

$$
\varepsilon_{m+1}^\rho(x)
\leq
\varepsilon_m^\rho(x).
$$

Thinner neighborhoods can therefore improve local match quality. At the same time, thinning reduces $n_m$, which increases sampling uncertainty.

When one of $M$ levels is selected using the calibration outcomes, the bound uses

$$
b_m(x)
=
\varepsilon_m^\rho(x)
+
B\sqrt{
\frac{\log(2K_xM/\alpha)}{2n_m}
}.
$$

The recommendation-regret guarantee is

$$
p_{a^*}(x)-p_{\widehat a_m}(x)
\leq 2b_m(x)
$$

with probability at least $1-\alpha$ under the theorem's assumptions.

Take

$$
K_x=2,
\qquad
M=3,
\qquad
\alpha=0.10,
\qquad
B=1.
$$

Then

$$
\log(2K_xM/\alpha)
=
\log(120)
\approx 4.787.
$$

Suppose the neighborhood mismatch values are

$$
\varepsilon_1^\rho=0.090,
\qquad
\varepsilon_2^\rho=0.015,
\qquad
\varepsilon_3^\rho=0.005.
$$

The complete bound calculation is

| Level | Mismatch $\varepsilon_m^\rho$ | Sampling uncertainty | $b_m$ | Regret bound $2b_m$ |
|---|---:|---:|---:|---:|
| 1 | $0.090$ | $\sqrt{4.787/10000}=0.0219$ | $0.1119$ | $0.2238$ |
| 2 | $0.015$ | $\sqrt{4.787/2000}=0.0489$ | **$0.0639$** | **$0.1278$** |
| 3 | $0.005$ | $\sqrt{4.787/200}=0.1547$ | $0.1597$ | $0.3194$ |

The two parts of the bound move in opposite directions:

$$
\underbrace{\varepsilon_m^\rho(x)}_{\text{better matching decreases}}
\quad+
\underbrace{
B\sqrt{\frac{\log(2K_xM/\alpha)}{2n_m}}
}_{\text{fewer peers increases}}.
$$

Level 1 has many observations but poorly matched peers. Level 3 has exceptionally well-matched peers but too few observations. Level 2 gives the smallest bound because its gain in match quality is worth the loss of peer support.

This is the refinement frontier.

## 6. Apply the recommendation-margin certificate

At level 2, program $B$ has the estimated lead

$$
\Delta_2(x)
=
0.615-0.475
=
0.140.
$$

The certificate requires this lead to exceed

$$
2b_2(x)
=
2(0.0639)
=
0.1278.
$$

Since

$$
0.140>0.1278,
$$

the bound certifies, at the stated confidence level, that program $B$ is the unique true best program.

## 7. What the extra coordinates have to do with the theorem

The logical chain is

$$
\text{more coordinates}
\Longrightarrow
\text{larger cumulative distances}
\Longrightarrow
\text{thinner fixed-radius neighborhoods}
$$

and consequently

$$
\text{thinner neighborhoods}
\Longrightarrow
\begin{cases}
\text{potentially smaller residual mismatch},\\
\text{necessarily weaker peer support}.
\end{cases}
$$

The propositions establish these geometric effects. The theorem prices their statistical consequences.

Moving from level $m$ to level $m+1$ improves the bound precisely when

$$
\varepsilon_m^\rho(x)-\varepsilon_{m+1}^\rho(x)
>
B\sqrt{\frac{\log(2K_xM/\alpha)}{2}}
\left(
\frac{1}{\sqrt{n_{m+1}}}
-
\frac{1}{\sqrt{n_m}}
\right).
$$

In words, another coordinate block is worthwhile only when the reduction in peer mismatch exceeds the additional sampling uncertainty caused by the thinner neighborhood.

The overall workflow is

$$
\text{customer coordinates}
\longrightarrow
\text{nested peer sets}
\longrightarrow
\text{peer residual averages}
\longrightarrow
\text{calibrated scores}
\longrightarrow
\text{match/support bound}
\longrightarrow
\text{certified recommendation}.
$$

The neighborhood therefore matters twice: it determines the correction applied to the propensity scores, and its match quality and size determine whether that correction is statistically trustworthy.
