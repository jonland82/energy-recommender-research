# Operational residual-neighborhood experiment

## Question

The paper's regret theorem is written in terms of the unknown target residual. This experiment asks what can be computed at recommendation time without using the test user's future outcome, and what guarantee survives that replacement.

The answer has two parts. A theorem-faithful max-neighbor envelope is operational but conservative. For the peer-mean calibrator used here, linearity gives a tighter operational corollary that selects neighborhoods often enough to test their empirical value.

## Protocol

The KuaiRec sample contains 1,411 users and 12 action categories. For each of 20 random seeds, users are divided into five disjoint roles:

1. 634 users fit the base propensity model;
2. 211 users fit a residual model;
3. 211 users supply observed peer residuals;
4. 141 users calibrate conformal radii; and
5. 214 users form an untouched test set.

Thus 4,280 held-out-user evaluations are aggregated. No test outcome is used to construct a neighborhood, choose a neighborhood level, or calculate an operational bound. Unlike the paper's earlier chronological embedding experiment, this diagnostic uses each held-out user's observed full-period category-rate vector; it does not call that vector a future target.

Write the base-model residual vector as

$$
R(x)=Y(x)-\widehat p_0(x),
$$

and let $\widehat\rho(x)$ predict that residual from a separately trained model. Split conformal calibration supplies $\eta$ such that, marginally over an exchangeable future user,

$$
\Pr\!\left(\lVert R(x)-\widehat\rho(x)\rVert_\infty\leq\eta\right)\geq 1-\alpha.
$$

For a neighborhood $N_m(x)$, the implemented correction is a shrunk peer mean,

$$
c_m(x)=w_m\frac{1}{|N_m(x)|}\sum_{i\in N_m(x)}R(x_i),
\qquad
w_m=\frac{|N_m(x)|}{|N_m(x)|+20}.
$$

The theorem-faithful observable envelope is

$$
B_m^{\max}(x)
=
w_m\max_{i\in N_m(x)}
\lVert \widehat\rho(x)-R(x_i)\rVert_\infty
+(1-w_m)\lVert\widehat\rho(x)\rVert_\infty.
$$

It implies

$$
\lVert R(x)-c_m(x)\rVert_\infty
\leq \eta+B_m^{\max}(x)
$$

on the conformal event. The selection rule also includes level zero, meaning no correction, so it can safely fall back to the base model.

Because $c_m(x)$ is an observed peer mean, a tighter corollary is available:

$$
B_m^{\mathrm{mean}}(x)
=
\lVert\widehat\rho(x)-c_m(x)\rVert_\infty,
$$

and therefore

$$
\lVert R(x)-c_m(x)\rVert_\infty
\leq \eta+B_m^{\mathrm{mean}}(x).
$$

Both quantities are known at recommendation time. A second split-conformal construction calibrates the complete peer-mean selection rule end to end rather than composing the residual band with the observable proxy.

## Primary results

The nominal miscoverage is $\alpha=0.10$, and the primary neighborhood radius is the 0.12 quantile of full-level residual-to-reference distances.

| Method | RMSE | Mean top-1 regret | Hit rate | Neighborhood use |
|---|---:|---:|---:|---:|
| Base model | 0.110439 | 0.019361 | 0.4918 | 0% |
| Theorem-faithful frontier | 0.110438 | 0.019361 | 0.4918 | 0.09% |
| Peer-mean frontier | 0.110129 | 0.019257 | 0.4935 | 63.36% |
| Legacy score-space proxy | 0.110318 | 0.019429 | 0.4900 | 100% |

The theorem-faithful frontier selected the base model for 4,276 of 4,280 test users. This is not a coverage failure: the max over every neighbor is simply too conservative to justify a correction except in four cases.

The peer-mean frontier selected a neighborhood for 2,712 users. Relative to the base model, its paired mean RMSE change was $-0.000310$, with a descriptive split-level 95% $t$ interval of $[-0.000560,-0.000060]$. Because repeated test sets overlap, this interval summarizes stability across splits rather than independent-sample inference. Its changes in top-1 regret and hit rate were not distinguishable from zero: their paired intervals were $[-0.000526,0.000318]$ and $[-0.00378,0.00705]$, respectively.

## Coverage and bound width

| Certificate | Max-error coverage | Mean regret bound | Nonvacuous rate |
|---|---:|---:|---:|
| Residual band + max-neighbor envelope | 0.9465 | 0.5645 | 1.000 |
| Residual band + peer-mean proxy | 0.9411 | 0.5543 | 1.000 |
| End-to-end conformal peer-mean rule | 0.9093 | 0.5053 | 1.000 |
| Base conformal benchmark | 0.9096 | 0.5018 | 1.000 |
| Legacy retrospective diagnostic | 1.0000 | 1.0730 | 0.275 |

The direct composed bounds over-cover because they combine a 90% residual band with a nonnegative observable discrepancy. The end-to-end certificate is close to its nominal 90% coverage and is much tighter than the legacy retrospective diagnostic. It is still slightly wider than the base conformal benchmark, so the current data do not show that neighborhood calibration improves certificate sharpness.

All reported regret inequalities covered 100% of evaluated test cases, but this is not evidence of exact 100% theoretical coverage: the regret bound is loose relative to the small realized regrets. The score-gap condition certified no individual top-1 choice because even the operational bands remained larger than the predicted action margins.

## Radius sensitivity

| Radius quantile | Peer-mean RMSE | Change vs. base | 95% paired interval | End-to-end coverage | Neighborhood use |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.110009 | -0.000430 | $[-0.000638,-0.000222]$ | 0.9086 | 66.29% |
| 0.12 | 0.110129 | -0.000310 | $[-0.000560,-0.000060]$ | 0.9093 | 63.36% |
| 0.25 | 0.110074 | -0.000365 | $[-0.000535,-0.000195]$ | 0.9100 | 59.98% |

The small RMSE gain is robust to these radius choices. None of the three settings establishes an improvement in recommendation regret or hit rate.

## Known propensities at test time

Some application coordinates are not latent. Writing the paper's latent propensity residual as $\rho(x)=p(x)-\widehat p_0(x)$, if current enrollment makes $p_j(x)=1$, then that coordinate is known exactly:

$$
\rho_j(x)=1-\widehat p_{0,j}(x).
$$

For that coordinate, the uncertainty radius is $\eta_j=0$ and the exact residual can be inserted into the neighborhood calculation. A mixed certificate can therefore use zero uncertainty on known coordinates and conformal uncertainty only on unknown coordinates. KuaiRec does not mark any category propensity as deterministically known at test time, so this experiment does not simulate that advantage.

## Interpretation before changing the paper

The experiment supports a narrow claim: an operational peer-mean corollary is testable without target-outcome leakage, attains its intended marginal coverage, and yields a small but consistent improvement in propensity-vector RMSE. It does not support a claim that neighborhood calibration improves top-1 recommendations or tightens certificates relative to a base conformal benchmark on KuaiRec.

The distinction between targets matters. These conformal statements cover the observed category-rate vector of a held-out exchangeable user. They do not provide distribution-free pointwise coverage of the latent conditional propensity vector. A latent-propensity theorem requires either an explicit confidence-band assumption for $\widehat\rho$, known test-time coordinates, or additional repeated-outcome structure.

## Reproduction

Primary run:

```powershell
python scripts/kuairec_operational_frontier.py --output results/kuairec_operational_frontier
```

Sensitivity runs:

```powershell
python scripts/kuairec_operational_frontier.py --radius-quantile 0.05 --output results/kuairec_operational_frontier_q05
python scripts/kuairec_operational_frontier.py --radius-quantile 0.25 --output results/kuairec_operational_frontier_q25
```

Machine-readable results are in `summary.json` and per-seed method metrics are in `metrics.csv` in each output directory.
