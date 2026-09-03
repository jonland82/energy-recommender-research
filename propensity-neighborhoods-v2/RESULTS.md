# First-round experimental results

## Bottom line

The experiments find reproducible incremental *pairwise ranking information* in propensity-space neighborhoods, but none of the tested rankers converts it into better top-1 recommendations.

That distinction matters. Propensity-neighborhood features lower held-out pairwise log loss relative to an otherwise identical base-score ranker on every one of 20 splits. They do not lower top-1 regret. The evidence therefore supports the information hypothesis in a narrow sense, while rejecting the stronger claim that the current neighborhood rankers improve the winning action.

No existing paper files were changed.

## Experimental design

The experiment preserves the current paper's chronological construction:

- the first 60% of each user's interactions create the behavioral representation;
- the last 40% create the future category-propensity targets;
- 60% of users form the estimation set, 25% tune hyperparameters, and 15% are untouched test users;
- a 32-coordinate SVD representation and static features fit the multi-output ridge propensity model;
- five-fold cross-fitted estimation predictions define reference propensity rows and residuals;
- each estimation user is excluded from its own neighborhood;
- neighborhoods use $k\in\{25,50,100,200\}$ in propensity space and embedding space;
- regularized pairwise logistic rankers are trained only with estimation-user outcomes;
- ranker regularization and plausible-set thresholds are selected only with tuning users.

The pairwise loss weights every program comparison by its future propensity gap. Neighborhood features include residual mean and dispersion, peer outcome mean, peer winning frequency, peer margin, corrected propensity, and distance interactions across all four neighborhood sizes.

## Initial ten splits

Seeds 20260902--20260911 were used for the initial experiment and geometry ablations.

| Method | Top-1 regret | Hit rate | NDCG | Pairwise log loss |
|---|---:|---:|---:|---:|
| Base propensity ordering | 0.038726 | 37.04% | 0.975672 | 0.637425 |
| Fixed propensity peer mean | 0.038365 | 37.14% | 0.976114 | 0.637425 |
| Base pairwise ranker | 0.037141 | 36.48% | 0.977340 | 0.209151 |
| Propensity-neighborhood ranker | 0.037311 | 36.10% | **0.977501** | **0.205154** |
| Embedding-neighborhood ranker | **0.037110** | 36.57% | 0.976969 | 0.212101 |
| Combined-neighborhood ranker | 0.037531 | 36.01% | 0.976839 | 0.212622 |
| Restricted base ranker | 0.037076 | 36.57% | 0.976625 | 0.536699 |
| Restricted neighborhood ranker | **0.037007** | 36.53% | 0.976779 | 0.535153 |

The restricted neighborhood ranker appeared promising: its regret was 4.44% below direct propensity ordering and it won on eight of ten splits. Its incremental change relative to the restricted base ranker was only $-0.000069$, however, with a paired 95% interval of $[-0.001931,0.001793]$. The apparent gain could not be attributed specifically to neighborhood features.

## Fresh ten-split confirmation

The full specification was then frozen and evaluated on seeds 20260912--20260921.

| Method | Top-1 regret | Hit rate | NDCG | Pairwise log loss |
|---|---:|---:|---:|---:|
| Base propensity ordering | **0.036119** | **39.30%** | 0.976716 | 0.637307 |
| Fixed propensity peer mean | 0.036931 | 38.83% | 0.976813 | 0.637297 |
| Base pairwise ranker | 0.037396 | 36.57% | 0.977515 | 0.207242 |
| Propensity-neighborhood ranker | 0.037757 | 35.82% | **0.977527** | **0.203715** |
| Embedding-neighborhood ranker | 0.037303 | 38.31% | 0.976847 | 0.211897 |
| Combined-neighborhood ranker | 0.037858 | 37.18% | 0.976598 | 0.212275 |
| Restricted base ranker | 0.036871 | 38.31% | 0.976969 | 0.546181 |
| Restricted neighborhood ranker | 0.037746 | 38.40% | 0.976703 | 0.534222 |

The initial regret improvement did not replicate. The restricted neighborhood ranker raised regret by 0.001627, or 4.50%, relative to direct propensity ordering and beat it on only three of ten fresh splits. It was also worse than the restricted base ranker by 0.000875 on average.

## Information test across all 20 splits

The cleanest incremental comparison holds the pairwise learner fixed and adds only propensity-neighborhood features:

$$
\Delta_{\mathrm{logloss}}
=
\mathcal L_{\mathrm{propensity\ neighborhood}}
-
\mathcal L_{\mathrm{base\ ranker}}.
$$

Across all 20 splits,

$$
\overline\Delta_{\mathrm{logloss}}=-0.003762,
\qquad
95\%\text{ paired interval }[-0.004529,-0.002995].
$$

Propensity-neighborhood features lower pairwise log loss on all 20 splits. This is evidence that the neighborhood summaries contain incremental information about pairwise program ordering after conditioning on the base propensity features.

That information does not improve the top choice:

$$
\overline\Delta_{\mathrm{top1\ regret}}=+0.000265,
\qquad
95\%\text{ paired interval }[-0.000401,0.000932],
$$

with neighborhood-ranker wins on nine of 20 splits. Lower pairwise cross-entropy and slightly better NDCG can coexist with worse top-1 regret because most correctly improved pairs do not determine the winning program.

Embedding-space neighborhood features do not show the same information gain. Their pairwise log loss is 0.003802 higher than the base ranker across 20 splits, with a paired interval of $[0.001760,0.005845]$.

The intervals above describe stability over repeated, overlapping user splits. They are not independent-sample inferential intervals.

## Follow-up losses and candidate-conditioned training

Two additional variants were explored on the initial ten splits:

1. A top-focused pairwise loss compared each user's best program only with its alternatives. It performed worse: regret was 0.040498 for the propensity-neighborhood version and 0.040238 for its base-only control, compared with 0.038726 for direct ordering.
2. A candidate-conditioned propensity ranker was trained only on pairs that fell inside each estimation user's calibrated plausible set. Its regret was 0.038301, essentially the same as the fixed peer mean at 0.038365 and worse than the ordinary base pairwise ranker at 0.037141.

These results suggest that the failure is not fixed simply by concentrating the existing linear loss on top or plausible-set comparisons.

## Interpretation

The first-round answer is:

- **Yes:** propensity neighborhoods contain measurable incremental ordering information. The 20-of-20 pairwise log-loss result is stronger than the earlier RMSE-only evidence.
- **Not yet:** the tested linear pairwise, restricted, top-focused, and candidate-conditioned rankers do not turn that information into a reproducible top-1 regret improvement.
- **The theorem remains useful:** the bound $2b_m(x)+\xi_\theta(x)$ and plausible-set restriction remain valid ways to price a learned ranker's departure from the calibrated ordering. The experiments do not yet justify claiming that the learned departure improves decisions.

That proposed choice-boundary experiment was subsequently completed and is reported in `DECISION_BOUNDARY_RESULTS.md`. It found substantial top-two candidate headroom, but neighborhood-triggered swaps were beneficial only about half the time and did not confirm on fresh splits. The accumulated conclusion is recorded in `OVERALL_CONCLUSION.md`.
