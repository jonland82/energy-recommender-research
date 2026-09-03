# Neighborhood preprocessing results

## Question

Can neighborhoods improve recommendations when they are used to construct features before the final propensity model, rather than to rerank or correct scores afterward?

The tested pipeline is

$$
x
\longrightarrow
\widetilde p^{(0)}(x)
\longrightarrow
\mathcal N(x)
\longrightarrow
\Phi_{\mathcal N}(x)
\longrightarrow
\widehat p^{(1)}(x)
\longrightarrow
\arg\max_a\widehat p^{(1)}_a(x).
$$

The final recommendation is still obtained by ordinary descending propensity sorting. There is no post-hoc ranker.

## Leakage-safe construction

The preliminary propensity model is cross-fitted on estimation users. For every estimation user, its neighborhood excludes that user. Reference outcomes from the remaining estimation users create multiscale summaries at $k\in\{25,50,100,200\}$:

- peer residual mean and dispersion;
- peer outcome mean;
- peer winning frequency;
- peer outcome margin;
- shrunk corrected propensity; and
- interactions with neighborhood distance.

The second stage learns an additive propensity residual rather than replacing the preliminary estimate:

$$
\widehat p^{(1)}_a(x)
=
\widetilde p^{(0)}_a(x)
+
h_a\!\left(x,\widetilde p^{(0)}(x),\Phi_{\mathcal N,a}(x)\right).
$$

Strong regularization lets $h_a$ shrink toward zero. A stacked control uses the same second-stage model and preliminary propensity row without neighborhood summaries. Tuning users select regularization; test outcomes are used only for final evaluation.

## Initial ten splits

| Method | RMSE | Top-1 regret | Hit rate | NDCG |
|---|---:|---:|---:|---:|
| Preliminary base | **0.086310** | 0.038726 | 37.04% | **0.975672** |
| Stacked control | 0.086547 | 0.039069 | 36.85% | 0.975415 |
| Propensity-neighborhood preprocessing | 0.086484 | 0.038869 | 36.62% | 0.975542 |
| Embedding-neighborhood preprocessing | 0.086434 | **0.038613** | **37.18%** | 0.975599 |
| Combined preprocessing | 0.086311 | 0.039075 | 36.53% | 0.975503 |

Embedding-neighborhood preprocessing improved regret relative to the stacked control by $0.000456$, with a paired 95% interval of $[0.000026,0.000886]$ in the favorable direction. Relative to the preliminary base, its improvement was only $0.000113$, with an interval crossing zero.

## Fresh ten-split confirmation

The preprocessing specification was frozen and evaluated on seeds 20260912--20260921.

| Method | RMSE | Top-1 regret | Hit rate | NDCG |
|---|---:|---:|---:|---:|
| Preliminary base | **0.083850** | 0.036119 | 39.30% | **0.976716** |
| Stacked control | 0.083932 | 0.036352 | 39.06% | 0.976525 |
| Propensity-neighborhood preprocessing | 0.083949 | 0.036465 | 39.25% | 0.976476 |
| Embedding-neighborhood preprocessing | 0.083878 | **0.035881** | **39.67%** | 0.976664 |
| Combined preprocessing | 0.083827 | 0.036333 | 39.34% | 0.976591 |

The embedding-neighborhood result repeats:

$$
\Delta_{\mathrm{regret\ vs\ stacked}}
=-0.000471,
\qquad
95\%\text{ paired interval }[-0.000865,-0.000078].
$$

Against the preliminary base,

$$
\Delta_{\mathrm{regret\ vs\ base}}
=-0.000239,
\qquad
95\%\text{ paired interval }[-0.000523,0.000046].
$$

The latter is a 0.66% mean regret reduction, but the interval narrowly includes zero and the method wins on only four of ten fresh splits.

## Combined twenty-split view

Across all 20 splits, embedding-neighborhood preprocessing has mean regret 0.037247, compared with 0.037422 for the preliminary base and 0.037710 for the stacked control.

Relative to the stacked control:

$$
\Delta_{\mathrm{regret}}=-0.000464,
\qquad
95\%\text{ paired interval }[-0.000726,-0.000201],
$$

and hit rate rises by 0.47 percentage points, with a paired interval of $[0.14,0.80]$ percentage points.

Relative to the preliminary base:

$$
\Delta_{\mathrm{regret}}=-0.000176,
\qquad
95\%\text{ paired interval }[-0.000507,0.000156].
$$

The improvement over the stacked control is reproducible. The improvement over the model that matters operationally—the original preliminary base—is small and not established by these splits. RMSE is also slightly worse than the base, so any possible benefit is specific to the top decision rather than probability calibration.

Propensity-space preprocessing does not help. Across 20 splits it raises regret by 0.000244 relative to the preliminary base. Combining propensity and embedding neighborhoods also does not improve regret.

## Conclusion

Preprocessing is more promising than post-hoc neighborhood reranking, but the current result is suggestive rather than sufficient:

- embedding-neighborhood summaries consistently repair some damage introduced by a generic second-stage model;
- they produce a small mean improvement over the original model on both the initial and fresh split sets;
- that improvement is not stable enough across users and splits to claim a dependable operational gain; and
- propensity-space summaries remain unhelpful for the winning action.

The subsequent direct embedding-smoothing and augmentation experiment is reported in `EMBEDDING_NEIGHBORHOOD_RESULTS.md`. Fixed augmentation produced a small RMSE improvement but no ranking improvement, reinforcing the conclusion that the multiscale preprocessing vector contains more variance than useful top-choice signal. The accumulated conclusion is recorded in `OVERALL_CONCLUSION.md`.
