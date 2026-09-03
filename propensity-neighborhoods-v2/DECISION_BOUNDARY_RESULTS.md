# Decision-boundary neighborhood results

## Motivation

The earlier ranking experiment showed that propensity neighborhoods contain incremental pairwise ordering information, but that information did not improve the winning action. This experiment concentrates the neighborhood on the decision that matters: whether the preliminary model's top two programs are reversed.

For the preliminary top two programs $a$ and $b$, the local corrected margin is

$$
\widehat d_{ab}(x)
=
\widetilde p_a(x)-\widetilde p_b(x)
+
\lambda
\frac{1}{|\mathcal N_{ab}(x)|}
\sum_{i\in\mathcal N_{ab}(x)}
\left[
(Y_{ia}-Y_{ib})
-(\widetilde p_{ia}-\widetilde p_{ib})
\right].
$$

The method swaps $a$ and $b$ only if the corrected margin becomes negative. Tested geometries use the two relevant propensity coordinates, their margin, agreement on the preliminary winner, and optionally the behavioral-embedding distance.

## Available opportunity

There is ample theoretical room for a successful decision correction.

| Split set | True winner in base top 2 | True winner in base top 3 | Top-2 oracle regret | Top-3 oracle regret |
|---|---:|---:|---:|---:|
| Initial ten splits | 66.67% | 82.86% | 0.015177 | 0.006941 |
| Fresh ten splits | 67.28% | 83.57% | 0.014702 | 0.006602 |

The base top-1 regret is approximately 0.037. A perfect selector restricted to the base top two would reduce it to approximately 0.015; a perfect top-three selector would reduce it below 0.007. Candidate generation is therefore not the immediate bottleneck.

## Ungated local-margin rule

The initial ten splits appeared promising.

| Method | Initial regret | Fresh regret |
|---|---:|---:|
| Base propensity | 0.038726 | **0.036119** |
| Tuned decision neighborhood | 0.038397 | 0.036790 |
| Fixed decision neighborhood | **0.037403** | 0.036673 |

The fixed rule used pair-specific propensity plus embedding distance, 50 neighbors, and residual shrinkage 50. Initially it improved regret by 0.001323, won eight of ten splits, and had a paired interval of $[-0.002234,-0.000412]$ relative to the base.

That result did not replicate. On the ten fresh splits, the same rule raised regret by 0.000554 and won five of ten splits. Across all 20 splits its mean change is

$$
\Delta_{\mathrm{regret}}=-0.000384,
\qquad
95\%\text{ paired interval }[-0.001159,0.000390],
$$

with wins on 13 of 20 splits.

## Uncertainty-gated swaps

A second version required the corrected margin to cross zero by a multiple $\gamma$ of its estimated neighborhood standard error:

$$
\widehat d_{ab}(x)<-\gamma\widehat{\operatorname{se}}_{ab}(x).
$$

The tuning grid used $\gamma\in\{0,0.5,1,1.5,2\}$. On the initial ten splits, the tuned gated method had regret 0.037845 versus 0.038726 for the base, but its paired interval crossed zero. Validation chose no gate on six splits and different positive gates on the other four. Only 50.7% of its overrides were beneficial.

A fixed one-standard-error gate reduced the initial apparent improvement to 0.000356, also with an interval crossing zero. Because the ungated rule had already failed on fresh splits and the gated initial result was not stable, another confirmation set was not run.

## Conclusion

The diagnostic resolves part of the confusion:

- the base candidate set leaves substantial recoverable regret;
- neighborhoods sometimes identify genuine top-two inversions;
- but the tested peer residual margin is only slightly better than a coin flip at deciding when to swap; and
- neither shrinkage nor an empirical standard-error gate makes the swap decision stable.

The failure is therefore not caused by a lack of candidate headroom. It is caused by weak correspondence between a target user's top-two residual margin and the average residual margin of the retrieved peers.

This points away from additional variants of peer averaging. A materially different next step would need to learn which historical peers are reliable for a particular program pair—pair-specific metric learning or attention—using a separate training objective and enough data to validate that extra flexibility.
