# Neighborhood-informed embedding experiment

## Construction

This experiment places the neighborhood before propensity scoring and ranking. A leakage-safe SVD embedding $z(x)$ is constructed from pre-decision behavior. Each estimation user is excluded from its own cosine-distance neighborhood, and tuning and test users retrieve only estimation users.

Two representations are tested:

$$
z_{\mathrm{smooth}}(x)
=
(1-\lambda)z(x)+\lambda\bar z_{\mathcal N}(x),
$$

and

$$
z_{\mathrm{augment}}(x)
=
\left[
z(x),
\lambda\bar z_{\mathcal N}(x),
\lambda(z(x)-\bar z_{\mathcal N}(x)),
\overline d_{\mathcal N}(x)
\right].
$$

The same ridge propensity model then scores the twelve programs and ranks them in descending order. No peer outcomes or post-hoc ranker are used.

## Results

On the initial ten splits, validation-tuned smoothing reduced mean regret from 0.038726 to 0.037968 and improved hit rate by 0.94 percentage points. The paired regret interval crossed zero, and selected neighborhood sizes varied from 10 to 200.

The result did not replicate on ten fresh splits:

| Method | Initial regret | Fresh regret |
|---|---:|---:|
| Base embedding | 0.038726 | **0.036119** |
| Tuned smoothed embedding | **0.037968** | 0.036459 |
| Tuned augmented embedding | 0.038689 | 0.037131 |
| Fixed smoothed embedding | 0.038904 | 0.036210 |
| Fixed augmented embedding | 0.038884 | 0.036304 |

Across all twenty splits, tuned smoothing changes regret by

$$
-0.000209,
\qquad
95\%\text{ paired interval }[-0.001088,0.000670],
$$

while increasing RMSE by 0.001888 with an interval entirely above zero. Its small average ranking gain is therefore unstable and comes with a clear loss of probability accuracy.

The fixed augmented embedding slightly improves RMSE across twenty splits:

$$
\Delta_{\mathrm{RMSE}}=-0.000139,
\qquad
95\%\text{ paired interval }[-0.000249,-0.000030],
$$

but raises regret by 0.000171. This repeats the broader pattern: neighborhood information can denoise average propensity estimates without improving the identity of the winning program.

## Interpretation

Simple message passing is not enough. Averaging nearby SVD coordinates removes some idiosyncratic noise, but it also removes customer-specific behavioral directions that determine the top action. Appending the neighborhood prototype preserves the original embedding and slightly helps RMSE, yet the linear propensity model does not extract a ranking benefit from it.

For neighborhoods to improve the embedding materially, the neighbor weights or representation objective would need to be supervised by future propensity similarity rather than cosine similarity alone. That would be a different experiment—metric learning or graph-regularized representation learning—not another fixed neighborhood average.

