# Propensity Neighborhoods v2: Certified Neighborhood Reranking

## Current overall conclusion

The follow-up experiments do not establish a neighborhood-driven improvement in the winning recommendation. The strongest empirical method remains behavioral embedding $\rightarrow$ propensity model $\rightarrow$ direct descending ranking. Neighborhoods retain value for small calibration improvements, diagnostics, support-aware confidence, fallback, and abstention. See `OVERALL_CONCLUSION.md` for the consolidated result across the original paper and all v2 experiments.

## Starting hypothesis

The propensity-neighborhood method should be tested as a source of ranking information, not only as a peer-mean calibration adjustment. The current method estimates

$$
\widehat p_{a,m}(x)
=
\widetilde p_a(x)
+
\frac{1}{n_m}\sum_{i\in\mathcal N_m(x)}
\left(Y_{ia}-\widetilde p_a(x_i)\right)
$$

and recommends the program with the largest corrected coordinate. That procedure trains each coordinate for probability accuracy even though the operational objective is to choose the best program. Better coordinatewise calibration therefore need not improve the winning recommendation.

The v2 hypothesis is that a ranker can learn from the *structure* of a customer's neighborhood: not only its mean residual, but also local dispersion, local winning frequency, local margins, distance, support, and how these quantities change across neighborhood scales.

For each customer-program pair, construct a feature vector such as

$$
\phi_{a,m}(x)=
\left[
\widetilde p_a(x),
\overline R_{a,m}(x),
\operatorname{sd}(R_{ia}),
\Pr_{i\in\mathcal N_m(x)}\!\left(a=\arg\max_bY_{ib}\right),
\overline{Y_{ia}-\max_{b\ne a}Y_{ib}},
n_m,
\text{distance summaries}
\right].
$$

Features from several nested or nearest-neighbor scales can be supplied together. A learned scoring function $s_\theta(a,x)$ then produces the final ordering.

## Ranking objective

The initial experiment uses a regularized linear pairwise logistic ranker. For programs $a$ and $b$, its loss is weighted by the observed future propensity gap:

$$
\mathcal L(\theta)
=
\sum_x\sum_{a<b}
|Y_{xa}-Y_{xb}|
\log\left(
1+
\exp\left\{
-\operatorname{sign}(Y_{xa}-Y_{xb})
[s_\theta(a,x)-s_\theta(b,x)]
\right\}
\right).
$$

The weighting makes inversions between materially different programs more costly than inversions between nearly tied programs. The first test deliberately uses a low-capacity ranker because the number of independent users is much smaller than the expanded number of within-user program pairs.

## How the existing theorem remains central

Let $a_\theta(x)$ be the program selected by any learned ranker and define its departure from the neighborhood-calibrated maximizer as

$$
\xi_\theta(x)
=
\max_a\widehat p_{a,m}(x)
-
\widehat p_{a_\theta(x),m}(x).
$$

On the same event used by the existing refinement-frontier theorem,

$$
p_{a^*}(x)-p_{a_\theta(x)}(x)
\le
2b_m(x)+\xi_\theta(x).
$$

The proof is immediate:

$$
\begin{aligned}
p_{a^*}(x)-p_{a_\theta}(x)
&\le
\max_a\widehat p_{a,m}(x)+b_m(x)
-\widehat p_{a_\theta,m}(x)+b_m(x)\\
&=
2b_m(x)+\xi_\theta(x).
\end{aligned}
$$

Thus $2b_m(x)$ is the neighborhood match-versus-support uncertainty, while $\xi_\theta(x)$ is the explicit price of departing from the corrected propensity ordering. Ordinary descending sorting is the special case $\xi_\theta(x)=0$.

A constrained version defines the plausible candidate set

$$
\mathcal C_m(x;\tau)
=
\left\{
a:
\max_b\widehat p_{b,m}(x)-\widehat p_{a,m}(x)
\le\tau
\right\}.
$$

If the ranker chooses only from $\mathcal C_m(x;\tau)$, then

$$
p_{a^*}(x)-p_{a_\theta(x)}(x)
\le 2b_m(x)+\tau.
$$

When local evidence gives one program a decisive lead, the candidate set is a singleton. When several programs remain close, the learned ranker can use richer neighborhood information to break the tie.

## Information supplied by a neighborhood

The existing theorem is a finite-sample concentration and regret result; it is not itself an information-theoretic result. The incremental ranking value of the neighborhood can nevertheless be expressed as

$$
I\!\left(A^*;\Phi_{\mathcal N}\mid\widetilde p\right),
$$

where $A^*$ is the best program and $\Phi_{\mathcal N}$ denotes the neighborhood summaries. Out-of-sample improvement in ranking log loss from adding $\Phi_{\mathcal N}$ to an otherwise identical base ranker provides an empirical test of whether this conditional information is present. Improvement in RMSE alone does not answer that question.

## Initial experimental comparison

The first chronological KuaiRec experiment compares:

1. direct ordering of the base propensity row;
2. the existing fixed full-row propensity-neighbor residual correction;
3. a pairwise ranker using base-score features only;
4. a pairwise ranker using base scores plus multiscale neighborhood features; and
5. a restricted neighborhood ranker that can override the corrected ordering only inside $\mathcal C_m(x;\tau)$.

The primary endpoint is top-1 regret. Secondary endpoints are hit rate, NDCG, RMSE where a probability vector exists, pairwise ranking log loss, the selected candidate-set threshold, candidate-set size, override rate, and the empirical departure $\xi_\theta(x)$.

All future outcomes for test users remain untouched until final evaluation. Behavioral embeddings and propensity models are estimated from the estimation users; cross-fitted estimation predictions define reference propensity rows and residuals; estimation labels train the rankers; tuning users select regularization and the restriction threshold; and test users provide reported results only.

## Files

- `OVERALL_CONCLUSION.md`: authoritative synthesis and recommended methodology.
- `experiment.py`: leakage-safe repeated-split experiment.
- `RESULTS.md`: first-round findings and interpretation.
- `PREPROCESSING_RESULTS.md`: neighborhood-before-ranking experiment and findings.
- `preprocessing_experiment.py`: cross-fitted neighborhood feature preprocessing.
- `DECISION_BOUNDARY_RESULTS.md`: top-two opportunity and local-margin results.
- `decision_boundary_experiment.py`: decision-specific neighborhood correction.
- `EMBEDDING_NEIGHBORHOOD_RESULTS.md`: neighborhood smoothing/augmentation test.
- `embedding_neighborhood_experiment.py`: pre-decision embedding experiment.
- `results/`: initial ten-split experiment and geometry ablations.
- `confirmation-results/`: frozen-specification results on ten fresh splits.
- `top-focused-results/`: exploratory top-focused loss results.
- `candidate-conditioned-results/`: exploratory plausible-set training results.

Run the initial or fresh ten-split sets from the repository root with:

```powershell
python propensity-neighborhoods-v2\experiment.py --seeds 20260902:20260911 --output propensity-neighborhoods-v2\results
python propensity-neighborhoods-v2\experiment.py --seeds 20260912:20260921 --output propensity-neighborhoods-v2\confirmation-results
```

Run the preprocessing experiment with:

```powershell
python propensity-neighborhoods-v2\preprocessing_experiment.py --seeds 20260902:20260911 --output propensity-neighborhoods-v2\preprocessing-results
python propensity-neighborhoods-v2\preprocessing_experiment.py --seeds 20260912:20260921 --output propensity-neighborhoods-v2\preprocessing-confirmation-results
```

The decision-boundary experiment is reproduced with:

```powershell
python propensity-neighborhoods-v2\decision_boundary_experiment.py --seeds 20260902:20260911 --output propensity-neighborhoods-v2\decision-gated-results
```

The embedding experiment is reproduced with:

```powershell
python propensity-neighborhoods-v2\embedding_neighborhood_experiment.py --seeds 20260902:20260911 --output propensity-neighborhoods-v2\embedding-neighborhood-results
python propensity-neighborhoods-v2\embedding_neighborhood_experiment.py --seeds 20260912:20260921 --output propensity-neighborhoods-v2\embedding-neighborhood-confirmation-results
```
