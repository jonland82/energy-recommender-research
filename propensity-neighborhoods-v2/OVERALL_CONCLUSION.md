# Overall conclusion across the propensity-neighborhood experiments

## Best supported result

The strongest and most stable empirical result is the behavioral representation itself:

- build a behavioral embedding from information available before recommendation;
- combine it with ordinary customer attributes;
- train a propensity model for every eligible program; and
- rank programs directly by fitted propensity.

Relative to the static-feature model, the prespecified behavioral-embedding model:

- lowers propensity RMSE by 31.9%;
- lowers top-1 regret by 6.4%;
- raises exact top-1 hit rate by 2.58 percentage points;
- improves RMSE on all ten chronological splits;
- improves regret on eight of ten splits; and
- improves hit rate on nine of ten splits.

This is the best supported recommendation methodology in the repository.

## Best supported workflow

1. Filter the catalog to programs for which the customer is eligible.
2. Construct a behavioral embedding from pre-decision history.
3. Fit one propensity score for each eligible program using the embedding and static attributes.
4. Rank the eligible programs directly by fitted propensity.
5. Use neighborhoods as an optional secondary layer for calibration, diagnostics, confidence, fallback, or abstention.
6. Do not let the neighborhood change the winning recommendation unless a future experiment supplies stable out-of-sample evidence for that decision rule.

## What neighborhoods reliably contribute

Neighborhoods have demonstrated narrower forms of value:

- operational peer-mean calibration lowers RMSE by approximately 0.000310;
- fixed full-row propensity neighborhoods lower RMSE by approximately 0.000351 on fresh confirmation splits;
- fixed neighborhood augmentation of embeddings lowers RMSE by approximately 0.000139 across twenty splits;
- propensity-neighborhood features lower pairwise ranking log loss relative to an otherwise identical base-score ranker on all twenty tested splits; and
- nested neighborhoods expose the practical loss of peer support as matching becomes more specific.

These findings show that neighborhoods can contain real calibration and ordering information. They do not show that the information is strong enough at the top-choice boundary to improve the selected program.

## What did not produce a confirmed ranking improvement

The following approaches were tested without a reproducible improvement in top-1 regret:

- raw peer averaging;
- shrunk residual correction;
- program-specific neighborhoods;
- full-row and leave-one-program-out propensity neighborhoods;
- learned pairwise and top-focused rankers;
- theorem-restricted or plausible-set reranking;
- neighborhood preprocessing before the final propensity model;
- top-two residual-margin correction;
- uncertainty-gated top-two swaps;
- smoothing behavioral embeddings toward their neighborhood mean; and
- augmenting embeddings with their neighborhood prototype and deviation.

Several variants looked favorable on exploratory splits and then failed on fresh splits. They should not be presented as ranking improvements.

## Why calibration gains do not become ranking gains

Top-1 selection depends on a small number of close decision boundaries. A method can improve many propensity coordinates or pairwise comparisons without correcting the particular comparison that determines the winner.

The decision-boundary experiment confirms that candidate availability is not the main limitation:

- the true winner appears in the base top two for approximately 67% of users;
- it appears in the base top three for approximately 83%; and
- a perfect selector within those candidate sets would have much lower regret.

The problem is that retrieved peers do not reliably predict when the base top-two order is wrong. Only about half of the tested neighborhood-triggered swaps are beneficial.

## Role of the theorem

The refinement-frontier theorem remains an important methodological result. It states the correct tradeoff:

- refining a fixed-radius neighborhood can improve worst-case local similarity;
- refinement can only preserve or reduce peer support; and
- the uncertainty cost rises as support falls.

Its practical role is to govern how much local evidence the system should trust. It supports:

- selection of neighborhood depth;
- shrinkage toward the base model;
- confidence and regret bounds under the stated assumptions;
- retention of the base recommendation when local support is inadequate; and
- abstention or low-confidence flags when the winning margin is not certified.

The theorem does not assert that useful residual similarity exists in every dataset, nor that neighborhood refinement must improve rankings. The experiments supply that missing empirical condition: a neighborhood can improve decisions only when its geometry predicts the remaining error at the relevant decision boundary.

## Best paper supported by the evidence

The strongest paper remains **Behavioral Embeddings and Propensity Neighborhoods for Program Recommendation**.

Its best-supported central claim is:

> Behavioral embeddings create the main predictive and ranking improvement. Propensity neighborhoods provide a smaller calibration and reliability layer. The refinement-frontier theorem explains when increasingly specific local evidence is worth the peer support it consumes and when the recommender should retain the direct propensity ranking or abstain.

The paper should not claim that neighborhoods improve the winning recommendation on KuaiRec. The accumulated experiments instead strengthen the distinction between better probability estimation, additional pairwise information, and better top-1 decisions.
