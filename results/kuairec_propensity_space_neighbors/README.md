# Propensity-space neighborhood experiment

## Question

Can the fitted propensity matrix itself define useful customer neighborhoods? For each target customer, this experiment compares their fitted propensity row with out-of-fold propensity rows for historical customers, then uses neighbor residuals to recalibrate and rerank programs.

Two constructions are tested:

1. Full-row neighborhoods use every fitted program propensity.
2. Leave-one-program-out neighborhoods exclude program (a) when calibrating its propensity.

The experiment retains the chronological design: early interactions create behavioral embeddings, later interactions define future propensities, 25% of users tune the neighborhood, and 15% remain untouched for testing.

## Exploratory ten-split results

| Method | RMSE | Top-1 regret | Top-1 hit rate |
|---|---:|---:|---:|
| Static features | 0.126761 | 0.041357 | 34.46% |
| Static features plus behavioral embedding | 0.086310 | 0.038726 | **37.04%** |
| Embedding-space shared neighbors | **0.085960** | 0.038530 | **37.09%** |
| Full propensity-row neighbors | 0.086176 | **0.038369** | 36.10% |
| Leave-one-program-out propensity neighbors | 0.086272 | 0.039423 | 36.48% |

Full propensity-row matching has the lowest mean regret in this exploratory set, but its increment over the direct embedding propensity model is small and inconsistent. It beats that model on four of ten splits. Leave-one-program-out matching performs worse, indicating that the omitted program coordinate contains important retrieval information.

The maximum-coordinate distance used in the note, implemented as Chebyshev distance, was selected for the full propensity row in eight of ten splits. This motivated a fixed confirmation rule using Chebyshev distance, 50 neighbors, and residual shrinkage 20.

## Fresh ten-split confirmation

The rule was frozen and evaluated on ten new random user splits.

| Method | RMSE | Top-1 regret | Top-1 hit rate |
|---|---:|---:|---:|
| Static features | 0.128234 | 0.039466 | 36.90% |
| Static features plus behavioral embedding | 0.083850 | **0.036119** | **39.30%** |
| Embedding-space shared neighbors | 0.084097 | 0.037256 | 38.40% |
| Fixed full-row propensity neighbors | **0.083498** | 0.036931 | 38.83% |
| Leave-one-program-out propensity neighbors | 0.083833 | 0.037950 | 37.51% |

The fixed propensity-space rule improves RMSE over the direct embedding model by 0.000351 on nine of ten splits. It worsens top-1 regret by 0.000812, or 2.2%, and beats the direct embedding model on only two of ten splits. The paired 95% interval for the regret change is (+0.000015) to (+0.001609), so the ranking loss is not explained by a single outlying split.

Leave-one-program-out matching is less successful. Relative to the embedding model, it raises regret by 0.001831 and lowers hit rate by 1.78 percentage points; it does not improve hit rate on any confirmation split.

## Conclusion

Propensity-space neighborhoods are a valid and inexpensive calibration layer. They compress the 32-coordinate behavioral representation to a 12-coordinate score row and slightly improve probability RMSE. They do not improve program ranking in the confirmation experiment.

The practical recommendation remains to train the propensity model with behavioral embeddings and rank its output directly. The propensity matrix can support diagnostics, cohort retrieval, and calibration, but the current evidence does not support residual reranking from propensity-space neighbors.

For the theorem, the experiment supplies a useful distinction: improved local calibration does not necessarily improve the identity of the top-ranked action. A ranking application needs a margin-aware decision rule, not only a bound on coordinatewise propensity error.

## Reproduction

```powershell
python scripts\kuairec_behavior_embedding_test.py --seed 20260902 --output results\kuairec_propensity_space_neighbors\seed_20260902
```

Exploratory runs are in this directory. The frozen-rule runs are in the adjacent `kuairec_propensity_space_confirmation` directory.
