# Behavioral embeddings and propensity neighborhoods

## Result

Learned behavioral coordinates produce a clear recommendation improvement. The additional neighborhood correction improves probability calibration slightly, but it does not yet provide a reliable ranking improvement beyond the embedding model itself.

## Leakage-safe design

Each user's interactions are ordered by time. The first 60% form the observable history and the final 40% define future category propensities. A 32-coordinate truncated-SVD representation is learned from the historical user-video completion matrix using estimation users only, then applied to tuning and test histories.

For each of ten repeated user splits:

- 60% of users estimate the propensity models and cross-fitted peer residuals.
- 25% tune distance type, neighbor count, and shrinkage.
- 15% remain untouched until final evaluation.
- The base model uses static customer attributes.
- The embedding model adds the 32 learned historical-behavior coordinates.
- The neighbor models use Euclidean or cosine distance in embedding space.

Approximately 2.22 million early interactions construct the representations and 1.48 million later interactions define the outcomes. No future test interaction contributes to a representation, fitted model, neighbor choice, or hyperparameter selection.

## Prespecified 32-coordinate experiment

| Method | RMSE | Top-1 regret | Top-1 hit rate |
|---|---:|---:|---:|
| Static customer features | 0.126761 | 0.041357 | 34.46% |
| Static features plus SVD embedding | 0.086310 | 0.038726 | 37.04% |
| Embedding plus shared kNN correction | 0.085960 | **0.038530** | **37.09%** |
| Embedding plus program-specific kNN | **0.085564** | 0.039276 | 36.48% |

The embedding model reduces RMSE by 31.9%, reduces top-1 regret by 6.4%, and raises hit rate by 2.58 percentage points relative to static customer features. RMSE improves on all ten splits, regret improves on eight, and hit rate improves on nine.

The shared neighbor method reduces regret by 6.8% relative to the static model and does so on all ten splits. Almost all of that gain comes from the embedding representation, however. Relative to the embedding model without neighbors, the shared correction changes mean regret by only -0.000196 and wins on four of ten splits. The program-specific correction improves RMSE relative to the embedding model but worsens mean ranking regret.

## Dimension sensitivity and confirmation

Dimensions 8, 16, 32, and 64 were compared on the first five splits. Sixteen coordinates made program-specific neighbors look most promising, so that configuration was extended to five additional confirmation splits.

On those five new splits, adding program-specific neighbors to the 16-coordinate embedding:

- improved RMSE from 0.085967 to 0.085545;
- worsened regret from 0.038327 to 0.038757;
- reduced hit rate from 37.37% to 36.71%.

Thus the probability-calibration increment survives, while the apparent ranking increment does not. This confirmation check prevents selecting the favorable first-five result after trying several embedding dimensions.

## Interpretation

The missing ingredient in the static-feature experiment was a geometry that actually represents customer preference. Historical behavior supplies that geometry, and it materially improves future propensity prediction and recommendation.

The theory continues to describe the refinement tradeoff correctly: adding coordinates changes which customers remain close and reduces effective support. The experiments add an empirical prerequisite. A useful coordinate system must preserve future propensity similarity before neighborhood refinement can help.

For this dataset, the practical implementation should use behavioral embeddings directly in the propensity model. A neighborhood residual layer can be retained when improved probability calibration is valuable, but the evidence does not support claiming an additional recommendation-ranking gain from that layer.

A subsequent experiment using the fitted propensity rows as the neighborhood geometry is reported in `results/kuairec_propensity_space_neighbors`. It also improves calibration slightly but does not improve ranking in a fresh fixed-rule confirmation.

## Reproduction

```powershell
python scripts\kuairec_behavior_embedding_test.py --seed 20260902 --output results\kuairec_behavior_embeddings\seed_20260902
```

Each 32-coordinate run takes approximately nine seconds locally. Full metrics, selected program hyperparameters, and residual diagnostics are stored in the `seed_*` directories. Dimension-sensitivity runs are in the adjacent `kuairec_behavior_embedding_sensitivity` directory.
