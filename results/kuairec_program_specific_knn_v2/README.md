# Program-specific k-neighbor propensity calibration

## Result

The proposed improvement did not work on KuaiRec. After cross-fitting and leakage-safe tuning, program-specific neighborhoods increased mean top-1 regret on every one of ten repeated splits. The experiment indicates that the available customer features do not contain stable local structure in the base model's remaining propensity errors.

## Design

- Outcome: propensity to complete a video in each of the 12 largest first-tag categories.
- Outer split: 60% estimation users, 25% tuning users, and 15% untouched test users.
- Base ranker: multi-output ridge regression.
- Reference residuals: five-fold out-of-fold predictions for all estimation users.
- Program geometry: one nonnegative weighting of four customer-feature blocks per category.
- Geometry learning: leave-one-out neighbor prediction of the cross-fitted residuals within the estimation pool.
- Program tuning: category-specific `k` and residual shrinkage selected on the tuning users by propensity MSE.
- Shared baseline tuning: one geometry, `k`, and shrinkage selected on the tuning users by top-1 regret.
- Evaluation: ten repeated outer splits, totaling 2,130 untouched test-user decisions.

No test outcome is used to fit the propensity model, estimate peer residuals, learn feature weights, or select hyperparameters.

## Ten-split test results

| Method | RMSE | Top-1 regret | Top-1 hit rate | Regret wins vs. base |
|---|---:|---:|---:|---:|
| Base ridge | **0.108265** | **0.018971** | **49.44%** | n/a |
| Prespecified broad kNN | 0.108423 | 0.019151 | 49.39% | 2 of 10 |
| Validation-tuned shared kNN | 0.108456 | 0.019214 | 49.11% | 3 of 10 |
| Program-specific kNN | 0.108454 | 0.019667 | 48.87% | 0 of 10 |

Relative to the base ranker, the program-specific method increased regret by 0.000696, or 3.7%, and reduced hit rate by 0.56 percentage points. Its regret was worse on all ten splits. The prespecified broad correction was closer to neutral but still increased mean regret by 0.9%.

## Diagnostic finding

For every one of the 120 category-by-split metric fits, none of the individual feature blocks improved leave-one-out prediction of the cross-fitted base residual. The metric learner therefore invoked its regularized fallback and assigned equal weights to the four blocks in all 120 cases.

This is the main finding. KuaiRec's user attributes help estimate propensity, but after the base model uses them, they do not reliably identify nearby users with similar remaining errors. Program-specific neighborhood calibration consequently has no stable residual signal to exploit. The validation-selected shared configuration was also unstable, with nine different choices across ten splits, which is consistent with tuning noise rather than a persistent best neighborhood.

## Interpretation for the geometry note

The geometric result remains correct: additional coordinates thin neighborhoods, and support-aware refinement guards against using small neighborhoods. The result does not imply that a useful local propensity signal exists. This experiment supplies the missing empirical condition: refinement can help only when closeness in the chosen coordinates predicts closeness in propensity or propensity residuals.

The earlier five-split pilot found a small benefit from a fixed shrunk correction. The stricter ten-split experiment does not confirm that benefit. The appropriate current conclusion is therefore that the neighborhood method is geometrically sound but has not demonstrated a reliable recommendation improvement on KuaiRec.

A subsequent chronological experiment using learned behavioral embeddings is reported in `results/kuairec_behavior_embeddings`. Those coordinates materially improve future propensity ranking, although the neighborhood layer adds little ranking value beyond the embedding model itself.

## Reproduction

From the repository root:

```powershell
python scripts\kuairec_program_specific_knn.py --seed 20260902 --output results\kuairec_program_specific_knn_v2\seed_20260902
```

Each run takes approximately six seconds locally. Per-split metrics and full program selections are stored in the `seed_*` directories.
