# KuaiRec propensity-neighborhood extensions

> **Status:** These are exploratory extensions of the first pilot. Later chronological confirmation and v2 experiments do not establish a neighborhood-driven ranking improvement. The consolidated conclusion is in `propensity-neighborhoods-v2/OVERALL_CONCLUSION.md` from the repository root.

## Purpose

These experiments ask whether the first pilot's ranking improvement survives a more conservative recommender implementation. Instead of replacing the base propensity vector with a peer mean, the neighborhood estimates a residual correction to the base ranker. The correction can be shrunk toward zero when peer support is limited.

All results are means over the same five disjoint user splits. Fixed levels with no peers fall back to the base prediction, so every method is scored on every target user. Each run takes about four seconds locally after data extraction.

## Main comparison

The standard outcome is category-level propensity to watch a video to completion, defined by `watch_ratio >= 1`.

| Adaptive calibration | RMSE | Top-1 regret | Top-1 hit rate | Regret change vs. base | Hit-rate change vs. base |
|---|---:|---:|---:|---:|---:|
| Base ridge ranker | 0.109017 | 0.019411 | 48.69% | n/a | n/a |
| Unshrunk residual | 0.109023 | 0.019152 | 49.05% | -1.3% | +0.35 pp |
| Residual, shrinkage 20 | 0.108990 | 0.019122 | 49.19% | -1.5% | +0.49 pp |
| Residual, shrinkage 50 | **0.108982** | **0.019044** | **49.33%** | **-1.9%** | **+0.64 pp** |
| Shrinkage 20, 90th-percentile mismatch | 0.109009 | 0.019071 | 49.19% | -1.7% | +0.49 pp |
| Shrinkage 20, 75th-percentile mismatch | 0.108999 | 0.019071 | 49.19% | -1.7% | +0.49 pp |

Shrinkage 50 is the cleanest compromise in this set. It improves mean RMSE slightly and reduces top-1 regret on all five splits. The absolute effects are small, as expected for a correction layered over an already fitted ranker.

The maximum-mismatch selector chose levels 1, 2, 3, and 4 on 71.1%, 17.7%, 11.2%, and 0% of the 1,415 held-out decisions. Replacing the maximum with the 90th percentile made it still shallower: 90.5% level 1 and 9.5% level 2. The 75th percentile chose level 1 in 97.7% of cases. Softer mismatch estimates therefore did not unlock useful deep refinement in this sample.

## Robustness checks

Using shrinkage 50, the experiment was repeated for two alternate binary outcomes and for the reverse order of the four feature blocks.

| Check | Base regret | Adaptive regret | Base hit rate | Adaptive hit rate | Adaptive RMSE |
|---|---:|---:|---:|---:|---:|
| Half-watch, `watch_ratio >= 0.5` | 0.012514 | 0.012354 | 59.65% | 59.65% | 0.108172 |
| Double-watch, `watch_ratio >= 2` | 0.017335 | 0.017338 | 26.08% | 26.50% | 0.046470 |
| Reversed feature order, completion outcome | 0.019411 | 0.018980 | 48.69% | 49.40% | 0.109096 |

The half-watch result retains a small regret improvement but no mean hit-rate improvement. The rare double-watch outcome, whose positive rate is 4.84%, is essentially neutral on regret. Reversing the feature order retains a similar ranking gain, even though RMSE becomes slightly worse. This suggests that the early-stopping effect is not tied to the original block order, while its practical value depends on the outcome being recommended.

## What the experiments say

Three findings are consistent across the runs:

1. Adding coordinates rapidly reduces peer support, and the selector almost never uses the deepest neighborhood.
2. Raw peer averaging can improve ranking more, but worsens propensity calibration. A shrunk residual correction is more balanced.
3. The theorem-inspired rule behaves as a conservative early-stopping rule. It has not yet shown that customer-specific depth selection beats the best shallow depth chosen globally.

The certificate remains loose for the standard completion outcome: its mean is 1.15 on a response scale bounded by one, and 14.1% of certificates are below one. Its practical contribution here is the structure of the support penalty and the resulting stopping behavior, not a tight numerical guarantee.

With only five splits, these are pilot estimates rather than precise effect sizes. The most useful next test is a larger repeated-split study with a validation set that chooses shrinkage and compares the adaptive rule against a globally tuned fixed level without using test outcomes.

That next test has now been completed in `results/kuairec_program_specific_knn_v2`. Its stricter ten-split evaluation did not reproduce the pilot improvement, so the pilot result should be treated as exploratory.

## Reproduction

The script accepts `--calibration`, `--shrinkage`, `--mismatch-quantile`, `--completion-threshold`, and `--block-order`. For the leading standard configuration:

```powershell
python scripts\kuairec_propensity_neighborhood_pilot.py --calibration shrunk_residual --shrinkage 50
```

Machine-readable metrics and run summaries are stored in each configuration-and-seed subdirectory.
