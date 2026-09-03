# KuaiRec propensity-neighborhood pilot

> **Status:** This is an exploratory pilot. Its apparent ranking gain did not survive the stricter chronological confirmation and v2 follow-up experiments. The consolidated conclusion is in `propensity-neighborhoods-v2/OVERALL_CONCLUSION.md` from the repository root.

## Question

Does progressively refining a user neighborhood improve recommendation, and does the paper's support-aware rule stop before the neighborhood becomes too small?

## Design

- Data: KuaiRec 2.0 dense `small_matrix.csv`.
- Scale: 1,411 users, 4,676,570 input interactions, and 2,632 analyzed videos.
- Actions: the 12 most prevalent first-tag video categories.
- Outcome: the fraction of videos in a category that a user watched to completion (`watch_ratio >= 1`).
- Split: 60% model-training users, 20% reference peers, and 20% held-out target users.
- Base ranker: multi-output ridge regression over user features.
- Neighborhoods: four cumulative feature blocks covering activity and tenure, creator status, social counts, and encrypted categorical user attributes.
- Selection: minimize estimated propensity mismatch plus the paper's finite-support term, with `alpha=0.10`.
- Robustness: five fixed random splits; three additional splits at each of two alternative radius quantiles.

The user splits are disjoint. Held-out target outcomes are not used to train the base ranker, construct peer outcomes, or select refinement depth.

## Five-split results

| Method | RMSE | Top-1 regret | Top-1 hit rate | Coverage | Mean peers |
|---|---:|---:|---:|---:|---:|
| Base ridge ranker | 0.1090 | 0.01941 | 48.69% | 100% | n/a |
| Fixed level 1 | 0.1104 | 0.01784 | 50.81% | 100% | 260.0 |
| Fixed level 2 | 0.1103 | 0.01797 | 50.60% | 100% | 236.6 |
| Fixed level 3 with base fallback | 0.1124 | 0.01863 | 49.19% | 98.23% | 130.8 |
| Fixed level 4 with base fallback | 0.1156 | 0.01961 | 47.99% | 90.11% | 37.6 |
| Adaptive theorem rule | 0.1104 | 0.01784 | 50.81% | 100% | 245.6 |

For a fair comparison, a fixed level with no peers falls back to the base ranker, and every method is scored on all target users. Relative to the base ranker, the adaptive rule reduced mean top-1 propensity regret by 8.1% and increased the exact top-1 hit rate by 2.12 percentage points. Its probability RMSE was 1.2% worse. Thus the neighborhood layer helped the ranking decision but did not improve probability estimation.

Across 1,415 held-out target decisions, the adaptive rule chose level 1 for 71.1%, level 2 for 17.7%, level 3 for 11.2%, and level 4 for none. Its ranking results were effectively identical to the broad level-1 neighborhood. The experiment therefore supports stopping refinement early, but it does not yet show an advantage over choosing the best shallow level globally.

## Certificate diagnostic

The theoretical certificate covered the realized top-1 regret in 100% of cases, but its mean value was 1.15 on a response scale bounded by one. Only 14.1% of certificates were below one. The bound is therefore valid in this pilot but usually too loose to be operationally informative.

## Interpretation

The first experiment confirms the basic mechanism:

1. Adding feature blocks reduces peer support.
2. The deepest neighborhood has lower coverage and worse held-out performance.
3. The support-aware rule avoids the deepest refinement.
4. Neighborhood averaging can improve the selected action while degrading probability calibration.

The adaptive rule did not beat the best broad fixed neighborhood, and the formal certificate was mostly vacuous. Follow-up experiments with residual calibration, shrinkage, mismatch quantiles, alternate outcomes, and reversed feature order are reported in the adjacent `kuairec_propensity_extensions` directory.

## Reproduction

From the repository root:

```powershell
python scripts\kuairec_propensity_neighborhood_pilot.py
```

Per-run computation is approximately four seconds after the data have been extracted. Individual split outputs are stored in the `seed_*` directories.
