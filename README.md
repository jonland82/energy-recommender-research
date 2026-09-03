# Energy Recommender Research

This private repository contains the propensity-neighborhood theory, recommender implementation, and KuaiRec experiments developed for energy-program recommendation research.

## Structure

- `papers/dual-geometry/`: the original dual-geometry note and compiled PDF.
- `papers/propensity-neighborhoods/`: the current paper, including behavioral embeddings, the propensity matrix, the residual-refinement theorem, workflow, plots, and experimental findings.
- `papers/propensity-neighborhoods-accessible/`: a shorter, less dense presentation of the same paper and formal results, designed for easier first reading.
- `propensity-neighborhoods-v2/`: follow-up ranking, preprocessing, decision-boundary, and embedding-neighborhood experiments, with an overall evidence synthesis.
- `scripts/`: reproducible KuaiRec propensity and neighborhood experiments.
- `results/`: per-split metrics, selected hyperparameters, summaries, and experiment reports.
- `data/`: acquisition and local-layout instructions. Raw public data are kept outside Git history.

## Current finding

Behavioral embeddings provide the main practical gain in the chronological KuaiRec study: relative to static features, they reduce propensity RMSE by 31.9%, reduce top-1 regret by 6.4%, and raise hit rate by 2.58 percentage points. The best supported implementation trains the propensity model with behavioral embeddings and ranks eligible programs directly by fitted propensity.

Neighborhoods provide smaller calibration, diagnostic, and uncertainty benefits. Across the original and v2 experiments, no residual correction, learned ranker, preprocessing layer, decision-boundary rule, or embedding-smoothing method produces a confirmed improvement in the winning recommendation. The refinement theorem remains useful for expressing the match-versus-support tradeoff and governing calibration, confidence, fallback, or abstention. See `propensity-neighborhoods-v2/OVERALL_CONCLUSION.md` for the consolidated evidence and recommended workflow.

## Reproduce

Create an environment and install the recorded dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

With KuaiRec placed in `data/kuairec/extracted`, run the main chronological experiment:

```powershell
python scripts\kuairec_behavior_embedding_test.py
```

The default scripts write machine-readable outputs below `results/`.

Run the leakage-free operational neighborhood diagnostic and its radius sensitivities with:

```powershell
python scripts\kuairec_operational_frontier.py --output results\kuairec_operational_frontier
python scripts\kuairec_operational_frontier.py --radius-quantile 0.05 --output results\kuairec_operational_frontier_q05
python scripts\kuairec_operational_frontier.py --radius-quantile 0.25 --output results\kuairec_operational_frontier_q25
```

The primary operational report is `results/kuairec_operational_frontier/README.md`.

## Papers

The current paper is `papers/propensity-neighborhoods/propensity_neighborhood_note.pdf`. Its figures are generated directly by the accompanying LaTeX source.

The accessible five-page version is `papers/propensity-neighborhoods-accessible/propensity_neighborhood_accessible.pdf`. It preserves the theorem, propositions, corollaries, and proofs while simplifying the narrative and notation load.

The current paper remains the strongest paper supported by the experiments. Its defensible central story is that behavioral embeddings create the main ranking gain, while propensity neighborhoods provide a secondary calibration and reliability layer rather than a demonstrated ranking improvement.
