# Energy Recommender Research

This private repository contains the propensity-neighborhood theory, recommender implementation, and KuaiRec experiments developed for energy-program recommendation research.

## Structure

- `papers/dual-geometry/`: the original dual-geometry note and compiled PDF.
- `papers/propensity-neighborhoods/`: the current paper, including behavioral embeddings, the propensity matrix, the residual-refinement theorem, workflow, plots, and experimental findings.
- `scripts/`: reproducible KuaiRec propensity and neighborhood experiments.
- `results/`: per-split metrics, selected hyperparameters, summaries, and experiment reports.
- `data/`: acquisition and local-layout instructions. Raw public data are kept outside Git history.

## Current finding

Behavioral embeddings provide the main practical gain in the chronological KuaiRec study. They improve future propensity estimation and recommendation ranking. Nested neighborhoods provide a principled support tradeoff, modest local calibration gains, a computationally staged retrieval scheme, and a margin-aware confidence gate for deciding when local evidence should affect a recommendation.

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

## Papers

The current paper is `papers/propensity-neighborhoods/propensity_neighborhood_note.pdf`. Its figures are generated directly by the accompanying LaTeX source.
