# Energy Recommender Research

This private repository contains the propensity-neighborhood theory, recommender implementation, and KuaiRec experiments developed for energy-program recommendation research.

## Structure

- `papers/dual-geometry/`: the original dual-geometry note and compiled PDF.
- `papers/propensity-neighborhoods/`: the current paper, including behavioral embeddings, the propensity matrix, the residual-refinement theorem, workflow, plots, and experimental findings.
- `scripts/`: reproducible KuaiRec propensity and neighborhood experiments.
- `results/`: per-split metrics, selected hyperparameters, summaries, and experiment reports.
- `data/`: acquisition and local-layout instructions. Raw public data are kept outside Git history.

## Current finding

Behavioral embeddings provide the main practical gain in the chronological KuaiRec study. Operational peer-mean neighborhoods add a small, consistent improvement in propensity RMSE, but no detectable improvement in the winning recommendation. The refinement theorem supplies the general support tradeoff; its operational corollaries distinguish latent-propensity guarantees under a valid residual band from marginal realized-outcome guarantees under exchangeability.

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
