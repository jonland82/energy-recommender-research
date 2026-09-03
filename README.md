# Behavioral Embeddings and Propensity Neighborhoods

Research on a practical question in energy-program recommendation: **when should a recommender trust a more specific local comparison, and when should it keep the broader model?**

The primary output is the accessible edition of **Behavioral Embeddings and Propensity Neighborhoods for Program Recommendation**:

- [Read the paper in HTML](papers/propensity-neighborhoods-accessible/propensity_neighborhood_accessible.html)
- [Download the accessible PDF](papers/propensity-neighborhoods-accessible/propensity_neighborhood_accessible.pdf)
- [Visit the project landing page](https://jonland82.github.io/energy-recommender-research/)

Everything else in this repository—earlier notes, follow-up experiments, scripts, and machine-readable results—is supporting material for that paper.

## The story arc

The argument moves through three ideas.

1. **Behavior creates a useful geometry.** A customer's pre-decision history is compressed into an embedding. Combined with ordinary customer attributes, that embedding improves the estimated probability that the customer will enroll in each eligible program.
2. **More specific neighborhoods spend support.** Adding matching criteria produces peers who are more alike, but fewer in number. Similarity improves while statistical uncertainty grows.
3. **A frontier governs trust.** The method prices mismatch against sample size. Local evidence can calibrate a score or support a confidence flag; when the evidence is too thin, the recommender keeps the direct propensity ranking or abstains.

The empirical ending matters: behavioral embeddings deliver the main predictive and ranking gain. Propensity neighborhoods deliver a smaller calibration and reliability benefit, but no confirmed improvement in the winning recommendation.

## The formalism, simply

For customer \(x\), let \(\mathcal A(x)\) be the eligible programs and let

\[
p_a(x)=\Pr(x\text{ enrolls in program }a)
\]

be the true enrollment propensity. A fitted model combines static attributes \(s(x)\) with a behavioral embedding \(z(x)\):

\[
\widetilde p_a(x)=f_a\!\left(s(x),z(x)\right).
\]

The direct recommender chooses the eligible program with the largest fitted propensity. To test whether local evidence should adjust that score, the method builds nested peer sets

\[
\mathcal N_1(x;r)\supseteq\mathcal N_2(x;r)\supseteq\cdots\supseteq\mathcal N_M(x;r).
\]

Each refinement adds detail. The worst mismatch among the remaining peers can only improve, while the peer count \(n_m\) can only fall. The central bound makes that tradeoff explicit:

\[
b_m(x)=\varepsilon_m^\rho(x)
+B\sqrt{\frac{\log(2K_x/\alpha)}{2n_m}}.
\]

Here \(\varepsilon_m^\rho\) measures how unlike the peers' remaining model errors are from the target's error; the second term is the uncertainty cost of limited support. The selected program's propensity regret is at most \(2b_m(x)\) under the theorem's stated assumptions. The useful neighborhood is therefore not automatically the deepest one—it is the point where a closer match is still worth the peers it costs.

## What the evidence says

In the chronological KuaiRec study, adding a behavioral embedding to static features:

- reduced propensity RMSE by **31.9%**;
- reduced top-1 regret by **6.4%**; and
- increased exact top-1 hit rate by **2.58 percentage points**.

The embedding improved RMSE on all ten splits. Neighborhood methods found real calibration information, but the accumulated exploratory and confirmation experiments did not establish a reproducible top-1 ranking improvement. The supported workflow is consequently:

1. filter to eligible programs;
2. construct a behavioral embedding from pre-decision history;
3. fit a propensity for every eligible program;
4. rank directly by fitted propensity; and
5. use neighborhoods as an optional layer for calibration, diagnostics, confidence, fallback, or abstention.

This is a study of **enrollment propensity**, not incremental enrollment, energy savings, or causal program impact.

## Repository map

| Path | Role |
| --- | --- |
| [`papers/propensity-neighborhoods-accessible/`](papers/propensity-neighborhoods-accessible/) | Primary accessible paper in HTML, PDF, and LaTeX |
| [`papers/propensity-neighborhoods/`](papers/propensity-neighborhoods/) | Longer technical note and extensions |
| [`propensity-neighborhoods-v2/`](propensity-neighborhoods-v2/) | Supplemental ranking and decision-boundary experiments |
| [`scripts/`](scripts/) | Reproducible KuaiRec experiments |
| [`results/`](results/) | Metrics, selected parameters, summaries, and experiment reports |
| [`data/`](data/) | Data acquisition and local-layout instructions; raw data are not tracked |

For the consolidated experimental interpretation, see [`propensity-neighborhoods-v2/OVERALL_CONCLUSION.md`](propensity-neighborhoods-v2/OVERALL_CONCLUSION.md).

## Reproduce the main experiment

The scripts target Python 3 and use the dependencies pinned in `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place KuaiRec at `data/kuairec/extracted` as described in [`data/README.md`](data/README.md), then run:

```powershell
python scripts\kuairec_behavior_embedding_test.py
```

The script writes machine-readable output under `results/`. Additional experiment commands and their interpretations live beside the relevant scripts and result folders.

## Citation

If this work informs your research, cite the paper title and author shown in the [accessible edition](papers/propensity-neighborhoods-accessible/propensity_neighborhood_accessible.html). A machine-readable citation record will be added when the manuscript receives a permanent identifier.
