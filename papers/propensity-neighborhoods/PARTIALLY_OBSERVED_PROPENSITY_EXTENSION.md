# Potential Extension: Propensity Anchors and Certified Neighborhood Transfer

## Status

This is a research direction, not yet a claim of the paper. It picks up from the operational peer-mean corollary and known-coordinate remark in `propensity_neighborhood_note.tex`. The current implementation to extend is `scripts/kuairec_operational_frontier.py`, especially `build_candidates()` and `run_seed()`.

The central idea is to use propensity coordinates that are genuinely known at recommendation time as exact anchors. Neighborhoods would be filtered by similarity in those known residual coordinates, and a separately validated transfer bound would connect anchor similarity to residual similarity on the still-unknown programs.

## Why this could matter

The current residual-refinement theorem depends on the unknown target residual

$$
\rho(x)=p(x)-\widetilde p(x).
$$

That makes its general mismatch term difficult to evaluate at recommendation time. If some coordinates of $p(x)$ are structurally known, however, their residuals are also known exactly. For example, if the estimand and decision time genuinely imply $p_b(x)=1$, then

$$
\rho_b(x)=1-\widetilde p_b(x).
$$

Similarly, a structurally impossible program with $p_b(x)=0$ has

$$
\rho_b(x)=-\widetilde p_b(x).
$$

These exact residuals reveal where the base model is over- or underpredicting the target on observed coordinates. Historical customers with similar revealed residual patterns may be especially informative peers for the target's unobserved coordinates.

This could turn partial propensity observation into an operational neighborhood methodology:

1. identify exact propensity coordinates;
2. convert them to exact residual anchors;
3. retain peers with similar anchor residuals;
4. calibrate the unknown program coordinates from those peers; and
5. select the refinement depth by an observable finite-sample bound.

## Essential estimand guardrail

An observed positive outcome is not normally a true propensity of one. If a customer enrolled once, then the enrollment outcome is $Y_b(x)=1$, but the probability of a future enrollment, renewal, or response need not satisfy $p_b(x)=1$.

There are therefore two distinct cases.

### Structurally known propensity

The decision-time information makes the modeled event deterministic. Here $p_b(x)\in\{0,1\}$ is genuinely known, and the coordinate can enter the theorem with zero uncertainty.

### Observed state anchor

Current participation $q_b(x)=1$ is known, but the future target $p_b(x)$ remains latent. The state $q_b(x)$ can still be used as a neighborhood coordinate, but a transfer assumption is required to connect state similarity to future residual similarity.

The extension should never turn a one-time observed outcome into a known propensity without a structural argument tied to the estimand.

## Proposed setup

Let $\mathcal A$ be the program catalog and $\mathcal A(x)$ the eligible action set. Let $O(x)\subseteq\mathcal A$ contain coordinates known exactly at recommendation time; these anchors need not themselves remain eligible. The unknown eligible coordinates are

$$
U(x)=\mathcal A(x)\setminus O(x).
$$

Define the revealed residual signature

$$
s_O(x)=\bigl(p_b(x)-\widetilde p_b(x):b\in O(x)\bigr).
$$

For a historical customer $x_i$ with the required anchor coordinates, define the observable anchor discrepancy

$$
d_O(x,x_i)
=
\max_{b\in O(x)}
\left|
\rho_b(x)-\rho_b(x_i)
\right|.
$$

The reference residuals must themselves be exact or estimated with explicitly accounted-for uncertainty. If anchor availability differs by customer, the initial theorem should require a common anchor set; partial overlap can be treated later.

Let $d_m(x,x_i)$ be the cumulative feature or embedding distance after $m$ blocks. Define an anchor-filtered neighborhood

$$
\mathcal N_m(x;r,\tau)
=
\left\{
i:
d_m(x,x_i)\le r,
\quad
d_O(x,x_i)\le\tau
\right\},
$$

with size $n_m(x;r,\tau)$.

Adding feature or anchor coordinates can only remove peers. The existing thin-neighborhood proposition therefore continues to apply.

## The substantive transfer condition

Exact anchors help only if similarity on their residuals says something about residuals on the unknown programs. A useful condition is

$$
\max_{a\in U(x)}
\left|
\rho_a(x)-\rho_a(x_i)
\right|
\le
L,d_O(x,x_i)+\gamma_m(r).
$$

Here:

- $L$ measures how strongly revealed residual disagreement transfers across program coordinates;
- $\gamma_m(r)$ allows residual differences not explained by the anchors; and
- refinement in behavioral or static coordinates may reduce $\gamma_m(r)$.

This condition is the central scientific assumption. Without some version of it, known coordinates can improve retrieval heuristically but cannot certify errors on unrelated unknown coordinates.

A more flexible version could replace $L d_O+\gamma_m$ with a learned monotone envelope $\Phi_m(d_O,d_m)$, but the linear form is the best starting point because its role in the bound is transparent.

## Candidate theorem: anchor-transfer refinement frontier

Suppose:

1. the anchor propensities in $O(x)$ are known exactly;
2. the residual-transfer condition holds simultaneously for all peers retained at level $m$;
3. reference outcomes $Y_{ia}\in[0,B]$ are conditionally independent for each $a\in U(x)$;
4. reference scores are out of fold; and
5. neighborhoods are fixed independently of the reference outcomes used for correction.

For

$$
\widehat p_{a,m}(x)
=
\widetilde p_a(x)
+
\frac{1}{n_m}
\sum_{i\in\mathcal N_m(x;r,\tau)}
\left(Y_{ia}-\widetilde p_a(x_i)\right),
$$

the existing proof strategy should yield, with probability at least $1-\alpha$,

$$
\max_{a\in U(x)}
\left|
\widehat p_{a,m}(x)-p_a(x)
\right|
\le
L\tau+\gamma_m(r)
+
B\sqrt{
\frac{
\log\!\left(2|U(x)|M/\alpha\right)
}{2n_m}
}.
$$

If the transfer condition is itself estimated with failure probability $\delta$, the combined guarantee becomes at least $1-\alpha-\delta$.

Define

$$
b_m^{\mathrm{anchor}}(x)
=
L\tau+\gamma_m(r)
+
B\sqrt{
\frac{
\log\!\left(2|U(x)|M/\alpha\right)
}{2n_m}
}.
$$

Then a general recommendation-regret bound is

$$
p_{a^*}(x)-p_{\widehat a_m}(x)
\le
2b_m^{\mathrm{anchor}}(x).
$$

The factor can sharpen according to which coordinates determine the decision:

- if both the true best and selected programs are unknown, the usual bound is $2b_m^{\mathrm{anchor}}(x)$;
- if exactly one is a known coordinate whose propensity is inserted exactly, the bound is $b_m^{\mathrm{anchor}}(x)$; and
- if both are known coordinates, their comparison has zero estimation error.

The union-bound dimension also falls from $|\mathcal A(x)|$ to $|U(x)|$. Thus revealed propensities can improve the certificate in three ways: better peer filtering, zero coordinatewise error, and a smaller simultaneous-comparison penalty.

## Operational selection rule

For candidate feature depths, anchor sets, radii, or tolerances, choose

$$
\widehat m(x)
\in
\arg\min_m
\left\{
\widehat L\tau_m
+
\widehat\gamma_m(r_m)
+
B\sqrt{
\frac{
\log\!\left(2|U(x)|M/\alpha\right)
}{2n_m}
}
\right\}.
$$

Level zero must remain available. It represents the uncorrected base model and prevents the procedure from forcing a neighborhood correction when the anchor match or peer support is inadequate.

This rule preserves the paper's central geometry:

- more exact coordinates or stricter anchor tolerances reduce the transfer term;
- those filters also produce thinner neighborhoods;
- thinner neighborhoods increase sampling uncertainty; and
- the bound identifies the last useful anchor or feature block.

## Making the transfer condition testable

The theorem is only compelling if $L$ and $\gamma_m$ are estimated without using the target outcome. A possible protocol is:

1. use a separate transfer-training sample containing multiple well-measured propensity coordinates per customer;
2. repeatedly mask some coordinates as unknown;
3. construct anchor distances from the remaining revealed coordinates;
4. measure residual mismatch on the masked coordinates;
5. fit a monotone upper-envelope model for masked residual mismatch; and
6. calibrate its excess error on another disjoint sample.

For example, define pairwise transfer scores

$$
T_{ij,m}
=
\max_{a\in U_{ij}}
|\rho_a(x_i)-\rho_a(x_j)|
-
\widehat L,d_O(x_i,x_j).
$$

A calibrated upper quantile of $T_{ij,m}$ can supply $\widehat\gamma_m$. Ordinary pairwise conformal calibration is not automatically valid because pairs sharing customers are dependent. The first implementation should preserve independence by forming disjoint customer pairs or use customer-clustered calibration with the target customer as the exchangeable unit.

An alternative is to calibrate the complete anchor-selection algorithm end to end on held-out customers. That provides marginal realized-outcome coverage, as in the current paper, but it must still be described separately from latent-propensity coverage.

## Experimental program

### 1. Synthetic verification

Start with a data-generating process where the full propensity vector is known and cross-coordinate residual transfer is controlled. Vary:

- the number of revealed coordinates;
- transfer strength $L$;
- unexplained slack $\gamma$;
- the number of reference peers;
- anchor noise; and
- embedding dimension.

Verify empirical coverage, bound width, neighborhood use, RMSE, regret, and the predicted thin-neighborhood frontier. This is the cleanest test of the latent-propensity theorem.

### 2. KuaiRec masked-coordinate study

KuaiRec does not contain genuinely deterministic propensity coordinates. It can nevertheless provide a diagnostic masked-coordinate study using category completion rates estimated from repeated interactions:

- divide interactions chronologically;
- estimate anchor-category rates using only information available before the recommendation time;
- mask other categories as unknown;
- learn the anchor-to-masked residual transfer envelope on separate users;
- evaluate future category rates on untouched users; and
- label the exercise semi-synthetic or masked-coordinate, not a real known-propensity deployment.

Do not reveal a held-out future category rate and then describe it as known at test time.

### 3. Energy-program validation

The strongest application requires data containing structural program states or repeated decision opportunities. Before modeling, create an anchor inventory with:

- the exact event whose propensity is modeled;
- why the coordinate is deterministically zero or one at decision time;
- whether the program remains eligible for recommendation;
- how historical anchor propensities are measured; and
- whether current participation is an anchor state or the actual target.

The method is most persuasive if known participation in one program predicts residual patterns for complementary unknown programs.

## Success criteria

The extension should enter the paper as a main theorem only if it demonstrates all of the following:

1. a non-tautological and empirically supported transfer condition;
2. no target-outcome leakage in anchor construction or level selection;
3. nominal latent-propensity coverage in synthetic experiments;
4. narrower bounds or more certified decisions as anchors are revealed;
5. an advantage over base conformal calibration or the current peer-mean rule; and
6. an honest distinction between structural propensity knowledge and observed outcomes.

If it improves retrieval but not certificate width, it remains a useful methodology but should be presented as an empirical anchor-selection procedure rather than a stronger guarantee.

## Where to pick up

The current paper already contains the necessary starting points:

- `Theorem~\ref{thm:frontier}`: the existing mismatch-versus-support result;
- `Corollary~\ref{cor:operational}`: the operational peer-mean bound;
- `Remark~\ref{rem:known}`: zero uncertainty for known coordinates; and
- the section `Chronological evidence from KuaiRec`: the current empirical boundary between latent and realized targets.

The existing experiment runner provides the split discipline and candidate construction:

- `scripts/kuairec_operational_frontier.py::build_candidates()` constructs corrections and observable frontiers;
- `scripts/kuairec_operational_frontier.py::run_seed()` implements the disjoint base, residual, reference, conformal, and test roles; and
- `results/kuairec_operational_frontier/README.md` records the present baseline and its limitations.

Resume in this order:

1. lock the estimand and write the anchor inventory;
2. formalize the transfer assumption and prove the anchor-transfer theorem on paper;
3. create `scripts/propensity_anchor_synthetic.py` and verify the theorem under known ground truth;
4. add a masked-coordinate KuaiRec runner only after the synthetic coverage test passes;
5. compare max-neighbor, peer-mean, anchor-transfer, and base conformal rules;
6. decide whether the result improves bounds or only prediction; and
7. update the main manuscript only after those results are known.

The immediate next task is therefore not manuscript revision. It is a synthetic experiment that tests whether increasing the number of exact anchors produces the predicted decrease in transfer error before peer support becomes too thin.
