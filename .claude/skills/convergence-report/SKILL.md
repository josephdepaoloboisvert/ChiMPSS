---
name: convergence-report
description: Build a convergence method-assessment report for a completed ChiMPSS parallel-tempering run. Use when asked to assess equilibration or convergence detection methods against a finished simulation, to produce or update a convergence report artifact, or to evaluate whether a run converged under alternate metrics.
---

# Convergence method-assessment report

Produces a published HTML report assessing equilibration and convergence
**detection methods** against a completed PTRE run.

The framing matters and is easy to get wrong. The deliverable is an assessment
of the *methods*; the run is the **test bed**. Findings about the simulation
(drift, resample collapse, non-stationarity) are evidence about how a detector
behaves, not the product. A report organised around "did this run converge?"
is the wrong document — organise it around "which detector should we use, and
what does this run demonstrate about it?"

## 0a. The production criterion measures pool overlap, not convergence

Established on NTSR1 rep0 at 4,025 ns (161 blocks) and true of the criterion
as designed, not of one run. The JSD between two blocks' MBAR resamples is a
monotone function of how much *source data* the two resamples share: median
0.02–0.05 at >90 % overlap of their `[t0:N]` intervals (92–99 % of pairs clear
0.10), 0.17–0.24 at zero overlap (0–28 % clear it), Spearman −0.24…−0.32, and
no dependence on time separation among pool-disjoint pairs. Under a trailing
max-N_eff t0, adjacent blocks share ~90 % of their pool, so trailing windows
pass by construction and the all-history window fails by construction. Earlier
campaigns, with an early stable t0 and nested pools, passed for the same
reason. The 0.10 threshold sits at or below the noise floor for *independent*
data (0.10–0.12 on interleaved halves; 0.11–0.18 between independent segments
of any length).

Consequences for a report: never present a trailing-window pass as evidence of
convergence without the pool overlap beside it (`chimpss-convergence-bundle`
now prints it; `replay(..., disjoint_only=True, pool_bounds=...)` gives the
statistic on independent pairs); do not build nulls by shuffling cumulative
resamples between halves (the null inherits the overlap and reads far too
tight — covariance overlap 0.52 vs a 0.83 "null" that became 0.64 vs 0.70 on
independent data); and do not use k-means populations as a convergence
observable unless more than one basin is established (cells 1.7 Å apart on a
1.5 Å ensemble turn a 0.2 Å shift into a "−5 kcal/mol population change").

## 0b. Validation runs: use the panel

```bash
chimpss-convergence-panel <run_dir> \
  --pdb <run_dir>/saved_variables/<block>/resampled_top.pdb \
  --pangpcr_prefix models/pangpcr1513 --pdb_code <PARENT_PDB> \
  --stor_dir many_structures/ --prefix_curve
```

`chimpss.analysis.convergence_panel` reads the eight lowest rungs of every
iteration together with their reduced potentials, solves MBAR over those
rungs per block so every frame is weighted in the block's 310 K ensemble
(on NTSR1 rep0: 41/26/15/7/4 % on rungs 0–4, the production resample's own
proportions; ESS 43 of 152 frames per block), superposes the conserved Cα
atoms on the pan-GPCR reference, and forms per-block weighted means — the
time-ordered series. Blocks weigh equally. It discards the first quarter of
blocks and gates on effect sizes against experimental tolerances on the two
contiguous halves: mean-structure RMSD vs the same-state reproducibility of
the receptor's own structures (0.92 Å for NTSR1; family median 0.69 Å),
state assignment per quarter plus half-shift vs within-state spread, ensemble
width as Wasserstein-1 in Å, and basin populations (ΔG vs kT) only when a
bimodality test finds more than one basin. Block bootstrap for intervals,
interleaved halves for the noise floor. Below 16 equilibrated blocks it
returns INSUFFICIENT. On rep0 it read CONVERGED from 32 blocks (800 ns) and at
every later prefix; on the 6-block rep1, INSUFFICIENT. τ, n_eff, naive vs
corrected p, R̂ and the Flyvbjerg–Petersen error are printed as diagnostics and
never gate — a null-rejection test gets harder to pass as a converged run
lengthens, so it cannot be a stopping rule. See the report "Which Metrics See
Convergence" (28 Sep 2026) for the full audit.

## 0c. Live stopping rule for a fresh simulation

`chimpss-fultonmarket --convergence panel` makes the panel the stopping rule
inside the PTRE block loop (`FultonMarket._evaluate_panel`, rank 0, verdict
broadcast). After every block it reads that block's `positions.npy` for the
lowest rungs into `<run_dir>/analysis/live_panel/`, evaluates the four gates on
everything cached, prints a `[PASS]/[FAIL]` table beside the JSD report, and
stops only when CONVERGED has held for `--panel_consecutive` blocks (default
3) and the minimum-fraction floor is past. The JSD machinery still runs every
block for its resampled artifacts; `both`/`either` combine the two verdicts;
the aggregate-time ceiling stops the run in every mode. State persists in
`live_panel_state.json`, so a resubmitted job continues its streak.

Pre-flight from a login node (compute nodes have no network):

```bash
chimpss-convergence-panel <run_dir> --pdb <equil.pdb> \
  --pangpcr_prefix models/pangpcr1513 --pdb_code <SYS_PDB> \
  --stor_dir many_structures/ --prepare
```

This resolves the 190 conserved atoms against the topology (every protein
chain), caches the GPCRdb numbering, computes the tolerances and writes
`live_panel_config.json`; it fails fast if any conserved position is absent.
That failure is information about the *system*: the 5HT2BR_LSD construct as
built ends at residue 336 (BW 6.47) -- no TM7, no H8 -- because the
RepairProtein tails were `[56, 336]` while the parent 7SRS resolves to 460.
Rebuild such a system; do not switch it to `jsd` to make the check pass.
`slurm/config.sh` sets `PTRE_CONVERGENCE=panel`; the job scripts run the
pre-flight when `live_panel_config.json` is absent and exit 3 if it fails.

## 0d. Two rules about what a frame is

**Reweight; never pool rungs unweighted.** Frames from rungs above the target
are samples of a different ensemble. Per-block MBAR over the rungs actually
read gives the exact weights for an estimate built from those frames; the
cost is one small solve per block. Uniform pooling of five rungs changed
nothing on rep0 (0.480 vs 0.482 Å) because it is one basin, and would bias a
system near a transition toward the warmer ensemble exactly when the state
gate matters.

**Only the block carries a time index.** The replica at the 310 K rung
changes identity at almost every exchange attempt (nine times per
19-iteration block on rep0, six replicas per block). A "rung-0 trajectory" is
a valid sample of the 310 K ensemble but not a dynamical series; blocking
analyses, drift plots and equilibration detection belong on per-block
weighted means or medians, never on frames.

## 0. What the criterion cannot see

Two blind spots are structural, not bugs, and both matter for how the report
argues its verdict.

The pairwise statistic histograms the *upper triangle of each block's own
distance matrix* -- the distribution of distances within one ensemble. That is
invariant to bodily displacement: a synthetic ensemble moved 20 standard
deviations scores exactly 0.000000, while widening its internal spread 10%
scores 0.348. So a run sliding along a collective coordinate with constant
internal geometry reads as converged.

A linear trend test is blind to non-monotone motion for the same kind of
reason. Each statistic is insensitive to a mode it was never designed to see.
The report should name both explicitly and carry a measurement that covers
them -- in practice, projection onto an external basis.

## 1. Collect the evidence

```bash
chimpss-convergence-bundle <run_dir> \
  --pdb <run_dir>/saved_variables/<block>/resampled_top.pdb \
  --pangpcr_prefix models/pangpcr1513 --pdb_code <PARENT_PDB> \
  --stor_dir many_structures/ \
  --out_dir <run_dir>/analysis/report
```

Writes `bundle.json` (every number) and `figures/*.svg` (every chart). CPU
only, minutes, from cached block data. The alpha-carbon and pairwise caches are
built once under `<run_dir>/analysis/` and reused.

Omit `--pangpcr_prefix` if no pan-GPCR basis exists; build one with
`chimpss-generate-pca` (see `models/` for existing bases).

## 2. Read the bundle before writing anything

Do not narrate the JSON. Decide, per method, what this run demonstrates — and
say when it demonstrates nothing. Negative and refuted results are findings:
record them.

## 3. Assemble the page

```python
from chimpss.analysis.report_html import page, figure, check_document
from chimpss.analysis import report_figures as rf
```

`page(title, body)` supplies the stylesheet and shell. `figure(svg, caption)`
wraps one chart. Existing class names: `.wrap .mast .eyebrow .dek .runline
.verdict .qualifier .tablewrap .sub .lines .line .src .chart .figure .mono
.num`.

Suggested structure — methods first, run second:

1. **What was assessed** — method / basis / verdict table with citations
2. One section per method family, each arguing its verdict from a figure
3. **What the test bed showed** — the run's own behaviour, explicitly demoted
4. **What would not have changed the verdict** — negative results
5. **Recommended protocol** — adopt / do not adopt / still unresolved

## 4. Validate, then publish

```python
problems = check_document(html)   # must be empty
```

Then publish with the `Artifact` tool. To update an existing report pass its
`url`, or it becomes a separate artifact.

## Pitfalls that have actually bitten this report

**Realign the alpha-carbon cache.** `ca_coords.npz` files built before the
shared-reference fix superpose each block onto its *own* first frame, leaving
blocks in arbitrary global orientations. Every structural metric then measures
rotation, not conformation: adjacent-block population JSD reads 0.825 instead
of 0.165, and clustering separates blocks trivially. `chimpss-convergence-bundle`
applies `realign_cache` by default; `--no_realign` is correct only for a cache
you know was built with a shared reference.

**Marks outside the viewBox vanish silently.** No error, and the chart looks
plausible without them — one published version dropped thirty JSD points off
the top of a plot. Always run `check_document`.

**Sample the null properly.** Fewer than ~20 half-splits per block gives an
unstable 99th percentile, and every derived pass-fraction moves with it. A thin
null once produced a "populations are mildly stabilising" claim that vanished
at `--null_reps 25`: the improvement was an artifact.

**Never publish a figure that disagrees with the prose beside it.** If a
recomputed number contradicts published text, fix the text. Two numbers in an
earlier version were not reproducible from the module's own code and had to be
corrected rather than kept.

**A trend test is not a stationarity test.** Linear regression on a per-block
mean detects *monotone* drift and nothing else. An ensemble that descends,
reverses, plateaus and turns again passes it while moving throughout. On
`NTSR1_ML-301.hmr.rep0` at 4,025 ns, PC1 shows no linear trend over the final
forty blocks (p = 0.23) and a two-sample KS test on the same blocks rejects
stationarity (p = 0.034). Run both, and report both.

**Do not conclude from a short tail.** The same PC1 test at 144 blocks returned
KS p = 0.69 on 23 blocks and was reported as provisional evidence of settling;
39 blocks inverted it. Treat a non-significant result on a short series as
absence of evidence, and say so rather than hedging with "provisionally".

**Track several principal components, and rank them by drift, not variance.**
Explained-variance ratio describes the *training* set, not the run: the top
three pan-GPCR components hold 48% of training variance but only 7% of this
trajectory's own variance. Because the basis is orthonormal, one angstrom along
PC5 is exactly as much atomic motion as one along PC1, so variance share
applies no discount to a displacement. What variance rank does affect is axis
determinacy -- check the eigenvalue gap, and flag any component whose gap to
its neighbour is under ~15%, since the subspace is then determined but the
split within it is not. Tracking PC1 alone would have called this run settled
while five other components were still drifting.

**Say which detector gated the window replay.** Passing counts depend on it,
heavily: a one-block window on NTSR1 rep0 passes 5 blocks under `max_neff` and
23 under the MSER family, from the same cache. `--gate_detector` selects it and
the bundle records the choice; quote it wherever you quote a passing count.

**State the missing positive control.** Methods assessed against a run that
never converged can be judged on stability, reproducibility, cost and
false-positive behaviour — never on sensitivity. Say so explicitly; it is the
main limitation of this whole exercise.

## Reference

Detectors and metrics live in `chimpss.analysis.equilibration`,
`chimpss.analysis.convergence`, `chimpss.analysis.mixing` and
`chimpss.analysis.pca_equilibration`; each module docstring carries its
citations and what has already been established or refuted against
`NTSR1_ML-301.hmr.rep0`.
