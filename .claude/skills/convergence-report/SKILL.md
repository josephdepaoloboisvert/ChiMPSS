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
