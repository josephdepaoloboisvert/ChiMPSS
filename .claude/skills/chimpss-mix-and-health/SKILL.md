---
name: chimpss-mix-and-health
description: Build the mixing and ensemble-health report for a ChiMPSS parallel-tempering run (a FultonMarket output directory under runs/). It covers exchange and walker-diffusion statistics, walker-continuous state walks of every replica, per-block series of alpha-carbon RMSD, ligand RMSD and frame enthalpy over each block's importance-resampled set (one time point per block, mean and std over its draws), and resampling health (MBAR effective sample size, distinct frames, pool nesting). Use it whenever the user asks how a run is mixing, for state walks, round trips or replica diffusion, for resampling health, ESS or unique frames of the resampled sets, or for an RMSD or energy time series of the resampled 310 K ensemble, even when they only say "how is rep4 mixing", "is the resampled ensemble healthy" or "update the mix-and-health report".
---

# Mixing and ensemble-health report

Deliverable: a Claude Doc titled `<run> Mixing and Ensemble Report` with six sections, two drawn
charts and two tables, built from one script run. If the Claude Docs tools are not available,
write the same content to `<run>/analysis/mix_and_health.md` and tell the user where it is.

Reference example: `NTSR1_ML-301.hmr.rep4` (ten blocks, 60-rung trailblazed ladder, 2026-10-06),
https://claude.ai/code/artifact/077dc5b7-dc13-496e-a76f-7bdbb703b338. Its numbers are quoted below
as orientation; never reuse them for another run.

## 1. Compute everything with the script

```bash
PY=/home/josephdb/.local/share/mamba/envs/chimpss_v2/bin/python
SK=/media/josephdb/Barracuda4Tb/Chimpss_V2/ChiMPSS/.claude/skills/chimpss-mix-and-health
$PY $SK/scripts/mix_and_health.py /media/josephdb/Barracuda4Tb/Chimpss_V2/runs/<RUN>      # ~2 min per 10 blocks x 60 rungs
```

- Auto-detects the reference PDB (`Found input_pdb:` in the run logs, else `analysis/live_panel_config.json`),
  the ligand selection (`--ligand` printed in the logs, default `resname UNK`) and ps per iteration
  (`iter_length` in the logs). Logs are looked up at `<campaign>/logs/*<run_name>*.out`; pass `--logs GLOB`
  or `--ref PDB` when the layout differs. Without logs the ESS of engine-made resamples is unknown
  (it is only printed, never saved); say so rather than guessing.
- Writes `<run>/analysis/mix_and_health.json` (every number plus `chart_data`) and prints three markdown
  tables: Mixing, Resampled-pool series, Resampling health. Take every number in the doc from that
  stdout or JSON; do not recompute by hand.
- `--no-offline` skips regenerating resampled sets for blocks the engine never checked; `--no-coords`
  skips RMSD and the coordinate checks when positions.npy is absent; `--blocks 0:10` restricts the run;
  leave `--skip 10` alone (it is the analyzer's default and defines the frame index of resampled_indcs.npy).
  `--assume-continuity no` forces segment mode (section 4) for a run saved before the per-replica seeding
  fix when positions.npy is not there to prove it; `auto` (default) checks the coordinates.
- Cost: about 2.5 minutes for 10 blocks x 60 rungs (rep4) and 7 minutes for 23 blocks x 120 rungs (rep3),
  most of it in the offline MBAR solves; run it in the background for anything larger and poll the log.
- Use absolute paths in every command: the working directory moves between Bash calls in this harness.

## 2. Build the doc, one section per call

Skeleton first (title, byline `<?claude block asof?> · <?claude block me?>`, six pending blocks), open
it, then fill in this order:

1. **At a glance**: one lead sentence (blocks, exchange iterations, ns per replica, ns aggregate, rungs
   and temperature range) and five bold-led bullets: Mixing, Structure of the resampled ensemble,
   Energy, Resampling health, Coverage (which blocks lacked an engine resample and why).
2. **Mixing statistics**: lead sentence, the Mixing table from stdout (drop the half-legs column when it
   is all zero and say so in prose), a footnote on how acceptance and tau2 are defined, then two short
   paragraphs: MSD against lag and the diffusion coefficient; the first-passage estimate and what the
   walkers covered (walkers touching each end, median rung range, median distinct rungs).
3. **State walks**: the walks chart (section 3) and a reading sentence: how the pairing across resumes
   was verified, who climbs and who descends, that no walker crossed the ladder if that is the case, and
   which walkers were born at inserted rungs.
4. **Resampled-pool time series**: the series chart, then the z-score ranges against the pooled
   reference, the pairwise Calpha width trend, and the nesting caveat (section 4).
5. **Resampling health**: lead sentence, the health table (move the "draws by rung" column into a
   sentence), and the paragraph explaining the engine's uniform-ESS warning lines (section 4).
6. **How each quantity was computed**: the bullets at the end of this file, with the run's own numbers.

Chat lines between calls stay short ("outline up; now mixing"). Hand off with one line and the link.

## 3. Charts

```bash
$PY $SK/scripts/render_charts.py <run>/analysis/mix_and_health.json \
    --title-walks "<finding, e.g. Walkers diffuse along the ladder; those that touched 310 K are blue, none crossed the whole ladder>" \
    --title-series "<finding, e.g. Calpha RMSD from the start climbs block after block while ligand RMSD and frame enthalpy stay flat>"
```

Writes `<run>/analysis/charts/state_walks.js` and `pool_series.js`: each file is the complete one-line
`code` of a widget. Both titles default to a neutral finding when the flags are omitted; the walks caption
and panel labels switch automatically to segment mode ("slot k") when `walker_continuity` is false, and the
series chart leaves out uniform-draw blocks (the legend says how many), so its x axis can skip block numbers. Create each widget (`verb: create, object: node, engine: widget`, parent = the tab's
file id) in the same `batch` as the prose `replace` that embeds it (`blocks: {chart: {type: embed, ref:
"node/$lid:w", caption: ...}}`); caption = the source files and extent. Then screenshot each widget
(`read`, `engine: widget`, `payload: {kind: screenshot}`) and look: titles and the 11 to 12 px caption
lines clip past about 125 characters, so shorten with the text arm (`{kind: text, id: title|subtitle|legend}`)
rather than redrawing. The walks chart is one panel per walker (six columns, eight above 72 walkers),
blue for walkers that touched the bottom rung; the series chart is three stacked panels (two when the run
has no ligand) with a point and a plus or minus one std whisker per block, the pooled mean and std as a
dashed line and grey band, hollow points for offline resamples, and the first and last values labelled.

## 4. How to read the numbers

- **Acceptance.** Trailblazed ladders target 0.40 per neighbour pair (ladder length 1.19); geometric
  campaign ladders run near 0.65 with insertion thresholds 0.5 then 0.35. The per-block min to max spans
  about 0.2 to 0.75 from counting noise alone over 30 iterations; the trailing-block audit, not one block,
  decides insertions.
- **Diffusion.** Rung changes per replica-iteration near 0.55 and an MSD that grows linearly with lag mean
  free diffusion; D = MSD / 2 lag, printed in rung² and in K² per iteration (rep4: 0.47 rung² = 0.44 K²;
  rep3 on 120 rungs: 2.6 rung² = 0.59 K²). A round trip needs about (K - 1)^2 / D iterations: on rep4
  that is 7,300 iterations, about 25 blocks or 130 ns per replica, so zero round trips in a run shorter
  than that is expected and must be stated as such. Quote walkers touching each end instead, and the MSD
  slope.
- **Walker continuity.** The engine relabels replicas at every resume (replica k takes the configuration
  that ended at rung k), so columns of states.npy are re-paired on their final rung and the script checks
  the pairing on coordinates (first saved frame of a block equals the paired walker's last frame, 0.000 nm,
  against about 0.5 nm for the same column). Runs saved before the per-replica seeding fix
  (commit 89ac4d6 on feat/thermodynamic-trailblazing, 2026-10-04; every campaign run up to rep3 and all
  5HT2BR runs) restarted every replica from the state-0 configuration at each resume: the script finds
  pairing RMSD well above zero (rep3: 0.47 nm), sets `walker_continuity` false and switches to segment
  mode: columns are replica slots, every block is an independent segment, the walks chart lifts the pen at
  each block start and labels panels "slot k: T K" with no blue, MSD and the diffusion coefficient use
  within-block pairs only, the end-touch and half-leg columns are blank (the restart scramble places slots
  at random, so touching an end is not coverage), and the summary reports slot-block segments instead:
  their median within-block rung range and step sizes (rep3: 16-step segments, mean step 1.8 rungs, D =
  2.6 rung² = 0.59 K² per iteration, tau2 about 700 iterations). Quote D in K² per iteration when comparing
  ladders of different density, and never quote round trips or cumulative coverage for such a run.
- **Restart scramble.** Replicas that start a block from one identical configuration accept every swap on
  the first exchange round (all energy differences are zero) and scatter over the whole ladder: on rep3 the
  first step of every block moves 40 rungs on average (up to 119) against 1.8 afterwards. The script
  detects this per block (first-step mean jump above 2 rungs and above three times the later median),
  lists the blocks as `scramble_blocks`, excludes those steps from every walker statistic and lifts the pen
  there in the chart. Expect it on every block of an old-seeding run and on block 0 of any run started
  without per-rung seeds (a geometric ladder without a trailblaze walk); rep4 has none.
- **Resampled sets are nested cumulative pools.** Each block's set is 1,000 draws from all post-t0 frames
  of all rungs so far, weighted to 310 K. Consecutive sets share most of their distinct frames (62 to 77 %
  on rep4 from block 2 on) and the newest block supplies a shrinking share of draws (12 to 18 % after block
  4), so the per-block series is a running weighted summary, not a set of independent snapshots. Always
  say this next to the series chart.
- **A uniform resample is a failed MBAR solve.** When the real resample line reads "effective sample size N
  of N (100.00 %)" with the uniform warning, pymbar did not converge (7 to 9 frames per state on the 110 to
  120-rung campaign ladders: rep3 blocks 0 to 2, 4, 15, 16 and 18) and the set is an unweighted draw from
  every rung, not a 310 K ensemble; the script marks it "uniform: MBAR did not converge" in the health table
  and the draws-by-rung column shows about 1 % per rung. Say in the doc that those blocks' convergence
  comparisons were made on unweighted pools, and exclude them from the structural series or mark them.
- **ESS fraction is set by the ladder, not by the run length.** At about 1 K spacing only the three lowest
  rungs carry 310 K weight (rep4: 55 to 60 % of draws from the 310.0 K rung, 30 to 37 % from the next,
  6 to 9 % from the third, at most 1 % beyond), so ESS stays near 1.7 rungs' worth of frames, 2.7 to 2.8 %
  of any pool. ESS and distinct frames grow with the pool (35 to 278 and 53 to 320 of 1,000 over ten
  blocks on rep4). A collapse (ESS below about 10, a handful of distinct frames) is a real problem; a
  low percentage is not.
- **Missing resamples.** The engine skips the convergence check on the block that reaches the aggregate
  time cap, so that block has no resampled set (rep4 blocks 1 and 9, each the last block of a
  continuation). The script regenerates them with `FultonMarketAnalysis(upper_limit=...)` and the same
  energy-based t0 detector and 1,000 draws; mark them "offline" in the tables and hollow in the chart.
  The engine's ratchet, freeze and pool-bound options are not replicated, so an offline t0 can differ.
- **The uniform-ESS warning.** Lines such as "effective sample size 40.0 of 40 (100.00 %)" followed by
  "weights are uniform" before each real resample come from the backfill of a rung missing from an
  earlier block (a single-state MBAR solve, uniform by construction); they are not the 310 K resample.
  Explain this in the health section so nobody reads them as a failed resample.
- **Structure and energy.** Calpha RMSD is measured against the equilibrated input structure after Calpha
  superposition; a monotone rise with a growing pairwise width (rep4: 1.9 to 3.2 A and 1.5 to 2.8 A over
  305 ns aggregate) means the receptor is still relaxing even when energies are flat. Ligand RMSD uses the
  heavy atoms of the ligand selection after the same superposition with no ligand fit; a 95th percentile
  under about 1.8 A means the pose is kept. Enthalpy is H = u / beta from energies.npy; block means
  within 0.2 pooled std are stationary energies. Report every quantity as mean and std over the draws
  and as z against the pooled reference of all sets; never as a series along one temperature state.

## 5. Pitfalls

- Never write into `saved_variables/`; everything the script and the doc need goes under `<run>/analysis/`.
- positions.npy is a headerless float32 memmap of shape (iterations, replicas, atoms, 3); the script
  checks the byte size and disables coordinate metrics for a block that does not match.
- Widget code is validated: no `top`, `parent`, `self` or `window` identifiers (use `ty`, `y0`), no
  computed array indexing (`rows[i]`; use `.at(i)` or `.map`), reader functions never stored in lists,
  text ids only on typed words outside loops. The templates already follow these rules; when editing a
  chart prefer the text arm or a `draft-edit` with an exact quote, then `publish`, then screenshot again.
- Do not shell out with `pkill -f` patterns that match your own shell; this harness kills the Bash call.
- The 60-walker chart holds 301 characters per walker; ladders above 62 rungs use two characters per
  frame (`cw: 2`), which the template decodes, and eight panel columns.

## Methods bullets (template for section 6)

- **Run.** `runs/<RUN>`, blocks <first> to <last> of `saved_variables/`, <ps> ps per exchange iteration,
  <rungs> rungs (<T_min> to <T_max> K); the first saved frame of every block is its start configuration.
- **State walks.** Columns re-paired at every resume on the walker's final rung (temperature-matched
  across ladder changes); verified on coordinates at every boundary (<max pairing RMSD> nm against
  <same-column mean> nm for the same column).
- **Mixing.** Pooled per-pair Metropolis acceptance from `energies.npy` (`analysis/ladder/*.json`); rung
  changes per replica-iteration; tau2 = -1 / ln |lambda2| of the rung transition matrix (per block, and
  cumulative on walker-continuous trajectories); D from the MSD slope; half-leg estimate (K - 1)^2 / 2D.
- **Resampled sets.** `resampled_indcs.npy` (state, frame) mapped through the analyzer's concatenation
  (skip 10 per block) to (block, iteration, replica) and checked against <n> random frames of each
  `resampled_trj.dcd` (max Calpha RMSD <x> A). Offline sets for blocks <list>: `FultonMarketAnalysis`
  truncated at the block's last frame, energy-based t0, MBAR weights at state 0, 1,000 draws.
- **Structure.** Coordinates from `positions.npy`; Calpha RMSD on <n> Calpha atoms after superposition on
  <reference PDB>; ligand RMSD on <n> heavy atoms of `<selection>` after that superposition, no ligand
  fit; width = draw-weighted mean pairwise Calpha RMSD within a set.
- **Enthalpy.** H = u / beta averaged over the state columns in float64, relative to the pooled mean (<H> kJ/mol).
- **Statistics.** Unweighted mean and std over the draws (they already carry the MBAR weights); pooled
  reference = all draws of all sets; z = (block mean - pooled mean) / pooled std.
- **Resampling health.** ESS = 1 / sum w^2 of the normalised 310 K weight column (engine log line before
  each written set, recomputed for offline sets); distinct frames = unique (block, iteration, replica)
  triples; overlap = share of a set's distinct frames present in the previous set.
- **Outputs.** `analysis/mix_and_health.json` and `analysis/charts/*.js`, produced by this skill's scripts.
