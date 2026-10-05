# Thermodynamic trailblazing for the ChiMPSS temperature ladder

## Context

FultonMarket (`src/chimpss/fultonmarket/`) runs pure-temperature parallel tempering on
openmmtools' `ParallelTemperingSampler`. The ladder is fixed up front by
`geometric_distribution(T_min, T_max, n_replicates)` and then only ever *grows*: whenever a
neighbour pair's measured exchange acceptance falls below a threshold (0.5 in block 0, 0.35
after), a rung is inserted at the arithmetic midpoint and the block restarts.

Two things are wrong with that, and both are visible in the campaign data:

1. `geometric_distribution` is not geometric. It is `Tmin + (Tmax-Tmin)·(e^{i/(n-1)}-1)/(e-1)`,
   so the gap grows ~2.7× from bottom to top of the ladder. Thermodynamic-length theory says
   that for near-constant heat capacity the optimal ladder has a *constant ratio* T_{i+1}/T_i,
   i.e. gaps that grow only ~18 % over 310–367 K. The campaign ladder is therefore over-dense
   at the bottom and under-dense at the top.
2. The insertion rule then compensates by bisecting the sparse pairs, which overshoots
   (a midpoint roughly doubles local density). `slurm/config.sh` asks for 60 rungs and claims
   insertion "should never fire"; every production run actually saved 110–132 rungs in block 0
   (NTSR1 rep3: 110 → 120 after seven block-0 rebuilds). Each rebuild discards the sampling
   done so far in the block, and the resulting ladder has non-uniform acceptance, so the
   replicas' walk through temperature is slowest wherever the ladder happens to be sparsest.

**Thermodynamic trailblazing** (Minh, AlGDock; theory from Shenfeld et al. 2009 and
Nguyen & Minh 2016, both in `reference_material/`) replaces the guessed ladder with one that is
*measured*: starting at T_min, sample the energy at the current rung, use its standard
deviation (the square root of the Fisher metric) to step to the temperature that is a fixed
thermodynamic length ΔL* away, seed the new rung from the previous one, verify the pairwise
exchange acceptance, and repeat until T_max. The number of rungs becomes an output; every
neighbour pair sits at the same thermodynamic length and hence the same predicted acceptance.

Intended outcome: a `--ladder trailblaze` mode for `chimpss-fultonmarket` (plus a standalone
`chimpss-trailblaze` stage), an offline audit that quantifies how non-uniform the current ladders
are from data already on disk, and tests. Expected payoff: fewer rungs for the same minimum
acceptance (more sampling per rung per GPU-hour), no block-0 rebuild churn, uniform mixing.

## Theory being implemented (what the code must compute)

Notation: β = 1/(k_B T), H = U + pV (the sampler's reduced potential is u = β·H; NPT at 1 bar).

| Quantity | Formula | Source |
|---|---|---|
| Fisher metric on β | g(β) = Var_β(H) = σ_H² | Shenfeld eq. 3 |
| Metric on ln β | g(ln β) = C_V / k_B → constant C_V ⇒ geometric ladder | Shenfeld §IV, Fig. 1 |
| Length of a pair | ΔL ≈ σ_H · (β_k − β_{k+1}), σ_H = mean of the two rungs' σ | Shenfeld eq. 11 |
| Cumulative length | L(T) = ∫ σ_H(T') dT' / (k_B T'²) | from eq. 2 |
| Optimality | equidistant ΔL maximises total acceptance; −ΔL²/2 − ΔL/√2 ≤ ln A ≤ −ΔL²/2 | Shenfeld Thm 2, eq. 20 |
| Gaussian acceptance | A = erfc(ΔL/2) ⇒ ΔL* = 2·erfcinv(A*) (A*=0.5→0.95, 0.4→1.19, 0.35→1.32) | standard, inside eq. 20 bounds |
| Next rung (exact in β) | β_n = β_o − ΔL*/σ_H(T_o) ⇒ T_n = T_o / (1 − ΔL*·k_B·T_o/σ_H) | AlGDock `_next_BC_state` linearised |
| Pairwise acceptance from samples | ⟨min(1, exp(−(u_{o,n} + u_{n,o} − u_{o,o} − u_{n,n})))⟩ | AlGDock BC loop |
| Sufficiency | mean neighbour acceptance > 35 % puts the ladder in the "large K" regime where precision no longer depends on K | Nguyen & Minh §Results |

Existing code that already does this for AlGDock's temperature ("BC") process and is the
template to port (not import): `src/chimpss/algdock/initialization.py` `_tL_tensor` (line 449,
`std(U)/(R T²)`), the BC loop (lines 647–775: step, seed from previous state, simulate, measure
acceptance, reject-and-shrink by 1.25^n if below `min_repX_acc`, merge if > 0.99) and
`_next_BC_state` (line 851).

## Findings that constrain the design (verified in code)

- **Ladders can only grow once a block is saved.** `FultonMarketAnalysis._determine_interpolation_inds`
  (`analysis.py:647-658`) asserts every earlier block's ladder is a subset of the final ladder
  (temperatures rounded to 2 decimals); `_backfill` MBAR-fills the missing rungs. So trailblazing
  must fix the ladder *before block 0 is saved*; the existing insertion rule can stay as a safety net.
- **The ladder lives in three places**: `FultonMarket.__init__` (`fulton_market.py:66`), the resume
  path `_load_initial_args` (`:445-447`, reads `saved_variables/<n-1>/temperatures.npy`), and
  `Randolph.save_simulation` (`randolph.py:112`). On resume the saved ladder wins.
- **Reference state construction** (`fulton_market.py:392-395`):
  `ThermodynamicState(system, temperature=T[0], pressure=1 bar)`. The HMR system XMLs carry no
  barostat; openmmtools adds an isotropic `MonteCarloBarostat` (`states.py:1771`). The trailblazer
  must build its per-rung states the same way so it samples the same H.
- **Energies on disk are enough for an audit.** `energies.npy` is `u[iter, replica, state] = β_state·H(x_replica)`
  for every rung, and `states.npy` says which replica sat at which rung, so per-rung σ_H and the
  *exact* nearest-neighbour acceptance `min(1, exp(−(u[a,k+1]+u[b,k]−u[a,k]−u[b,k+1])))` are
  computable for every saved block without new simulation. Per-pair acceptance is otherwise
  only printed to stdout (`randolph.py:222`) and lost.
- **Pre-existing seeding issue**: `Randolph._build_simulation` passes `sampler_states=self.sampler_states[0]`
  (`randolph.py:181`); openmmtools then seeds every replica from that one configuration
  (`replicaexchange.py:251`). Every block > 0 restarts all replicas from the state-0 configuration.
  Per-rung seeds from the trailblazer are only useful if this is fixed (see Phase 4, optional).
- **CLI gaps**: `init_overlap_thresh` / `term_overlap_thresh` are not reachable from
  `chimpss-fultonmarket` (`cli/fultonmarket.py`). They must become flags because a trailblazed
  ladder at target acceptance A* < 0.5 would otherwise trigger insertions everywhere in block 0.
- **MPI**: rank-0-only work uses `@mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)`
  (`randolph.py:92`, `fulton_market.py` panel methods). The walk is serial and must run on rank 0
  with the ladder broadcast.
- **Tests**: unit tests run with `pytest -m "not slow and not gpu"`; openmm-dependent modules are
  gated with `pytest.importorskip("openmm")` (`tests/unit/test_fultonmarket/test_math.py`); CLI
  plumbing tests monkeypatch `sys.argv` (`tests/unit/test_cli/test_argument_plumbing.py`).
  No test runs a simulation; `tests/regression/test_fultonmarket_e2e.py` is stale and skips.
- Reference papers: neither PDF names "trailblazing". Shenfeld gives the metric, the equidistance
  theorem and the acceptance bounds; Nguyen & Minh use a geometric PT ladder and establish that
  > 35 % neighbour acceptance is sufficient. The sequential procedure itself is AlGDock's.

## What the existing data already says (measured during planning from `saved_variables/`)

Reduced potentials on disk give H = u[it, r, k]/β_k exactly (constant across k to 7.5e-8, the
float32 limit; statistics must be done in float64). All seven local runs were audited with
iteration 0 skipped:

| | NTSR1 rep0 (161 blocks) | NTSR1 rep3 (23) | 5HT2BR rep1 (6) |
|---|---|---|---|
| σ_H bottom → top (kJ/mol) | 1086 → 1365 | 1047 → 1346 | 1270 → 1347 |
| C_V/k_B bottom → top | 1.67e5 → 1.95e5 | 1.65e5 → 1.95e5 | noisy |
| ΔL per pair min / mean / max (CV) | 0.42 / 0.68 / 1.00 (0.22) | 0.40 / 0.61 / 1.06 (0.26) | 0.39 / 0.62 / 0.97 (0.26) |
| L_total | 73.6 | 72.8 | 77.4 |
| measured neighbour A min / mean / max | 0.50 / 0.65 / 0.78 | 0.52 / 0.68 / 0.79 | 0.49 / 0.67 / 0.81 |
| corr(erfc(ΔL/2), A_meas); mean abs dev | 0.995; 0.016 | 0.976; 0.017 | 0.935; 0.023 |
| Shenfeld bounds violated | 0 / 109 pairs | 0 / 119 | 0 / 125 |
| rungs needed at A* = 0.35 / 0.40 / 0.50 | 57 / 63 / 79 | 57 / 63 / 78 | 60 / 67 / 83 |

- C_V(T) rises ~17 % over 310–367 K, linear in T within noise. A constant-ratio geometric ladder
  would already be nearly uniform (ΔL CV 0.02); the observed 0.22–0.26 non-uniformity comes from
  `geometric_distribution` (top spacing 2.7× bottom ⇒ A 0.77 → 0.48 along the ladder) and from
  midpoint insertions (2:1 ΔL discontinuities, e.g. 5HT2BR rep1 pair 0).
- **Current ladders carry ~2× the rungs needed** for A* = 0.35–0.40. Respacing rep0's measured
  profile at A* = 0.4 gives K = 63 with reweighted "virtual" acceptance 0.41–0.49 on every pair.
- The Gaussian mapping `ΔL* = 2·erfcinv(A*)` is the right calibration for this engine: it reproduces
  the swap-all rates the engine logs (rep3 NetCDF min/mean/max 0.48/0.68/0.82 vs energy-derived
  0.52/0.68/0.79).
- Iterations within a block are independent (lag-1 autocorrelation of H ≈ −0.02); per-pair acceptance
  from one block is ±0.07, from a trailing 5 blocks ±0.03.
- **Restart transient**: iteration 0 of every block has identical H on every replica (−34 σ), then the
  deviation decays −0.76, −0.42, −0.27, … σ and reaches 0 only by iteration ~8. Every 17–19-iteration
  block discards all hot-rung configurations and spends ~8 iterations re-heating (see Phase 4). Any σ
  estimate must skip iteration 0; the 1–7 transient inflates within-block σ by ~3 %.
- Cost of a walk: ~120 ps of single-replica MD per rung (20 ps equilibration + 100 ps sampling)
  ⇒ ~7–8 ns for ~60 rungs ≈ 45–75 min on one GPU, less than one production block (2.76 h locally).

## Decisions (confirmed with the user)

- Default target neighbour acceptance **A* = 0.40** (ΔL* = 1.19), safety-net `min_acc = 0.30`.
- The per-replica seeding fix (`randolph.py:181`) **is in scope**, as its own commit (Phase 4).
- Campaign default stays **geometric**; trailblazing is opt-in per run (`PTRE_LADDER=geometric`
  in `config.sh`, `CHIMPSS_LADDER` override in `run_local.sh` / `generate_jobs.sh`).

## Design

Three layers, each usable alone:

1. **`src/chimpss/fultonmarket/trailblaze.py`** — pure-numpy metric/placement math + the sequential
   walk (`Trailblazer`) + an openmmtools single-rung sampler (`OpenMMRungSampler`). No openmm import
   at module top level so the math is unit-testable (same trick as `retro_convergence.py` tests).
2. **`src/chimpss/analysis/ladder.py`** — offline audit from `saved_variables/` (σ_H profile, exact
   per-pair acceptance, cumulative thermodynamic length, equal-length respacing) + CLI
   `chimpss-ladder-audit`. Validates the method on existing runs and gives a simulation-free
   ladder for new replicates of an already-run system.
3. **Integration** — `FultonMarket(ladder='geometric'|'trailblaze'|'file', ...)`, CLI flags, a
   standalone `chimpss-trailblaze` stage, threshold defaults that match the walk, campaign wiring.

### Layer 1: `trailblaze.py`

Pure helpers (numpy + `scipy.special.erfc/erfcinv` only):

| Function | Does |
|---|---|
| `KB_KJ_PER_MOL_K = 0.008314462618`, `beta(T)` | matches `openmmtools.constants.kB` |
| `target_spacing(A*)` / `predicted_acceptance(ΔL)` | `2·erfcinv(A*)` / `erfc(ΔL/2)` |
| `pair_spacing(T_o, T_n, σ_o, σ_n)` | `(β_o − β_n)·(σ_o+σ_n)/2` (Shenfeld eq. 11) |
| `next_temperature(T_o, σ_o, ΔL, T_max, step_scale, max_dT=10, min_dT=0.1)` | exact-in-β step `T_n = T_o/(1 − ΔL·step_scale·k_B T_o/σ_o)`; cap at `max_dT`; raise below `min_dT`; clamp to `T_max` and set `crossed`; snap to `T_max` if the remaining sliver is < half a step |
| `pair_acceptance(H_o, H_n, T_o, T_n)` | exact Metropolis mean over all (i, j) sample pairs `min(1, exp(−(β_o−β_n)(H_n[j]−H_o[i])))` (AlGDock's `exp(−u01−u10+u00+u11)`), subsampled above 250k pairs |
| `stationary_window(H)` | `(t0, g, Neff)` via `pymbar.timeseries.detect_equilibration` when importable, else a two-halves drift test |
| `ladder_from_sigma(T_min, T_max, sigma_fn, ΔL)` | deterministic "dry" sequential walk on a known σ(T): used by the walk's tests (the audit's `respace` is the integral form of the same placement) |
| `load_ladder_file(path)` | `.npy` or text, validated sorted/unique/≥ 2 |

(`KB`, `pair_spacing`, `target_spacing`, `predicted_acceptance` are thin re-exports of
`analysis/ladder.py`'s `R_KJ`, `pair_lengths`, `length_for_acceptance`, `predicted_acceptance`;
per-rung σ from saved blocks is `analysis.ladder.metric_profile`.)

`OpenMMRungSampler(system, dt_fs, pressure_bar=1, equil_ps=20, sample_ps=100, sample_interval_ps=1,
max_sample_ps=400, neff_min=50)` — callable `(T, seed SamplerState) → (H samples, last SamplerState, meta)`:
- Reference state built exactly as `FultonMarket._build_thermodynamic_states` (`ThermodynamicState(system, T, pressure=1 bar)`),
  deep-copied per rung with `.temperature = T` (what `ParallelTemperingSampler.create` does; also
  retunes the barostat, `states.py:1183`).
- First move: `mcmc.LangevinDynamicsMove(..., n_steps=equil, reassign_velocities=True)` so velocities
  are redrawn at T (a seed's stale velocities would otherwise be reused; `apply_to_context` only
  sets velocities the state carries). Then `sample_ps/sample_interval_ps` moves with
  `reassign_velocities=False`; after each, `H = state.reduced_potential(ss)·k_B T` is free
  (`update_from_context` caches U and V; `states.py:818-910`).
- Uses `cache.global_context_cache`, the same cache Randolph uses; the Context hash is
  temperature-independent so one GPU Context serves all rungs.
- Extends sampling in `sample_ps` chunks until `Neff ≥ neff_min` or `max_sample_ps`.

`Trailblazer(sampler, T_min, T_max, target_acc=0.4, min_acc=0.3, max_acc=0.75, shrink=1.25,
max_retries=6, max_rungs=400, save_seeds=True, out_dir)` — `run(initial_seed) → TrailblazeRecord`:
1. Sample at `T_min` from the input-state configuration; record σ.
2. Loop: `T_n = next_temperature(..., step_scale=shrink**(−stage))`; sample at `T_n` seeded from the
   previous rung's final `SamplerState`; `acc = pair_acceptance(H_prev, H_n, …)`.
   - `acc < min_acc` → reject, `stage += 1` (shrink ×1.25, as `_next_BC_state`'s `tL_tensor·1.25**pow`).
   - `acc > max_acc` and not crossed → reject, `stage −= 1` (symmetric stretch; replaces AlGDock's "merge").
   - else accept: persist `seeds/rung_XXX.npz` (positions, velocities, box, H samples) and append to
     `trailblaze.json` (atomic write via `.tmp.<pid>` + `os.replace`, the repo's own pattern); `stage = 0`.
   - After `max_retries` accept with a warning so the walk terminates; `max_rungs` → `RuntimeError`.
3. On `crossed`: assert uniqueness after rounding to 2 decimals (`analysis.py:101`), write
   `trailblaze/temperatures.npy`, mark `complete`.
Per-rung record: T, σ_H, mean H, n_samples, g, Neff, acc_est, acc_pred, realised ΔL, step_scale,
rejects, seed path, wall seconds. Resumable: an incomplete `trailblaze.json` resumes from its last
accepted rung's seed; a complete one returns immediately; changed inputs refuse unless `--force`.
Layout `<output_dir>/trailblaze/{trailblaze.json, temperatures.npy, seeds/}` — never under
`saved_variables/`, which the analysis globs and `int()`s.

### Layer 2: `analysis/ladder.py` + `chimpss-ladder-audit`

Sibling of `analysis/mixing.py`, same block-reading pattern (`mmap_mode='r'` on the three small
arrays, never `positions.npy`), numpy + `scipy.special` only. This module is the **single source of
truth** for the shared formulas (`R_KJ`, `betas`, `pair_lengths`, `predicted_acceptance`,
`length_for_acceptance`, `acceptance_bounds`); `trailblaze.py` imports them from here.

| Function | Does |
|---|---|
| `enthalpy_from_reduced(energies, temperatures)` | `H[t,r] = mean_s(u[t,r,s]/β_s)` in float64; raises if the columns disagree (non-PT energies) |
| `by_state(H, states)` | `Hk[t,k]` via fancy indexing (replaces the double loop in `FultonMarketAnalysis._reshape_array`) |
| `load_run(run_dir, blocks=None, skip=1) → LadderData` | per-block state-ordered H; skips blocks whose 2-decimal ladder differs from the final one (mirrors `mixing.concatenated_states`); masks non-finite rows |
| `metric_profile(temperatures, Hk_blocks, variance='within')` | per rung mean H, σ_H, C_V (k_B units), √g; within-block variance by default (immune to slow drift); also `sigma_total/sigma_within` and the per-iteration deviation profile |
| `smooth_sigma(T, σ, model='cv_linear')` | least-squares `C_V = a + bT`, `σ_fit = R T √(a+bT)`; cuts σ noise from 1–7 % to ~1 %; default for respacing when K ≥ 8 |
| `pair_acceptance_from_energies(energies, states, skip=1, mode='simultaneous'|'pooled')` | exact engine criterion in u-form: `min(1, exp(−(u[a,k+1]+u[b,k]−u[a,k]−u[b,k+1])))`; pooled mode uses all (t, t') pairs within a block |
| `pair_acceptance_from_enthalpy(...)` | H-form of the same; tested to agree with the u-form |
| `cumulative_length(T_grid, σ_grid)` | trapezoid of `σ/(R T²)` (checked: equals Σ pair_lengths to 4 digits on rep0) |
| `respace(T, σ, n_rungs | target_dL | target_acceptance, T_min, T_max, n_fine=4000, decimals=2)` | σ interpolated linear in T onto a fine grid, invert L(T) at equal arc length; forces exact endpoints; `K = ceil(L_total/ΔL*) + 1`; raises on a 2-decimal collision |
| `virtual_pair_acceptance(T_a, T_b, T, Hk_blocks)` | acceptance of any hypothetical pair by exponential reweighting from the nearest sampled rung (ESS ≥ 85 % at this density) — validates a proposed ladder before GPU time |
| `ladder_report(run_dir, targets=(0.35,0.4,0.5))`, `format_table` | per-rung and per-pair tables, summary (L_total, ΔL CV, A min/mean/max measured vs predicted, corr, K needed per target, C_V fit, restart-transient diagnostics), JSON-serialisable |
| `block_audit(run_dir, block, trailing=5)` | one block's per-pair A / ΔL / σ plus a trailing-5-block pooled A, for the in-run hook |

CLI `src/chimpss/cli/ladder_audit.py` (`chimpss-ladder-audit`, lazy imports inside `main()` like
`cli/run_stats.py`):
```
chimpss-ladder-audit <run_dir> [--blocks ...] [--skip 1] [--targets 0.35 0.4 0.5] [--smooth cv_linear|none]
    [--out PREFIX] [--tsv] [--respace --target_acceptance 0.4 | --n_rungs K] [--T_min] [--T_max] [--virtual]
```
Writes `<run>/analysis/ladder_audit.json` (+ `_pairs.tsv` via `run_stats.write_tsv`); `--respace`
writes `<prefix>_respaced.npy` (loadable with `--ladder file --temperatures_file`) and a JSON with
provenance, per-pair ΔL, predicted and (with `--virtual`) reweighted acceptance.

In-run hook: `FultonMarket._audit_ladder()` (`@mpiplus.on_single_node(rank=0, sync_nodes=True)`,
try/except + `printf` warning like `_evaluate_panel`) called right after `_save_sub_simulation()`;
writes `<run>/analysis/ladder/<block:04d>.json` and prints one line (min/mean/max A, pairs below
`term_overlap_thresh`). This preserves the per-pair acceptance that `randolph.py:222` currently
prints and loses.

### Layer 3: integration

`FultonMarket.__init__` gains trailing defaulted parameters (public API stays stable per CLAUDE.md):
`ladder='geometric'|'trailblaze'|'file'`, `temperatures_file=None` (implies `'file'`),
`trailblaze_kwargs=None`. `'file'`: ladder wins, warn when `n_replicates`/`T_min`/`T_max` disagree.
`'trailblaze'`: keep the geometric placeholder until `run()` knows `output_dir`.

`FultonMarket.run()`:
- `init_overlap_thresh` / `term_overlap_thresh` defaults become `None` = auto (0.5/0.35 for geometric
  and metadata-less file ladders, unchanged behaviour; the walk's `min_acc` for trailblazed ladders).
- New `_resolve_ladder()` after `_configure_experiment_parameters()`: at `sim_no == 0` call
  `_trailblaze_rank0()` (`@mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)`),
  which returns the cached ladder if `trailblaze.json` is complete, else walks. At `sim_no > 0` only the
  metadata (`min_acc`) is read; the ladder keeps coming from `saved_variables/<n-1>/temperatures.npy`.
  After resolution re-run `_set_init_positions`/`_set_init_box_vectors` so the per-replica lists match.
- Public `trailblaze(output_dir, dt, **kwargs)` so the standalone CLI and `run()` share one path.
- `_build_sampler_states` at `sim_no == 0`: if every `seeds/rung_*.npz` exists, build one `SamplerState`
  per rung from them; else the existing `from_context` path. (Requires Phase 4 to take effect.)

CLI (`src/chimpss/cli/fultonmarket.py`): `--ladder {geometric,trailblaze,file}`, `--temperatures_file`,
`--target_acc 0.4`, `--min_acc 0.3`, `--tb_equil_ps 20`, `--tb_sample_ps 100`, `--tb_sample_interval_ps 1`,
`--tb_max_rungs 400`, `--tb_no_seeds`, `--init_overlap_thresh None`, `--term_overlap_thresh None`;
`p.error` on `--temperatures_file` + `--ladder trailblaze`. New `src/chimpss/cli/trailblaze.py`
(`chimpss-trailblaze`): mode 1 runs only the walk into `<output_dir>/trailblaze/` (so stage 05 with
`--ladder trailblaze` finds the cache and never re-walks); mode 2 `--from_run RUN_DIR` is the
simulation-free respacing from saved energies (delegates to layer 2). Register in `pyproject.toml`
`[project.scripts]`; export `Trailblazer`, `ladder_from_sigma`, `sigma_from_block`, `load_ladder_file`
from `chimpss.fultonmarket`; add an automodule block to `docs/api/fultonmarket.rst`.

Campaign wiring (`/media/josephdb/Barracuda4Tb/Chimpss_V2/slurm/`, outside the git repo), opt-in:
`config.sh` gets `PTRE_LADDER=geometric`, `PTRE_TARGET_ACC=0.4`, `PTRE_MIN_ACC=0.3`, `PTRE_TB_*`
and loses the "insertion should never fire" comment (replaced by the measured 110–132 rungs);
`bin/run_local.sh` and `bin/generate_jobs.sh` honour a `CHIMPSS_LADDER` override the same way they
honour `CHIMPSS_CONVERGENCE`; `generate_jobs.sh` emits a `04b_trailblaze_<RUN_ID>.job` (1 GPU,
≤ 4 h, skipped if `trailblaze/temperatures.npy` exists) only for runs whose ladder is `trailblaze`,
and the 05 job then passes `--ladder trailblaze --target_acc --min_acc` and requires the cache file;
`README.md` stage list updated. Under MPI the in-run walk would idle the other GPUs for ~1 h, which
is why the standalone stage exists.

## Phases (one commit each, in order)

1. **Audit first (no simulation, de-risks everything)** — `analysis/ladder.py`, `chimpss-ladder-audit`,
   the in-run `_audit_ladder` hook, unit tests on synthetic Gaussian energies
   (`tests/unit/test_analysis/test_ladder.py`: synthetic PT blocks with known C_V, analytic
   `ΔL = √C_V (r²−1)/(2r)` for a ratio-r ladder, u-form vs H-form acceptance on a hand-built
   2-replica case, respacing an exponential-in-index ladder to ΔL CV < 1e-3, collision guard,
   mixed 17/19-iteration blocks). Run it on all seven local runs and keep the JSON/TSV outputs as the
   baseline the trailblazed runs are compared against.
2. **Math + walk** — `trailblaze.py` helpers, `Trailblazer`, record/resume, fake-sampler unit tests.
3. **Sampler + integration** — `OpenMMRungSampler`, `FultonMarket` ladder modes and threshold defaults,
   CLI flags, `chimpss-trailblaze`, plumbing tests, the `@slow @gpu` test on `test_data/RS_sets/RS1125`
   (45,635 atoms, state XML present, 2 fs).
4. **Per-replica seeding fix, its own commit** — `randolph.py:181`
   `sampler_states=self.sampler_states[0]` → `self.sampler_states`, plus the seed-aware
   `_build_sampler_states`. Verify with the Phase 1 audit on the first block of a new run: the
   iteration-0 identical-H signature and the ~8-iteration re-heating transient must be gone.
   Blast radius, to state in the commit message and `slurm/README.md`: every block > 0 of a run
   *resumed* under the new code starts each replica from its own configuration instead of the
   state-0 one, so its trajectory diverges from what the old code would have produced; cycle-0
   measured acceptance drops by up to ~1/n_iters because the identical-configuration first swap
   disappears (another reason the threshold defaults change in the same release).
   `_interpolate_states` already builds a full list, so insertion inherits the fix. Memory: ~440 MB
   per rank for 130 × 70k-atom sampler states.
5. **Campaign wiring (opt-in)** — `slurm/config.sh`, `generate_jobs.sh`, `run_local.sh`, `README.md`.

## Verification

```bash
PY=/home/josephdb/.local/share/mamba/envs/chimpss_v2/bin/python
cd /media/josephdb/Barracuda4Tb/Chimpss_V2/ChiMPSS
$PY -m pytest tests/unit/test_analysis/test_ladder.py tests/unit/test_fultonmarket/test_trailblaze_math.py \
    tests/unit/test_fultonmarket/test_trailblaze_walk.py tests/unit/test_cli -m "not slow and not gpu" -q
$PY -m ruff check src/chimpss/fultonmarket/trailblaze.py src/chimpss/analysis/ladder.py src/chimpss/cli/trailblaze.py

# Phase 1: audit existing runs (expect sigma_H ~1000 kJ/mol flat in T; bottom pairs ~0.8, top ~0.5;
# predicted vs measured acceptance on a line; ~58 rungs needed at A*=0.4)
chimpss-ladder-audit /media/josephdb/Barracuda4Tb/Chimpss_V2/runs/NTSR1_ML-301.hmr.rep3 --skip 1 --target_acceptance 0.4 --respace
chimpss-ladder-audit /media/josephdb/Barracuda4Tb/Chimpss_V2/runs/NTSR1_ML-301.hmr.rep0 --blocks 120:161 --skip 1 --target_acceptance 0.4

# Phase 3: GPU integration test (minutes), then a real walk (~1 h) and resume checks
$PY -m pytest tests/regression/test_trailblaze_gpu.py -m "slow and gpu" -v
chimpss-trailblaze <equil.pdb> prep/NTSR1/systems/NTSR1_ML-301.FG_HMR_system.xml runs/tb_NTSR1 \
    --input_state <state.xml> --T_min 310 --T_max 367 --timestep 3.5 --target_acc 0.4
#   expect 50-65 rungs, acc_est in [0.3, 0.75], sigma ~1000; rerun -> "cached" in seconds;
#   kill a fresh walk after ~5 rungs, rerun -> "resuming from rung 5", identical final ladder.

# One PTRE block on the walked ladder
chimpss-fultonmarket <same args> --ladder trailblaze --timestep 3.5 --iter_length 0.0175 --sim_length 25 --total_sim_time 50 --convergence jsd
grep -c "Inserting state" <log>                      # expect 0
grep "Exchange" <log> | tail -60 | awk '{print $NF}'  # median ~0.4, none < 0.3
$PY -c "import numpy as np; a=np.load('runs/tb_NTSR1/saved_variables/0/temperatures.npy'); b=np.load('runs/tb_NTSR1/trailblaze/temperatures.npy'); print(np.allclose(a,b), len(a))"
```
Compare per-replica ns per block against an existing run (25 ns / 58 ≈ 0.43 ns vs 25 / 110 ≈ 0.23 ns).

## Risks and mitigations

- **σ noise / non-stationarity after a temperature jump** (100 samples ⇒ ~7 % SE on σ ⇒ ±0.03 in
  acceptance; drift inflates σ and over-steps): `stationary_window` + `Neff ≥ 50` with adaptive
  extension to 400 ps, the exact pairwise acceptance check on real samples, and the shrink rule.
- **NPT volume relaxation**: pV ≈ 26 kJ/mol ≪ σ_H; 20 ps equilibration is ample for ~1 K steps;
  `max_dT=10` bounds any jump.
- **2-decimal rounding and midpoint insertion**: `min_dT=0.1` and a uniqueness assertion; at ~1 K
  spacing a midpoint insertion stays ≥ 0.45 K from its neighbours.
- **Swap-all statistics**: `_eval_acc_rates` is a permutation-ensemble Metropolis average, not a
  neighbour chain, but it agreed with the exact pairwise estimator (0.68 vs 0.64), so thresholds are
  comparable; cycle-0 block-0 rates are biased upward until Phase 4 lands.
- **Live panel `n_states=8`** was tuned for 0.5 K spacing; at ~1 K spacing the upper rungs carry even
  less 310 K weight. Harmless, but the `config.sh` comment should be re-derived after the audit.
- **Reference-paper mismatch**: neither PDF names "trailblazing"; the plan cites Shenfeld for the
  metric/equidistance/acceptance bounds, Nguyen & Minh for the ≥ 35 % sufficiency result, and
  AlGDock's `initialization.py` for the sequential procedure.

## Status (2026-10-04, end of day)

All five phases landed on `feat/thermodynamic-trailblazing` (seven commits), plus two
corrections found by the validation run:

- **σ pooling and the slope source.** Stepping on each rung's own σ (≈50 effective samples)
  scattered the spacing by ~10 % and provoked rejections; the step now uses σ pooled over the
  last five accepted rungs (`--tb_sigma_window`). The walk's variance-based σ also runs 5–10 %
  below the production metric, because one 100–300 ps trajectory misses the slow configurational
  variance a block samples across walkers; the enthalpy-slope estimate
  σ = √(k_B T² d⟨H⟩/dT) tracks production within 1–4 % and is available as
  `--tb_sigma_source slope` (default remains `pooled`, the validated setting).
- **Confidence-aware insertion.** The historical insertion test judged pairs on a single
  five-iteration cycle (SE 0.15–0.2) and split ten of 58 walked pairs whose readings had a
  median of 0.415 against the 0.30 floor; it is also why the geometric ladders grew to 110–132.
  Walked ladders now use `insertion_rule='ucb'` (insert only when mean + 2 SE across iterations
  is below threshold after ≥ 10 iterations); geometric and file ladders keep `'mean'`.

**Validation, NTSR1_ML-301.hmr.rep4** (`CHIMPSS_LADDER=trailblaze`, local RTX 5060 Ti):

| Quantity | Result |
|---|---|
| Walk | 59 rungs, 310→367 K, 75 min; 10 rejections; acceptance 0.31–0.51 (median 0.39) by the walk's measure |
| Reweighted prediction from rep0 / rep3 samples | mean 0.40, min 0.34–0.36, no pair < 0.30 |
| Block 0 (30 iterations, per-rung seeds, `ucb` rule) | no insertion; pooled neighbour acceptance mean 0.402, median 0.394, min 0.235, max 0.614; ΔL mean 1.226 (CV 0.14) |
| Restart signature at iteration 0 | +0.08 σ (every pre-fix run: −30 to −37 σ, ~8 iterations to re-heat) |
| Walk acceptance vs block-0 production acceptance | mean difference +0.004 |
| Ladder size vs the adaptive-ladder replicates | 59 vs 110 (rep0) and 120 (rep3) |

Open items: seven pairs read below 0.30 in block 0 (single-block SE ≈ 0.05); the trailing audit
(`analysis/ladder/<block>.json`, `pairs_below_threshold`) over five blocks (SE ≈ 0.03) is the
right basis for deciding whether any deserves a rung, since the `ucb` rule at 30 iterations can
only act on pairs below ≈ 0.18. A walk with `--tb_sigma_source slope` should land closer to
0.40 on every pair and is the recommended setting for the next replicate.
