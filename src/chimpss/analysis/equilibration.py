r"""Alternative equilibration-point detectors for PTRE production runs.

``FultonMarket._evaluate_stopping_criterion`` derives its equilibration point
from :func:`pymbar.timeseries.detect_equilibration`, which maximises the
effective sample size ``(T - t0) / g(t0)`` over candidate truncation points
(Chodera, *JCTC* 2016, **12**, 1799).  That objective is not guaranteed
unimodal: when the autocorrelation estimate is noisy it develops competing
local maxima and the reported ``t0`` jumps between them.  Clark, Cole and
Michel (*JCTC* 2025, **21**, 88) document exactly this, reporting "strongly
bimodal distributions of discard times" for strongly correlated series and
tracing it to dips in the maximum lag index.

NTSR1_ML-301.hmr.rep0 is a clean instance.  Across 78 blocks the discard
fraction occupied two populations -- 29 readings below 60% and 42 at or above
70%, with one in between -- and moved as much as 53 percentage points between
adjacent blocks.  Because ``t0`` feeds *both* the MBAR resample pool and the
post-equilibration comparison window, that instability propagates twice: it
decides how many frames the resample may draw from, and which earlier blocks
the convergence criterion must agree with.

This module implements alternatives so they can be replayed against completed
runs and compared against what the production criterion actually did.  Nothing
here is wired into the live stopping criterion; it is analysis-only.

Validation, on synthetic AR(1) series with a known truncation point grown
block by block (``tests/unit/test_analysis/test_equilibration.py`` covers the
same detectors on simpler inputs):

===============  ========  ==========  ==============
detector         \|bias\|   max jump    branch flips
===============  ========  ==========  ==============
``max_neff``       0.096       0.803          6 / 6
``mser``           0.044       0.297          1 / 6
``mser_5``         0.039       0.267          1 / 6
``mser_w5``        0.038       0.251          1 / 6
``mser_w50``       0.048       0.397          2 / 6
``mser_sqrt``      0.027       0.251          1 / 6
``rca``            0.297       0.833          6 / 6
``rca_stat``       0.502       0.899          6 / 6
===============  ========  ==========  ==============

"branch flips" counts trials showing a >0.25 swing in discard fraction between
adjacent blocks -- the production failure mode.  The incumbent shows it in
every trial; ``mser_sqrt`` is both the most accurate and among the most
stable, which is why it is the recommended replacement.

**The RCA implementations here are experimental and should not be used.**  A
faithful reading of Yang et al. requires a Gaussian-region test tuned to
free-energy derivative data; the sequential-test approximation used here is
badly biased and no more stable than the incumbent.  They are kept only so the
comparison is complete.

Detectors operating on a single series
--------------------------------------
``detect_max_neff``       the incumbent, wrapped for baseline comparison
``mser``                  White's marginal standard error rule, generalised
                          over autocorrelation treatment (window=0/5/'sqrt')
``reverse_cumulative``    reverse cumulative averaging, working backwards from
                          the converged end

Detectors using several independent chains
------------------------------------------
``gelman_rubin``          classic / split potential scale reduction factor
``rank_normalized_rhat``  rank-normalised, folded split-Rhat
``detect_equil_rhat``     earliest truncation at which Rhat falls below a
                          threshold

Stabilising wrappers, orthogonal to the choice of detector
----------------------------------------------------------
``monotone``              forbid ``t0`` from moving backwards
``freeze_after``          stop re-deriving ``t0`` past a burn-in of blocks
``trailing_window``       comparison window decoupled from ``t0`` entirely
``bound_pool``            cap ``t0`` so the MBAR pool never collapses

Series builders and drivers
---------------------------
``energy_series``         the average reduced energy the production code sees
``volume_series``         box volume, a cheaper thermodynamic alternative
``structural_series``     per-frame CA RMSD at the lowest rung -- the
                          observable the convergence criterion tests
``compare_detectors``     every detector applied to one series
``sweep_blocks``          replay every detector block-by-block over a run

References
----------
Chodera, J. D. *JCTC* 2016, **12**, 1799-1805.  doi:10.1021/acs.jctc.5b00784
Clark, F.; Cole, D. J.; Michel, J. *JCTC* 2025, **21**, 88-101.
    doi:10.1021/acs.jctc.4c01359
Yang, W.; Bitetti-Putzer, R.; Karplus, M. *J. Chem. Phys.* 2004, **120**, 2618.
White, K. P. *Simulation* 1997, **69**, 323-334.
Oliveira, F. L. et al. (pyMSER) *JCTC* 2024.  doi:10.1021/acs.jctc.4c00417
Gelman, A.; Rubin, D. B. *Statistical Science* 1992, **7**, 457-472.
Vehtari, A. et al. *Bayesian Analysis* 2021, **16**, 667-718.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    'detect_max_neff', 'mser', 'reverse_cumulative',
    'gelman_rubin', 'rank_normalized_rhat', 'detect_equil_rhat',
    'monotone', 'freeze_after', 'trailing_window', 'bound_pool',
    'energy_series', 'volume_series', 'structural_series',
    'compare_detectors', 'sweep_blocks', 'DETECTORS',
]


# ---------------------------------------------------------------------------
# Variance estimation
# ---------------------------------------------------------------------------

def long_run_variance(x: np.ndarray, window: int = 0) -> float:
    """Bartlett-windowed estimate of the long-run (autocorrelation-aware) variance.

    ``window=0`` returns the plain sample variance, which is what the original
    MSER assumes and which under-estimates the variance of the mean whenever
    the series is correlated.  Larger windows include more lags:

        sigma^2_w = gamma_0 + 2 * sum_{k=1..w} (1 - k/(w+1)) * gamma_k

    The Bartlett taper keeps the estimate non-negative for moderate ``w``; the
    result is clamped at zero regardless, because a truncated estimator can
    still go slightly negative on short, strongly anti-correlated series.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return float('nan')

    xc = x - x.mean()
    g0 = float(xc @ xc) / n
    if window <= 0 or g0 == 0.0:
        return max(g0, 0.0)

    w = int(min(window, n - 1))
    total = g0
    for k in range(1, w + 1):
        gk = float(xc[:-k] @ xc[k:]) / n
        total += 2.0 * (1.0 - k / (w + 1.0)) * gk
    return max(total, 0.0)


def _resolve_window(window: Union[int, str], n_retained: int) -> int:
    """Turn a window spec into an integer lag count for ``n_retained`` points."""
    if isinstance(window, str):
        if window == 'sqrt':
            return int(np.floor(np.sqrt(max(n_retained, 1))))
        raise ValueError(f"unknown window spec {window!r}; use an int or 'sqrt'")
    return int(window)


# ---------------------------------------------------------------------------
# Single-series detectors
# ---------------------------------------------------------------------------

def detect_max_neff(series: Sequence[float], **_) -> int:
    """The incumbent: maximise effective sample size (Chodera 2016).

    Thin wrapper over :func:`pymbar.timeseries.detect_equilibration`, present
    so the baseline can be compared on exactly the same footing as everything
    else in :data:`DETECTORS`.
    """
    from pymbar.timeseries import detect_equilibration
    t0, _, _ = detect_equilibration(np.asarray(series, dtype=np.float64))
    return int(t0)


def mser(series: Sequence[float],
         window: Union[int, str] = 0,
         batch: int = 1,
         max_fraction: float = 0.5,
         **_) -> int:
    """White's marginal standard error rule, generalised over autocorrelation.

    Truncate at the ``d`` minimising the estimated variance of the truncated
    mean, ``sigma^2(d) / (n - d)``.  The variants differ only in how
    ``sigma^2`` is estimated:

    ``window=0``       plain sample variance -- White's original MSER.  Cheap,
                       but biased toward truncating too early on correlated
                       data.
    ``batch=5``        MSER-5: average non-overlapping batches of 5 first,
                       which decorrelates the series crudely before the rule
                       is applied.
    ``window='sqrt'``  Bartlett window of size sqrt(n_retained).  This is the
                       variant Clark et al. recommend as the best compromise
                       between the early-truncation bias of the uncorrelated
                       estimator and the noisy, late, variable truncation of
                       initial-sequence estimators.

    ``max_fraction`` caps the search at a fraction of the series.  Without it
    the rule can select a ``d`` that retains a handful of points, where the
    variance of the mean is small simply because almost nothing is left.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0

    if batch > 1:
        n_batches = n // batch
        if n_batches < 4:
            return 0
        xb = x[:n_batches * batch].reshape(n_batches, batch).mean(axis=1)
    else:
        xb = x

    nb = xb.size
    d_max = max(1, int(nb * max_fraction))

    best_d, best_score = 0, float('inf')
    for d in range(d_max):
        tail = xb[d:]
        m = tail.size
        if m < 2:
            break
        w = _resolve_window(window, m)
        var = long_run_variance(tail, w)
        if not np.isfinite(var):
            continue
        score = var / m
        if score < best_score:
            best_score, best_d = score, d

    return int(best_d * batch)


def reverse_cumulative(series: Sequence[float],
                       alpha: float = 0.05,
                       min_tail_fraction: float = 0.10,
                       n_steps: int = 200,
                       test: str = 'normality',
                       n_confirm: int = 3,
                       **_) -> int:
    """Reverse cumulative averaging (Yang, Bitetti-Putzer & Karplus 2004).

    Rather than scanning forward from the start, grow the retained region
    *backwards* from the final frame and stop when the retained data stop
    looking like a single equilibrium distribution.  Because the converged
    portion of a trajectory is at the end, this is structurally less prone to
    latching onto an early local optimum than a forward max-Neff scan.

    ``test='normality'`` follows the paper most closely: the equilibrated
    region is assumed Gaussian, and the region is extended backwards until a
    D'Agostino-Pearson test rejects normality -- which is what absorbing a
    monotonic transient does to an otherwise Gaussian sample.

    ``test='stationarity'`` instead splits the retained region in half and
    rejects when the two halves' means differ by more than ``z_alpha``
    autocorrelation-aware standard errors.  This is more robust when the
    equilibrium distribution is genuinely non-Gaussian.

    Two guards keep the sequential scan honest.  The region grows in
    ``n_steps`` coarse increments rather than frame by frame, and a stop
    requires ``n_confirm`` *consecutive* rejections.  Without them the scan is
    a sequence of ~200 tests at level ``alpha``, so a spurious rejection near
    the converged end is close to certain and the method returns a truncation
    point just short of the final frame.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 20:
        return 0

    start = max(8, int(n * (1.0 - min_tail_fraction)))
    cuts = np.unique(np.linspace(start, 0, max(n_steps, 2)).astype(int))[::-1]

    t0 = int(cuts[0])
    consecutive = 0
    for cut in cuts:
        retained = x[int(cut):]
        if retained.size < 20:
            t0 = int(cut)
            continue

        if test == 'normality':
            from scipy.stats import normaltest
            try:
                p = float(normaltest(retained).pvalue)
            except Exception:
                p = 1.0
            reject = p < alpha
        elif test == 'stationarity':
            from scipy.stats import norm
            half = retained.size // 2
            a, b = retained[:half], retained[half:]
            w = _resolve_window('sqrt', retained.size)
            va, vb = long_run_variance(a, w), long_run_variance(b, w)
            se = np.sqrt(max(va, 0) / a.size + max(vb, 0) / b.size)
            reject = se > 0 and abs(a.mean() - b.mean()) / se > float(norm.ppf(1 - alpha / 2))
        else:
            raise ValueError(f"unknown test {test!r}; use 'normality' or 'stationarity'")

        if reject:
            consecutive += 1
            if consecutive >= n_confirm:
                break          # sustained rejection: this region is not equilibrated
            continue           # a lone rejection at level alpha is expected noise
        consecutive = 0
        t0 = int(cut)

    return int(t0)


# ---------------------------------------------------------------------------
# Multi-chain detectors
# ---------------------------------------------------------------------------

def _as_chain_matrix(chains: Sequence[Sequence[float]]) -> np.ndarray:
    """Stack chains to (m, n), truncating to the shortest."""
    arrs = [np.asarray(c, dtype=np.float64).ravel() for c in chains]
    if len(arrs) < 2:
        raise ValueError('at least two chains are required for Rhat')
    n = min(a.size for a in arrs)
    if n < 4:
        raise ValueError('chains are too short for Rhat')
    return np.vstack([a[:n] for a in arrs])


def _split(mat: np.ndarray) -> np.ndarray:
    """Split each chain in half, doubling m.  Split-Rhat detects within-chain trends."""
    m, n = mat.shape
    half = n // 2
    return np.vstack([mat[:, :half], mat[:, half:2 * half]])


def gelman_rubin(chains: Sequence[Sequence[float]], split: bool = True) -> float:
    """Potential scale reduction factor (Gelman & Rubin 1992).

    Compares within-chain to between-chain variance; converges to 1 from above
    as the chains become indistinguishable.  ``split=True`` halves each chain
    first, which is what makes the statistic sensitive to a chain that is
    still drifting rather than merely dispersed.

    NOTE for PTRE: the replicas *within* one run exchange coordinates and are
    strongly coupled, so they are not valid independent chains.  The valid
    units are the independent stochastic replicates (rep0/rep1/rep2), which
    the campaign design already provides.
    """
    mat = _as_chain_matrix(chains)
    if split:
        mat = _split(mat)
    m, n = mat.shape

    means = mat.mean(axis=1)
    variances = mat.var(axis=1, ddof=1)

    W = variances.mean()
    B = n * means.var(ddof=1)
    if W <= 0:
        return float('nan')

    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def rank_normalized_rhat(chains: Sequence[Sequence[float]]) -> float:
    """Rank-normalised, folded split-Rhat (Vehtari et al. 2021).

    Rank-normalising before computing Rhat makes the statistic robust to
    heavy tails and to chains with infinite variance; the *folded* version,
    computed on ``|x - median|``, additionally catches chains that agree in
    location but differ in scale.  The reported value is the larger of the
    two, which is the conservative choice.
    """
    from scipy.stats import norm, rankdata

    mat = _as_chain_matrix(chains)

    def _rank_norm(a: np.ndarray) -> np.ndarray:
        flat = rankdata(a.ravel())
        z = norm.ppf((flat - 0.375) / (flat.size - 2 * 0.375 + 1.0))
        return z.reshape(a.shape)

    plain = gelman_rubin(_rank_norm(mat), split=True)
    folded = gelman_rubin(_rank_norm(np.abs(mat - np.median(mat))), split=True)
    return float(np.nanmax([plain, folded]))


def detect_equil_rhat(chains: Sequence[Sequence[float]],
                      threshold: float = 1.01,
                      n_candidates: int = 40,
                      rank_normalized: bool = True,
                      max_fraction: float = 0.75) -> int:
    """Earliest truncation point at which Rhat drops below ``threshold``.

    Scans candidate ``t0`` values and returns the first at which the retained
    portion of every chain is mutually consistent.  Returns ``-1`` when the
    threshold is never met, which is itself informative: it means the
    replicates have not converged to a common distribution at any truncation.
    """
    mat = _as_chain_matrix(chains)
    n = mat.shape[1]
    stat = rank_normalized_rhat if rank_normalized else gelman_rubin
    limit = int(n * max_fraction)

    for t0 in np.unique(np.linspace(0, limit, n_candidates).astype(int)):
        retained = mat[:, int(t0):]
        if retained.shape[1] < 8:
            break
        try:
            if stat(retained) < threshold:
                return int(t0)
        except ValueError:
            continue
    return -1


# ---------------------------------------------------------------------------
# Stabilising wrappers -- orthogonal to the detector
# ---------------------------------------------------------------------------

def monotone(t0_series: Iterable[int]) -> List[int]:
    """Forbid the equilibration point from moving backwards.

    A running maximum.  This alone removes the failure mode where a low
    reading widens the comparison window and re-admits blocks that had
    already aged out -- the criterion has no ratchet without it, so progress
    made while the discard is high is handed back when it drops.
    """
    out, best = [], 0
    for t0 in t0_series:
        best = max(best, int(t0))
        out.append(best)
    return out


def bound_pool(t0: int, n_frames: int, min_pool: int) -> int:
    """Cap ``t0`` so the retained pool never falls below ``min_pool`` frames.

    Measured on NTSR1_ML-301.hmr.rep0, pool size is what predicts an MBAR
    weight collapse -- not replica mixing, which was tested and refuted.  The
    five blocks whose resample drew 1-10 distinct configurations from 1000
    draws had pools of 1-29 frames against a healthy mean of 106
    (Spearman(pool, unique) = 0.798, p = 4e-18; Mann-Whitney pool(collapsed) <
    pool(healthy) p = 1.6e-4).  A pool below ~35 frames flagged every collapse
    with no false negatives.

    Capping ``t0`` is therefore a more direct guard than rejecting the resample
    afterwards on effective sample size: it is knowable *before* the MBAR solve
    is attempted, it prevents the failure rather than detecting it, and the
    quantity it bounds is the one that actually causes the problem.  The cost
    is retaining a few frames that a late estimate would have discarded, which
    is the cheaper error -- excluding good frames inflates variance and
    ill-conditions the solve, while including slightly unequilibrated ones adds
    a little bias.

    The cap loosens as the trajectory grows, so it constrains only the early
    blocks and the excursions where the detector jumps to a late branch.
    """
    if min_pool is None or min_pool <= 0:
        return int(t0)
    return int(min(int(t0), max(0, int(n_frames) - int(min_pool))))


def freeze_after(t0_series: Iterable[int], n_blocks: int) -> List[int]:
    """Stop re-deriving ``t0`` once ``n_blocks`` have been observed.

    Re-estimating equilibration from scratch every block buys little on a run
    that will last tens of blocks and costs all of the stability.
    """
    out, frozen = [], None
    for i, t0 in enumerate(t0_series):
        if i == n_blocks:
            frozen = int(t0)
        out.append(int(t0) if frozen is None else frozen)
    return out


def trailing_window(sim_no: int, fraction: float = 0.25) -> int:
    """First block of a comparison window decoupled from ``t0`` entirely.

    Returns ``first_valid_sim_no`` as a fixed trailing fraction of the blocks
    completed so far.  Because it does not read the equilibration estimate,
    the window advances monotonically and the equilibration gate stops
    fighting the matrix checks through a shared variable.
    """
    return int(sim_no * (1.0 - fraction))


# ---------------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------------

def energy_series(run_dir: str, pdb: str, skip: int = 10) -> np.ndarray:
    """The average reduced energy exactly as the production criterion sees it.

    Delegates to :class:`~chimpss.fultonmarket.analysis.FultonMarketAnalysis`
    rather than re-reading ``energies.npy`` directly, because ``_backfill``
    expands each block's energy matrix to the final ladder shape before the
    average is taken.  A naive concatenation of the per-block arrays does not
    reproduce the same series.
    """
    from chimpss.fultonmarket.analysis import FultonMarketAnalysis
    analyzer = FultonMarketAnalysis(input_dir=run_dir, pdb=pdb, skip=skip,
                                    verbosity='none')
    return np.asarray(analyzer.get_average_energy(), dtype=np.float64)


def volume_series(run_dir: str, skip: int = 10) -> np.ndarray:
    """Box volume per frame, pooled over blocks.

    A far cheaper thermodynamic observable than the energy: it reads only the
    small ``box_vectors.npy`` arrays and needs no ladder backfill.  Useful as
    an independent check on whether a detector's instability is a property of
    the method or of the particular series.
    """
    vols = []
    for d in _block_dirs(run_dir):
        path = os.path.join(d, 'box_vectors.npy')
        if not os.path.exists(path):
            continue
        bv = np.asarray(np.load(path, mmap_mode='r'))[skip:]
        # (frames, replicas, 3, 3) -> per-frame mean over replicas of det(box)
        if bv.ndim == 4:
            v = np.abs(np.linalg.det(bv)).mean(axis=1)
        elif bv.ndim == 3:
            v = np.abs(np.linalg.det(bv))
        else:
            continue
        vols.append(np.asarray(v, dtype=np.float64).ravel())
    return np.concatenate(vols) if vols else np.array([])


def structural_series(run_dir: str, topology: str, skip: int = 10,
                      state: int = 0, cache: bool = True,
                      _printf=print) -> np.ndarray:
    """Per-frame alpha-carbon RMSD of the replica at ``state``, time-ordered.

    The production criterion detects equilibration on the replica-averaged
    reduced potential energy, but tests the *structural* distance matrices.
    Energy equilibrates in nanoseconds; a conformational ensemble on a 110-rung
    ladder plainly does not, so the two are not interchangeable.  This builds
    the observable the criterion actually cares about, so any detector in
    :data:`DETECTORS` can be run on it via ``sweep_blocks(series_fn=...)``.

    ``states.npy`` records which replica occupies each thermodynamic state at
    each iteration, so the trajectory of the *lowest rung* -- the ensemble the
    analysis reweights to -- can be followed even though replicas permute.
    RMSD is measured against the final frame's structure after superposition,
    which makes the transient a decay toward zero.

    Reading is by memmap with alpha-carbon slicing: ``positions.npy`` is ~1.9 GB
    per block but the alpha carbons are a few hundred atoms, so the result is a
    few kB.  The full pass is one-time and cached under ``<run_dir>/analysis/``.
    """
    import mdtraj as md

    out_path = os.path.join(run_dir, 'analysis', f'structural_series_state{state}.npy')
    if cache and os.path.exists(out_path):
        return np.load(out_path)

    _top = md.load(topology)
    ca = _top.topology.select('name CA')
    n_atoms_total = _top.n_atoms
    frames: List[np.ndarray] = []
    for d in _block_dirs(run_dir):
        ppath = os.path.join(d, 'positions.npy')
        spath = os.path.join(d, 'states.npy')
        if not (os.path.exists(ppath) and os.path.exists(spath)):
            continue
        st = np.load(spath)
        tpath = os.path.join(d, 'temperatures.npy')
        n_replicas = int(np.load(tpath).size) if os.path.exists(tpath) else st.shape[1]
        pos = _load_positions(ppath, st.shape[0], n_replicas, n_atoms_total)
        if pos is None:
            _printf(f'  block {os.path.basename(d)}: positions unreadable, skipped')
            continue
        n_it = min(st.shape[0], pos.shape[0])
        for it in range(skip, n_it):
            hit = np.where(st[it] == state)[0]
            if hit.size == 0:
                continue
            frames.append(np.asarray(pos[it, hit[0]][ca], dtype=np.float32))
        del pos
        _printf(f'  block {os.path.basename(d)}: {len(frames)} frames so far')

    if not frames:
        return np.array([])

    X = np.stack(frames)
    ref = X[-1]
    series = np.array([_rmsd_after_superposition(x, ref) for x in X], dtype=np.float64)

    if cache:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, series)
    return series


def _load_positions(path: str, n_frames: int, n_replicas: int, n_atoms: int):
    """Memory-map ``positions.npy``, which is raw float32 with no header.

    Despite the extension these files carry no NPY magic and are not pickles --
    ``Randolph.save_simulation`` writes the buffer directly, so the shape lives
    only in the sibling arrays.  ``np.load`` misreports this as "pickled
    (object) data"; ``FultonMarket._load_initial_args`` works around it the same
    way, with an explicit-shape ``np.memmap``.  Verified against block 0:
    1,906,155,240 bytes == 19 x 110 x 76003 x 3 x float32 exactly.

    Falls back to a plain ``np.load`` first, so a properly-formed NPY written by
    some other path still works.
    """
    try:
        return np.load(path, mmap_mode='r')
    except Exception:
        pass
    itemsize = np.dtype('float32').itemsize
    expected = n_frames * n_replicas * n_atoms * 3 * itemsize
    actual = os.path.getsize(path)
    if actual < expected:
        # a truncated block: keep whatever whole frames are present
        n_frames = actual // (n_replicas * n_atoms * 3 * itemsize)
        if n_frames == 0:
            return None
    try:
        return np.memmap(path, mode='r', dtype='float32',
                         shape=(n_frames, n_replicas, n_atoms, 3))
    except Exception:
        return None


def _rmsd_after_superposition(x: np.ndarray, ref: np.ndarray) -> float:
    """RMSD between two (n_atoms, 3) frames after optimal Kabsch superposition."""
    a = x - x.mean(axis=0)
    b = ref - ref.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ r - b) ** 2, axis=1))))


def _block_dirs(run_dir: str) -> List[str]:
    base = os.path.join(run_dir, 'saved_variables')
    names = sorted((d for d in os.listdir(base) if d.isdigit()), key=int)
    return [os.path.join(base, d) for d in names]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

#: Named single-series detectors, for sweeps and reports.
DETECTORS: Dict[str, Callable[..., int]] = {
    'max_neff':        detect_max_neff,                                    # incumbent
    'mser':            lambda s, **k: mser(s, window=0, **k),              # White 1997
    'mser_5':          lambda s, **k: mser(s, window=0, batch=5, **k),     # MSER-5
    'mser_w5':         lambda s, **k: mser(s, window=5, **k),
    'mser_w50':        lambda s, **k: mser(s, window=50, **k),
    'mser_sqrt':       lambda s, **k: mser(s, window='sqrt', **k),         # Clark 2025
    # EXPERIMENTAL -- see the caveat in the module docstring.  On synthetic
    # series with a known truncation point both RCA variants show large bias
    # (0.30 / 0.50 in discard fraction) and flip branches in 6 of 6 trials.
    # Retained for comparison; do NOT use for production truncation.
    'rca':             reverse_cumulative,                                 # Yang 2004
    'rca_stat':        lambda s, **k: reverse_cumulative(s, test='stationarity', **k),
}


def compare_detectors(series: Sequence[float],
                      detectors: Optional[Dict[str, Callable]] = None) -> Dict[str, dict]:
    """Run every detector on one series and report ``t0`` and the discard fraction."""
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    out = {}
    for name, fn in (detectors or DETECTORS).items():
        try:
            t0 = int(fn(x))
            out[name] = {'t0': t0, 'discard': t0 / n if n else float('nan'), 'error': None}
        except Exception as exc:                     # a detector failing is data, not fatal
            out[name] = {'t0': None, 'discard': None, 'error': f'{type(exc).__name__}: {exc}'}
    return out


def sweep_blocks(run_dir: str,
                 pdb: str,
                 skip: int = 10,
                 detectors: Optional[Dict[str, Callable]] = None,
                 faithful: bool = False,
                 series_fn: Optional[Callable[[str, str, int], np.ndarray]] = None,
                 max_blocks: Optional[int] = None) -> dict:
    """Replay every detector block-by-block over a completed run.

    For each block ``N`` this reconstructs the series as it stood when block
    ``N`` was evaluated and asks every detector what it would have chosen --
    directly comparable to the ``Currently would discard ...`` lines in the
    production log.

    ``faithful=True`` re-instantiates the analyzer per block, reproducing what
    the production code saw exactly (``_backfill`` reshapes earlier blocks in
    light of the final ladder, so a prefix of the full series is not quite the
    same object).  It costs O(N^2) reads.  ``faithful=False`` slices prefixes
    of the full series, which is far cheaper and usually indistinguishable.

    Returns a dict with the per-block ``t0`` for each detector plus the
    monotone-constrained variant of each, ready to tabulate.
    """
    detectors = detectors or DETECTORS
    builder = series_fn or (lambda rd, p, sk: energy_series(rd, p, skip=sk))
    n_blocks = len(_block_dirs(run_dir))
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)

    per_block: Dict[str, List[Optional[int]]] = {k: [] for k in detectors}
    lengths: List[int] = []

    if faithful:
        import shutil
        import tempfile
        base = os.path.join(run_dir, 'saved_variables')
        for n in range(1, n_blocks + 1):
            tmp = tempfile.mkdtemp(prefix='chimpss_sweep_')
            sv = os.path.join(tmp, 'saved_variables')
            os.makedirs(sv)
            for b in range(n):
                os.symlink(os.path.join(base, str(b)), os.path.join(sv, str(b)))
            try:
                s = builder(tmp, pdb, skip)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            lengths.append(int(np.size(s)))
            for name, fn in detectors.items():
                try:
                    per_block[name].append(int(fn(s)))
                except Exception:
                    per_block[name].append(None)
    else:
        full = builder(run_dir, pdb, skip)
        per = max(1, int(np.size(full) / max(n_blocks, 1)))
        for n in range(1, n_blocks + 1):
            s = full[:min(n * per, np.size(full))]
            lengths.append(int(np.size(s)))
            for name, fn in detectors.items():
                try:
                    per_block[name].append(int(fn(s)))
                except Exception:
                    per_block[name].append(None)

    result = {'n_blocks': n_blocks, 'series_length': lengths, 't0': per_block,
              'discard': {}, 't0_monotone': {}}
    for name, seq in per_block.items():
        result['discard'][name] = [
            (t / L if (t is not None and L) else None) for t, L in zip(seq, lengths)
        ]
        clean = [t if t is not None else 0 for t in seq]
        result['t0_monotone'][name] = monotone(clean)
    return result
