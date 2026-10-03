"""Convergence panel: effect sizes against physical tolerances, on independent halves.

What the NTSR1 rep0 audit (4,025 ns, 161 blocks) established, and what this
module is built to avoid:

* The production statistic -- JSD between the frame-frame distance histograms
  of two MBAR resamples -- is a monotone function of how much *source data* the
  two resamples share.  At >90 % pool overlap the median is 0.02-0.05 and
  92-99 % of pairs clear 0.10; at zero overlap the median is 0.17-0.24 and
  0-28 % clear it, with no dependence on time separation.  It therefore reports
  the equilibration detector's t0 trajectory, not the ensemble, and can be met
  by independent data at no run length while t0 tracks the end of the run.
* Significance tests of stationarity (Kolmogorov-Smirnov on halves, OLS slope)
  applied to block means ignore autocorrelation (tau = 4-24 blocks, so 161
  blocks are 7-37 independent samples) and, being null-rejection tests, become
  *harder* to satisfy the longer the run.
* k-means populations partition one basin into cells 1.7 A apart on an ensemble
  1.5 A wide; the resulting "population" changes are cell-boundary crossings.

Every gate here compares an effect size to a tolerance in physical units, on
the two contiguous halves of the equilibrated run, with a block bootstrap for
the interval and an interleaved-halves split for the noise floor.  Tolerances
come from experiment: the RMSD between structures of the same receptor in the
same activation state over the same conserved atoms (0.69 A median across the
GPCRdb training set, 0.92 A for NTSR1), and the spread of those structures
along each principal component.  More sampling narrows the intervals, so a
converged run becomes easier, not harder, to certify.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from chimpss.analysis.pca_equilibration import (NM_TO_ANGSTROM, PanGPCRModel,
                                                _block_dirs, _load_positions,
                                                kabsch_superpose, resolve_atoms)

KT_310_KCAL = 0.6163          # k_B * 310 K in kcal/mol
BIMODALITY_CUTOFF = 0.555     # Pfister et al. 2013; uniform distribution scores 5/9
MIN_BLOCKS_PER_HALF = 8

# ---------------------------------------------------------------------------
# Series statistics
# ---------------------------------------------------------------------------

def acf(x: np.ndarray, maxlag: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    maxlag = n // 2 if maxlag is None else min(maxlag, n - 1)
    v = float(np.dot(x, x)) / n
    if v == 0:
        return np.ones(maxlag + 1)
    return np.array([np.dot(x[:n - k], x[k:]) / n / v for k in range(maxlag + 1)])


def tau_int(x: np.ndarray) -> float:
    """Integrated autocorrelation time by Geyer's initial positive sequence.

    Sums pairs of autocorrelations while each pair is positive, which is
    consistent for reversible chains and does not need a window parameter.
    Returns 1.0 for white noise; ``n / tau_int`` is the effective sample size.
    """
    r = acf(x)
    tau, m = -1.0, 0
    while 2 * m + 1 < r.size:
        g = r[2 * m] + r[2 * m + 1]
        if g <= 0:
            break
        tau += 2 * g
        m += 1
    return max(tau, 1.0)


def ols_corrected(y: np.ndarray) -> Dict[str, float]:
    """OLS slope against index with the standard error inflated by sqrt(tau).

    ``p_naive`` is what ``scipy.stats.linregress`` reports; ``p_corr`` uses the
    autocorrelation-corrected error and ``n_eff - 2`` degrees of freedom.
    """
    y = np.asarray(y, float)
    n = y.size
    x = np.arange(n, dtype=float)
    res = stats.linregress(x, y)
    resid = y - (res.intercept + res.slope * x)
    tau = tau_int(resid)
    neff = n / tau
    se_c = res.stderr * np.sqrt(tau)
    df = max(neff - 2.0, 1.0)
    t = res.slope / se_c if se_c > 0 else 0.0
    return dict(slope=float(res.slope), p_naive=float(res.pvalue), tau=float(tau),
                neff=float(neff), se_corr=float(se_c),
                p_corr=float(2 * stats.t.sf(abs(t), df)), df=float(df))


def rhat_split(y: np.ndarray, m: int = 4) -> float:
    """Gelman-Rubin R-hat treating ``m`` consecutive segments as chains."""
    y = np.asarray(y, float)
    n = (y.size // m) * m
    if n < 2 * m:
        return float('nan')
    ch = y[:n].reshape(m, -1)
    L = ch.shape[1]
    W = ch.var(axis=1, ddof=1).mean()
    B = L * ch.mean(axis=1).var(ddof=1)
    if W <= 0:
        return float('nan')
    return float(np.sqrt(((L - 1) / L * W + B / L) / W))


def fp_block_se(x: np.ndarray) -> Tuple[List[int], List[float], float]:
    """Flyvbjerg-Petersen blocking: standard error of the mean vs block size.

    Returns ``(sizes, se, plateau)``; the plateau is the mean of the last three
    points and is the honest standard error of the series mean.
    """
    x = np.asarray(x, float)
    n = x.size
    sizes, se = [], []
    j = 0
    while n // (2 ** j) >= 8:
        L = 2 ** j
        nb = n // L
        bm = x[:nb * L].reshape(nb, L).mean(axis=1)
        sizes.append(L)
        se.append(float(bm.std(ddof=1) / np.sqrt(nb)))
        j += 1
    plateau = float(np.mean(se[-3:])) if len(se) >= 3 else (se[-1] if se else float('nan'))
    return sizes, se, plateau


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.wasserstein_distance(a, b))


def jsd_1d(a: np.ndarray, b: np.ndarray, bins) -> float:
    pa, _ = np.histogram(a, bins=bins)
    pb, _ = np.histogram(b, bins=bins)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    return float(np.sqrt(0.5 * kl(pa, m) + 0.5 * kl(pb, m)))


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient; > 0.555 suggests more than one mode."""
    x = np.asarray(x, float)
    n = x.size
    g = stats.skew(x)
    k = stats.kurtosis(x)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _fingerprint(model: PanGPCRModel) -> str:
    key = f"{model.selection}|{model.n_atoms}|{'|'.join(model.conserved_bw)}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def superpose_frames(coords_nm: np.ndarray, model: PanGPCRModel) -> np.ndarray:
    """Every frame superposed on the model reference, in angstroms."""
    x = np.asarray(coords_nm, np.float64) * NM_TO_ANGSTROM
    return np.stack([kabsch_superpose(f, model.ref_positions, model.ref_masses) for f in x])


def project_superposed(X: np.ndarray, model: PanGPCRModel, n_pc: int = 10) -> np.ndarray:
    return np.asarray(model.pca.transform(X.reshape(X.shape[0], -1))[:, :n_pc], np.float64)


def block_mbar_weights(u: np.ndarray, states: np.ndarray, n_states: int,
                       target: int = 0) -> Tuple[np.ndarray, np.ndarray, bool]:
    """MBAR weights of one block's frames at the target rung.

    ``u[i, k]`` is the reduced potential of frame ``i`` at rung ``k`` for the
    ``n_states`` lowest rungs; ``states[i]`` the rung frame ``i`` was sampled
    at.  Solving MBAR over these rungs alone is exact for an estimate built
    from these frames, and on the campaign ladder the rungs above the fifth
    carry 0.7 % of the full-ladder weight at 310 K.  Returns
    ``(weights, f_k, ok)``; on a solver failure the weights fall back to the
    frames sampled at the target rung, uniformly, and ``ok`` is False.
    """
    u = np.asarray(u, np.float64)
    states = np.asarray(states, np.int64)
    n = u.shape[0]
    N_k = np.bincount(states, minlength=n_states)[:n_states]
    fallback = np.where(states == target, 1.0, 0.0)
    fallback = fallback / fallback.sum() if fallback.sum() else np.full(n, 1.0 / n)
    if n == 0 or N_k[target] == 0:
        return fallback, np.full(n_states, np.nan), False
    try:
        from pymbar import MBAR
        mbar = MBAR(u.T, N_k, initialize='BAR')
        W = np.asarray(mbar.weights())                     # (n, K)
        w = W[:, target]
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('non-finite weights')
        w = w / w.sum()
        f_k = np.asarray(mbar.f_k, np.float64) if hasattr(mbar, 'f_k') else np.full(n_states, np.nan)
        return w, f_k, True
    except Exception:
        return fallback, np.full(n_states, np.nan), False


def normalize_block_weights(blocks: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-frame weights that sum to one within every block.

    A block is a fixed interval of wall-clock time over which the whole
    ensemble was sampled, so blocks carry equal weight; within a block the
    MBAR weights (or, without them, a uniform share) define the 310 K
    ensemble.  Every statistic here is a mean over blocks of a weighted mean
    within blocks.
    """
    blocks = np.asarray(blocks)
    out = np.ones(blocks.size, np.float64) if w is None else np.asarray(w, np.float64).copy()
    for b in np.unique(blocks):
        m = blocks == b
        tot = out[m].sum()
        out[m] = out[m] / tot if tot > 0 else 1.0 / m.sum()
    return out


def _set_weights(blocks: np.ndarray, w: np.ndarray, bl: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Frame indices in the blocks ``bl`` and their weights, summing to one."""
    m = np.isin(blocks, bl)
    idx = np.where(m)[0]
    ww = w[idx] / len(np.unique(blocks[idx])) if idx.size else w[idx]
    return idx, ww


def _wmean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.tensordot(w, X, axes=(0, 0)) / w.sum()


def _wdraw(idx: np.ndarray, w: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(idx, size=n, replace=True, p=w / w.sum())


# ---------------------------------------------------------------------------
# Tolerances from experiment
# ---------------------------------------------------------------------------

@dataclass
class Tolerances:
    same_state_rmsd: float                 # A; receptor-specific when available
    family_same_state_rmsd: float          # A; median over all (protein, state) groups
    diff_state_rmsd: float                 # A; same protein, different state
    centroids: Dict[str, np.ndarray]       # state -> (n_pc,)
    within_state_sd: Dict[str, np.ndarray]  # state -> (n_pc,)
    receptor: Optional[str] = None
    receptor_n_structures: int = 0
    receptor_groups: List[dict] = field(default_factory=list)
    state_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def states(self) -> List[str]:
        return list(self.centroids)


def experimental_tolerances(prefix: str, receptor: Optional[str] = None,
                            n_pc: int = 10, min_pairs: int = 3) -> Tolerances:
    """Structural reproducibility and state geometry from the training set.

    ``receptor`` is the GPCRdb protein slug (``ntr1_human``).  The receptor's
    own same-state RMSD is used when it has at least ``min_pairs`` structure
    pairs in one state; otherwise the family median stands in, which is the
    conservative choice (the receptor-specific values run 0.3-1.2 A).
    """
    with open(f'{prefix}_meta.json') as fh:
        meta = json.load(fh)
    codes = meta['codes_retained']
    sm = meta['structure_meta']
    T = np.load(f'{prefix}_train_projections.npy')     # full basis: distances preserved
    n_atoms = int(meta['n_atoms'])
    state_of = np.array([str(sm[c].get('state')) for c in codes])
    prot_of = np.array([str(sm[c].get('protein')) for c in codes])

    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, (p, s) in enumerate(zip(prot_of, state_of)):
        groups.setdefault((p, s), []).append(i)

    def med_rmsd(idx_a, idx_b=None):
        pairs = (combinations(idx_a, 2) if idx_b is None
                 else ((i, j) for i in idx_a for j in idx_b))
        d = [np.linalg.norm(T[i] - T[j]) / np.sqrt(n_atoms) for i, j in pairs]
        return (float(np.median(d)), len(d)) if d else (float('nan'), 0)

    same = [med_rmsd(idx)[0] for idx in groups.values() if len(idx) >= 2]
    family_same = float(np.median(same))
    diff = []
    for p in set(prot_of):
        ks = [k for k in groups if k[0] == p]
        for a, b in combinations(ks, 2):
            diff.append(med_rmsd(groups[a], groups[b])[0])
    diff_state = float(np.median(diff)) if diff else float('nan')

    rec_groups, rec_n, rec_rmsd = [], 0, float('nan')
    if receptor:
        for (p, s), idx in groups.items():
            if p == receptor:
                m, npairs = med_rmsd(idx)
                rec_groups.append(dict(state=s, n=len(idx), median_rmsd=m, n_pairs=npairs))
                rec_n += len(idx)
        usable = [g for g in rec_groups if g['n_pairs'] >= min_pairs]
        if usable:
            rec_rmsd = float(np.median([g['median_rmsd'] for g in usable]))

    centroids, within, counts = {}, {}, {}
    for s in ('Inactive', 'Intermediate', 'Active'):
        m = state_of == s
        if m.sum() >= 2:
            centroids[s] = T[m, :n_pc].mean(0)
            within[s] = T[m, :n_pc].std(0)
            counts[s] = int(m.sum())
    return Tolerances(same_state_rmsd=rec_rmsd if np.isfinite(rec_rmsd) else family_same,
                      family_same_state_rmsd=family_same, diff_state_rmsd=diff_state,
                      centroids=centroids, within_state_sd=within, receptor=receptor,
                      receptor_n_structures=rec_n, receptor_groups=rec_groups,
                      state_counts=counts)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _halves(ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mid = ub.size // 2
    return ub[:mid], ub[mid:]


def _interleaved(ub: np.ndarray, chunk: int, rng: np.random.Generator
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Alternate chunks of ``chunk`` blocks between two sets, random phase.

    Two sets built this way sample the same time span, so any difference
    between them is finite-sample noise plus autocorrelation shorter than
    ``chunk`` -- the floor against which a contiguous-halves difference is read.
    Chunks alternate from a random phase so both sets are populated whenever
    there are at least two chunks.
    """
    chunk = max(1, min(int(chunk), ub.size // 2))
    chunks = [ub[i:i + chunk] for i in range(0, ub.size, chunk)]
    phase = int(rng.integers(0, 2))
    a = np.concatenate([c for i, c in enumerate(chunks) if (i + phase) % 2 == 0])
    b = np.concatenate([c for i, c in enumerate(chunks) if (i + phase) % 2 == 1])
    return a, b


def _mean_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def _frame_rmsd_triangle(X: np.ndarray, idx: np.ndarray, w: np.ndarray, n_atoms: int,
                         sub: int, rng: np.random.Generator) -> np.ndarray:
    """Frame-frame RMSDs of ``sub`` frames drawn by weight from ``idx``.

    Drawing with replacement is what the production importance resample does;
    pairs of the same source frame would add zeros, so they are dropped.
    """
    draw = _wdraw(idx, w, min(sub, idx.size), rng)
    Z = X[draw].reshape(draw.size, n_atoms * 3)
    sq = (Z ** 2).sum(1)
    D2 = sq[:, None] + sq[None] - 2.0 * Z @ Z.T
    D = np.sqrt(np.clip(D2, 0, None) / n_atoms)
    i, j = np.triu_indices(draw.size, 1)
    keep = draw[i] != draw[j]
    return D[i[keep], j[keep]]


# ---------------------------------------------------------------------------
# Block-level time series
# ---------------------------------------------------------------------------

def block_series(X: np.ndarray, P: np.ndarray, blocks: np.ndarray, w: np.ndarray,
                 ub: Optional[np.ndarray] = None) -> dict:
    """Per-block weighted statistics: the time-ordered description of the run.

    A block is the unit of time.  Within a block the replica at the target
    rung changes identity at almost every exchange attempt (nine times per
    19-iteration block on NTSR1 rep0), so frames at one rung are a valid
    sample but not a trajectory; the block's weighted mean or median is the
    quantity with a time index.
    """
    ub = np.unique(blocks) if ub is None else np.asarray(ub)
    mean_pc, med_pc, mean_x, ess = [], [], [], []
    for b in ub:
        m = blocks == b
        wb = w[m] / w[m].sum()
        mean_pc.append(_wmean(P[m], wb))
        order = np.argsort(P[m], axis=0)
        cw = np.cumsum(np.take_along_axis(np.repeat(wb[:, None], P.shape[1], 1), order, 0), axis=0)
        med_idx = np.argmax(cw >= 0.5, axis=0)
        med_pc.append(np.take_along_axis(P[m], order, 0)[med_idx, np.arange(P.shape[1])])
        mean_x.append(_wmean(X[m], wb))
        ess.append(1.0 / np.sum(wb ** 2))
    mean_x = np.stack(mean_x)
    grand = mean_x.mean(0)
    return dict(blocks=ub.tolist(), mean_pc=np.stack(mean_pc), median_pc=np.stack(med_pc),
                mean_structure=mean_x, rmsd_to_grand_mean=[_mean_rmsd(x, grand) for x in mean_x],
                ess=ess)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def mean_structure_drift(X: np.ndarray, blocks: np.ndarray, ub: np.ndarray, w: np.ndarray,
                         n_boot: int = 200, chunks: Sequence[int] = (8, 16),
                         n_null: int = 6, seed: int = 0) -> dict:
    """RMSD between the weighted mean structures of the two contiguous halves.

    ``boot_q95`` resamples blocks within each half (block bootstrap), the upper
    bound of the drift itself; ``null_p95`` is the same statistic on
    interleaved splits, what two samples of one ensemble give.
    """
    rng = np.random.default_rng(seed)
    h1, h2 = _halves(ub)
    bm = {b: _wmean(X[blocks == b], w[blocks == b]) for b in ub}

    def mean_of(bl):
        return np.mean([bm[b] for b in bl], axis=0)

    obs = _mean_rmsd(mean_of(h1), mean_of(h2))
    boots = [_mean_rmsd(mean_of(rng.choice(h1, h1.size)), mean_of(rng.choice(h2, h2.size)))
             for _ in range(n_boot)]
    null = []
    for c in chunks:
        for _ in range(n_null):
            a, b = _interleaved(ub, c, rng)
            null.append(_mean_rmsd(mean_of(a), mean_of(b)))
    idx, ww = _set_weights(blocks, w, ub)
    grand = _wmean(X[idx], ww)
    spread = float(np.sqrt(np.sum(ww * ((X[idx] - grand) ** 2).sum(-1).mean(-1)) / ww.sum()))
    return dict(rmsd=obs, boot_q95=float(np.quantile(boots, 0.95)),
                boot_median=float(np.median(boots)),
                null_median=float(np.median(null)), null_p95=float(np.quantile(null, 0.95)),
                per_frame_rmsd_to_mean=spread)


def state_stability(P: np.ndarray, blocks: np.ndarray, ub: np.ndarray, tol: Tolerances,
                    w: np.ndarray, n_pc: int = 2, min_state_n: int = 20) -> dict:
    """Quarter-wise state assignment and half-vs-half shift along PC1..n_pc.

    The tolerance along each component is the experimental spread of
    structures in the assigned state; a thinly populated state (GPCRdb's
    *Intermediate*, 12 structures) borrows the tighter of the well-populated
    spreads so it cannot loosen the test.
    """
    quarters = np.array_split(ub, 4)
    names = tol.states
    C = np.stack([tol.centroids[s][:n_pc] for s in names])
    bm = {b: _wmean(P[blocks == b, :n_pc], w[blocks == b]) for b in ub}
    qmeans = np.array([np.mean([bm[b] for b in q], axis=0) for q in quarters])
    assign = [names[int(np.argmin(np.linalg.norm(C - m, axis=1)))] for m in qmeans]
    h1, h2 = _halves(ub)
    shift = np.mean([bm[b] for b in h2], axis=0) - np.mean([bm[b] for b in h1], axis=0)
    state = max(set(assign), key=assign.count)
    if tol.state_counts.get(state, 0) >= min_state_n:
        sd = tol.within_state_sd[state][:n_pc]
    else:
        cands = [tol.within_state_sd[s][:n_pc] for s in names
                 if tol.state_counts.get(s, 0) >= min_state_n]
        sd = np.min(np.stack(cands), axis=0) if cands else tol.within_state_sd[state][:n_pc]
    overall = np.mean([bm[b] for b in ub], axis=0)
    d_all = np.linalg.norm(C - overall, axis=1)
    order = np.argsort(d_all)
    return dict(quarter_means=qmeans.tolist(), quarter_assign=assign, state=state,
                same_state=len(set(assign)) == 1, half_shift=shift.tolist(),
                tolerance=np.asarray(sd, float).tolist(),
                shift_over_tolerance=(np.abs(shift) / sd).tolist(),
                nearest_other_state_gap=float(d_all[order[1]]) if len(order) > 1 else float('nan'),
                pass_=bool(len(set(assign)) == 1 and np.all(np.abs(shift) <= sd)))


def ensemble_width(X: np.ndarray, blocks: np.ndarray, ub: np.ndarray, n_atoms: int,
                   w: np.ndarray, sub: int = 400, chunks: Sequence[int] = (8, 16),
                   n_null: int = 6, seed: int = 0) -> dict:
    """Distance between the two halves' frame-frame RMSD distributions.

    The production observable -- the distribution of inter-frame distances
    within an ensemble -- with two changes: the two ensembles are independent
    halves of the run drawn by MBAR weight, not overlapping cumulative
    resamples, and the distance is Wasserstein-1 in angstroms beside the JSD.
    """
    rng = np.random.default_rng(seed)
    h1, h2 = _halves(ub)

    def dist(bl_a, bl_b):
        ia, wa = _set_weights(blocks, w, bl_a)
        ib, wb = _set_weights(blocks, w, bl_b)
        ta = _frame_rmsd_triangle(X, ia, wa, n_atoms, sub, rng)
        tb = _frame_rmsd_triangle(X, ib, wb, n_atoms, sub, rng)
        e = np.linspace(min(ta.min(), tb.min()), max(ta.max(), tb.max()), 101)
        return wasserstein1(ta, tb), jsd_1d(ta, tb, e), float(np.median(ta)), float(np.median(tb))

    w1, j, ma, mb = dist(h1, h2)
    null = [dist(*_interleaved(ub, c, rng))[:2] for c in chunks for _ in range(n_null)]
    nw, nj = np.array(null).T
    return dict(w1=w1, jsd=j, median_interframe_rmsd=[ma, mb],
                null_w1_median=float(np.median(nw)), null_w1_p95=float(np.quantile(nw, .95)),
                null_jsd_median=float(np.median(nj)), null_jsd_p95=float(np.quantile(nj, .95)))


def basin_count(P: np.ndarray, w: Optional[np.ndarray] = None, n_pc: int = 3,
                max_components: int = 4, n_draw: int = 5000, seed: int = 0) -> dict:
    """How many basins the equilibrated ensemble occupies.

    Sarle's bimodality coefficient on each leading component of a weighted
    resample is the first check; if any exceeds 0.555 a Gaussian mixture on
    the leading components is fitted and the BIC-optimal count returned.  One
    basin is the common case for a ligand-bound receptor, and population tests
    are then not applicable rather than failed.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(P.shape[0])
    Z = P[_wdraw(idx, np.ones(idx.size) if w is None else w, min(n_draw, idx.size), rng)]
    bc = [bimodality_coefficient(Z[:, k]) for k in range(min(n_pc, Z.shape[1]))]
    if max(bc) < BIMODALITY_CUTOFF:
        return dict(n_basins=1, bimodality=bc, method='bimodality coefficient')
    from sklearn.mixture import GaussianMixture
    Zk = Z[:, :n_pc]
    bics = [GaussianMixture(k, n_init=3, random_state=seed).fit(Zk).bic(Zk)
            for k in range(1, max_components + 1)]
    return dict(n_basins=int(np.argmin(bics) + 1), bimodality=bc, bic=[float(b) for b in bics],
                method='GMM BIC')


def population_free_energies(P: np.ndarray, blocks: np.ndarray, ub: np.ndarray,
                             n_basins: int, w: np.ndarray, n_pc: int = 3,
                             kT: float = KT_310_KCAL, min_population: float = 0.05,
                             n_boot: int = 300, seed: int = 0) -> dict:
    """Lyman-Zuckerman population test on basins, not on arbitrary cells.

    A mixture with ``n_basins`` components is fitted on a weighted resample of
    the equilibrated ensemble; each basin's weighted population is compared
    between the halves as a free-energy difference with a block-bootstrap
    interval.
    """
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    idx, ww = _set_weights(blocks, w, ub)
    Z = P[_wdraw(idx, ww, min(5000, idx.size), rng), :n_pc]
    gm = GaussianMixture(n_basins, n_init=3, random_state=seed).fit(Z)
    lab = gm.predict(P[:, :n_pc])
    h1, h2 = _halves(ub)
    bp = {b: np.bincount(lab[blocks == b], weights=w[blocks == b], minlength=n_basins)
          / w[blocks == b].sum() for b in ub}

    def pops(bl):
        return np.mean([bp[b] for b in bl], axis=0)

    p1, p2 = pops(h1), pops(h2)
    b1 = np.array([pops(rng.choice(h1, h1.size)) for _ in range(n_boot)])
    b2 = np.array([pops(rng.choice(h2, h2.size)) for _ in range(n_boot)])
    rows = []
    for k in range(n_basins):
        if max(p1[k], p2[k]) < min_population:
            continue
        dg = -kT * np.log(max(p2[k], 1e-6) / max(p1[k], 1e-6))
        dgb = -kT * np.log(np.clip(b2[:, k], 1e-6, None) / np.clip(b1[:, k], 1e-6, None))
        rows.append(dict(basin=k, pop_half1=float(p1[k]), pop_half2=float(p2[k]), dG=float(dg),
                         dG_lo=float(np.quantile(dgb, .025)), dG_hi=float(np.quantile(dgb, .975))))
    return dict(basins=rows, kT=kT,
                pass_=all(abs(r['dG']) <= kT for r in rows) if rows else True)


def drift_diagnostics(series: dict, blocks: np.ndarray, P: np.ndarray, w: np.ndarray,
                      train_sd: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """Per-component autocorrelation, corrected trend test, R-hat, blocking error.

    All on the block-mean series -- the time-ordered object -- never on frames.
    Reported, never gated: a significance test on drift becomes harder to pass
    the longer a converged run continues, so it cannot be a stopping rule.
    """
    out = {}
    bm = np.asarray(series['mean_pc'])
    ub = np.asarray(series['blocks'])
    for k in range(bm.shape[1]):
        o = ols_corrected(bm[:, k])
        _, _, fp = fp_block_se(bm[:, k])
        within = []
        for b in ub:
            m = blocks == b
            wb = w[m] / w[m].sum()
            mu = np.sum(wb * P[m, k])
            within.append(np.sqrt(np.sum(wb * (P[m, k] - mu) ** 2)))
        d = dict(o, total_drift=o['slope'] * (ub.size - 1), rhat=rhat_split(bm[:, k]),
                 fp_se=fp, between_block_sd=float(bm[:, k].std(ddof=1)) if ub.size > 1 else float('nan'),
                 within_block_sd=float(np.mean(within)))
        if train_sd is not None:
            d['drift_over_train_sd'] = d['total_drift'] / float(train_sd[k])
        out[f'pc{k + 1}'] = d
    return out


# ---------------------------------------------------------------------------
# MBAR source-pool accounting
# ---------------------------------------------------------------------------

def pool_bounds(run_dir: str) -> Dict[int, Tuple[int, int, int]]:
    """``block -> (first_iteration, last_iteration, n_unique_frames)`` of each resample.

    Read from ``resampled_indcs.npy``.  Two resamples are independent only if
    these intervals are disjoint; adjacent blocks under a trailing t0 share
    ~90 % of their pool.
    """
    out = {}
    for d in _block_dirs(run_dir):
        p = os.path.join(d, 'resampled_indcs.npy')
        if not os.path.exists(p):
            continue
        ind = np.asarray(np.load(p, allow_pickle=True))
        if ind.ndim != 2 or ind.shape[1] < 2:
            continue
        its = ind[:, 1]
        out[int(os.path.basename(d))] = (int(its.min()), int(its.max()),
                                         int(np.unique(ind, axis=0).shape[0]))
    return out


def pool_overlap(bounds: Dict[int, Tuple[int, int, int]], a: int, b: int) -> float:
    """Fraction of block ``b``'s source interval also inside block ``a``'s."""
    lo_a, hi_a, _ = bounds[a]
    lo_b, hi_b, _ = bounds[b]
    inter = max(0, min(hi_a, hi_b) - max(lo_a, lo_b) + 1)
    return inter / max(hi_b - lo_b + 1, 1)


def disjoint_predecessor(bounds: Dict[int, Tuple[int, int, int]], b: int) -> Optional[int]:
    cands = [a for a, (lo, hi, _) in bounds.items() if a < b and hi < bounds[b][0]]
    return max(cands) if cands else None


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def equilibrated_blocks(blocks: np.ndarray, b0: Optional[int] = None,
                        fraction: float = 0.25) -> np.ndarray:
    ub = np.unique(blocks)
    if b0 is None:
        b0 = int(ub[int(np.floor(fraction * ub.size))]) if ub.size else 0
    return ub[ub >= b0]


def panel(X: np.ndarray, P: np.ndarray, blocks: np.ndarray, tol: Tolerances,
          b0: Optional[int] = None, width_fraction: float = 0.25,
          train_sd: Optional[np.ndarray] = None, seed: int = 0,
          min_blocks: int = 2 * MIN_BLOCKS_PER_HALF,
          w: Optional[np.ndarray] = None) -> dict:
    """Evaluate every gate on the equilibrated blocks and return one verdict.

    ``X`` superposed conserved-atom coordinates (n_frames, n_atoms, 3) in A;
    ``P`` their projections (n_frames, n_pc); ``blocks`` the block of each
    frame; ``w`` per-frame MBAR weights at the target rung (any scale; they
    are normalised within each block, and blocks weigh equally).  Without
    ``w`` every frame in a block counts equally.  ``b0`` is the first
    equilibrated block (default: the first quarter is discarded).  The width
    gate's tolerance is ``width_fraction`` of the same-state experimental RMSD.
    """
    ub = equilibrated_blocks(blocks, b0)
    n_atoms = X.shape[1]
    w = normalize_block_weights(blocks, w)
    res = dict(n_blocks_total=int(np.unique(blocks).size), b0=int(ub[0]) if ub.size else None,
               n_blocks_equilibrated=int(ub.size), n_frames=int(np.isin(blocks, ub).sum()),
               weighted=True,
               tolerances=dict(same_state_rmsd=tol.same_state_rmsd,
                               family_same_state_rmsd=tol.family_same_state_rmsd,
                               diff_state_rmsd=tol.diff_state_rmsd,
                               receptor=tol.receptor, receptor_n=tol.receptor_n_structures,
                               width_rmsd=width_fraction * tol.same_state_rmsd))
    if ub.size < min_blocks:
        res['verdict'] = 'INSUFFICIENT'
        res['reason'] = (f'{ub.size} equilibrated blocks; the panel needs at least '
                         f'{min_blocks} ({min_blocks // 2} per half)')
        return res

    eq = np.isin(blocks, ub)
    series = block_series(X, P, blocks, w, ub)
    ms = mean_structure_drift(X, blocks, ub, w, seed=seed)
    ms['tolerance'] = tol.same_state_rmsd
    ms['pass_'] = bool(ms['boot_q95'] <= tol.same_state_rmsd)
    ss = state_stability(P, blocks, ub, tol, w)
    ew = ensemble_width(X, blocks, ub, n_atoms, w, seed=seed)
    ew['tolerance'] = width_fraction * tol.same_state_rmsd
    ew['pass_'] = bool(ew['w1'] <= ew['tolerance'])
    idx, ww = _set_weights(blocks, w, ub)
    bc = basin_count(P[idx], ww, seed=seed)
    if bc['n_basins'] > 1:
        pf = population_free_energies(P, blocks, ub, bc['n_basins'], w, seed=seed)
    else:
        pf = dict(basins=[], kT=KT_310_KCAL, pass_=True, not_applicable=True)
    gates = dict(mean_structure=ms, state=ss, width=ew, basins=bc, populations=pf)
    res['gates'] = gates
    res['series'] = dict(blocks=series['blocks'],
                         mean_pc=np.asarray(series['mean_pc'])[:, :5].tolist(),
                         median_pc=np.asarray(series['median_pc'])[:, :5].tolist(),
                         rmsd_to_grand_mean=series['rmsd_to_grand_mean'],
                         ess=series['ess'])
    res['ess_median'] = float(np.median(series['ess']))
    res['diagnostics'] = drift_diagnostics(series, blocks, P, w, train_sd)
    failed = [g for g in ('mean_structure', 'state', 'width', 'populations')
              if not gates[g]['pass_']]
    res['failed'] = failed
    res['verdict'] = 'CONVERGED' if not failed else 'NOT CONVERGED'
    return res


def prefix_curve(X: np.ndarray, P: np.ndarray, blocks: np.ndarray, tol: Tolerances,
                 prefixes: Sequence[int], b0_fraction: float = 0.25,
                 w: Optional[np.ndarray] = None, **kw) -> List[dict]:
    """The panel as it would have read at each prefix length, in blocks.

    A metric fit to be a stopping rule should, once it passes, keep passing as
    the run continues; this is the check.
    """
    out = []
    for n in prefixes:
        sel = blocks < n
        if sel.sum() == 0:
            continue
        ubn = np.unique(blocks[sel])
        b0 = int(ubn[int(np.floor(b0_fraction * ubn.size))])
        r = panel(X[sel], P[sel], blocks[sel], tol, b0=b0,
                  w=None if w is None else w[sel], **kw)
        g = r.get('gates', {})
        out.append(dict(N=int(n), verdict=r['verdict'], b0=r['b0'],
                        mean_rmsd=g.get('mean_structure', {}).get('rmsd'),
                        mean_rmsd_q95=g.get('mean_structure', {}).get('boot_q95'),
                        state=g.get('state', {}).get('state'),
                        same_state=g.get('state', {}).get('same_state'),
                        pc1_shift=(g.get('state', {}).get('half_shift') or [None])[0],
                        w1=g.get('width', {}).get('w1'),
                        n_basins=g.get('basins', {}).get('n_basins')))
    return out


def format_panel(r: dict) -> str:
    t = r['tolerances']
    lines = [f"blocks {r['n_blocks_total']}  equilibrated from {r['b0']}: {r['n_blocks_equilibrated']} blocks, "
             f"{r['n_frames']} frames" + (f", ESS/block median {r['ess_median']:.0f}" if 'ess_median' in r else ''),
             f"tolerances: same-state RMSD {t['same_state_rmsd']:.2f} A "
             f"({'receptor ' + str(t['receptor']) + ', n=' + str(t['receptor_n']) if t['receptor_n'] else 'family median'}; "
             f"family {t['family_same_state_rmsd']:.2f}, different state {t['diff_state_rmsd']:.2f})"]
    if r['verdict'] == 'INSUFFICIENT':
        lines.append(f"verdict: INSUFFICIENT -- {r['reason']}")
        return '\n'.join(lines)
    g = r['gates']
    ms, ss, ew, bc, pf = g['mean_structure'], g['state'], g['width'], g['basins'], g['populations']
    ok = lambda b: 'PASS' if b else 'FAIL'
    lines += [
        f"[{ok(ms['pass_'])}] mean-structure drift   {ms['rmsd']:.3f} A (block-bootstrap q95 {ms['boot_q95']:.3f}) "
        f"<= {ms['tolerance']:.2f} A   [interleaved floor p95 {ms['null_p95']:.3f}]",
        f"[{ok(ss['pass_'])}] state stability        quarters {'/'.join(a[:3] for a in ss['quarter_assign'])}; "
        f"half shift PC1 {ss['half_shift'][0]:+.2f} A (tol {ss['tolerance'][0]:.2f}), "
        f"PC2 {ss['half_shift'][1]:+.2f} A (tol {ss['tolerance'][1]:.2f})",
        f"[{ok(ew['pass_'])}] ensemble width         W1 {ew['w1']:.3f} A <= {ew['tolerance']:.3f} A; "
        f"JSD {ew['jsd']:.3f}   [interleaved floor p95: W1 {ew['null_w1_p95']:.3f}, JSD {ew['null_jsd_p95']:.3f}]",
        f"[{ok(pf['pass_'])}] basin populations      {bc['n_basins']} basin(s) "
        f"(bimodality {', '.join(f'{b:.2f}' for b in bc['bimodality'])})"
        + ('; not applicable' if pf.get('not_applicable') else
           '; ' + ', '.join(f"basin {b['basin']} dG {b['dG']:+.2f} [{b['dG_lo']:+.2f},{b['dG_hi']:+.2f}]" for b in pf['basins'])),
        f"verdict: {r['verdict']}" + (f"  (failed: {', '.join(r['failed'])})" if r['failed'] else ''),
        "diagnostics on the block-mean series (reported, not gated):",
        f"  {'PC':>4} {'tau':>5} {'n_eff':>5} {'p_naive':>8} {'p_corr':>7} {'R-hat':>5} {'drift':>6} {'FP se':>6}"]
    for k, d in list(r['diagnostics'].items())[:5]:
        lines.append(f"  {k:>4} {d['tau']:>5.1f} {d['neff']:>5.0f} {d['p_naive']:>8.1e} {d['p_corr']:>7.3f} "
                     f"{d['rhat']:>5.2f} {d['total_drift']:>+6.2f} {d['fp_se']:>6.3f}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Live use inside the PTRE block loop
# ---------------------------------------------------------------------------

STOP_MODES = ('jsd', 'panel', 'both', 'either')


def combine_stop(mode: str, jsd_stop: Optional[bool], panel_stop: Optional[bool],
                 at_max_time: bool = False) -> bool:
    """One stop decision from the two criteria.

    ``jsd`` keeps the published behaviour.  ``panel`` hands the decision to the
    panel while the JSD machinery still runs every block for its resampled
    artifacts and prints its own report, so nothing the campaign already
    produces is lost.  ``both`` requires agreement; ``either`` stops on the
    first.  The aggregate-time ceiling ends the run in every mode.  A criterion
    that was not evaluated (``None``) counts as not stopping.
    """
    if mode not in STOP_MODES:
        raise ValueError(f'unknown convergence mode {mode!r}; choose from {STOP_MODES}')
    if at_max_time:
        return True
    j, p = bool(jsd_stop), bool(panel_stop)
    return {'jsd': j, 'panel': p, 'both': j and p, 'either': j or p}[mode]


def _tol_to_dict(tol: Tolerances) -> dict:
    return dict(same_state_rmsd=tol.same_state_rmsd,
                family_same_state_rmsd=tol.family_same_state_rmsd,
                diff_state_rmsd=tol.diff_state_rmsd,
                centroids={k: np.asarray(v, float).tolist() for k, v in tol.centroids.items()},
                within_state_sd={k: np.asarray(v, float).tolist() for k, v in tol.within_state_sd.items()},
                receptor=tol.receptor, receptor_n_structures=tol.receptor_n_structures,
                receptor_groups=tol.receptor_groups, state_counts=tol.state_counts)


def _tol_from_dict(d: dict) -> Tolerances:
    return Tolerances(same_state_rmsd=d['same_state_rmsd'],
                      family_same_state_rmsd=d['family_same_state_rmsd'],
                      diff_state_rmsd=d['diff_state_rmsd'],
                      centroids={k: np.asarray(v, float) for k, v in d['centroids'].items()},
                      within_state_sd={k: np.asarray(v, float) for k, v in d['within_state_sd'].items()},
                      receptor=d.get('receptor'), receptor_n_structures=d.get('receptor_n_structures', 0),
                      receptor_groups=d.get('receptor_groups', []), state_counts=d.get('state_counts', {}))


class LivePanel:
    """The panel as a stopping rule, evaluated after every completed block.

    Runs inside :meth:`FultonMarket.run` on rank 0.  Each block's conserved
    atoms for the lowest rungs are read once from the block's ``positions.npy``
    together with their reduced potentials from ``energies.npy``; MBAR over
    those rungs gives every frame its weight in the block's 310 K ensemble, and
    the block is the unit of time (blocks weigh equally, frames within a block
    by MBAR weight).  All of it is kept under ``<run_dir>/analysis/live_panel/``,
    so an evaluation costs a few seconds regardless of run length and a
    resubmitted job resumes from the cache.  The verdict must hold for ``consecutive`` blocks before the run
    stops: a single pass at the bootstrap bound can flicker, and a stopping
    rule should not act on one.

    Everything the loop needs that could require the network -- GPCRdb
    numbering, tolerances from the training set -- is resolved once by
    :meth:`prepare` and written to ``live_panel_config.json``; run it from a
    login node before submitting (``chimpss-convergence-panel --prepare``).
    """
    CONFIG = 'live_panel_config.json'
    STATE = 'live_panel_state.json'
    FRAMES = 'live_panel'

    def __init__(self, run_dir: str, topology: str, model_prefix: str,
                 pdb_code: Optional[str] = None, receptor: Optional[str] = None,
                 stor_dir: str = './many_structures/', n_states: int = 5, skip: int = 0,
                 b0_fraction: float = 0.25, width_fraction: float = 0.25,
                 consecutive: int = 3, min_blocks: int = 2 * MIN_BLOCKS_PER_HALF,
                 n_pc: int = 10, chain_index: Optional[int] = None,
                 model: Optional[PanGPCRModel] = None,
                 atom_indices: Optional[Sequence[int]] = None,
                 tolerances: Optional[Tolerances] = None,
                 n_atoms_total: Optional[int] = None,
                 printf: Callable = print):
        self.run_dir = run_dir
        self.topology = topology
        self.model_prefix = model_prefix
        self.pdb_code = pdb_code
        self.receptor = receptor
        self.stor_dir = stor_dir
        self.n_states = int(n_states)
        self.skip = int(skip)
        self.b0_fraction = float(b0_fraction)
        self.width_fraction = float(width_fraction)
        self.consecutive = int(consecutive)
        self.min_blocks = int(min_blocks)
        self.n_pc = int(n_pc)
        self.chain_index = None if chain_index is None else int(chain_index)
        self.printf = printf
        self._model = model
        self._atom_indices = None if atom_indices is None else np.asarray(atom_indices, np.int64)
        self._tol = tolerances
        self._n_atoms_total = n_atoms_total
        self._train_sd = None
        self.mapping: dict = {}
        self.analysis_dir = os.path.join(run_dir, 'analysis')
        self.frames_dir = os.path.join(self.analysis_dir, self.FRAMES)

    # -- lazy resources -------------------------------------------------------

    @property
    def model(self) -> PanGPCRModel:
        if self._model is None:
            from chimpss.analysis.pca_equilibration import load_model
            self._model = load_model(self.model_prefix)
        return self._model

    @property
    def train_sd(self) -> Optional[np.ndarray]:
        if self._train_sd is None:
            p = f'{self.model_prefix}_train_projections.npy'
            if os.path.exists(p):
                self._train_sd = np.load(p)[:, :self.n_pc].std(0)
        return self._train_sd

    # -- preparation ----------------------------------------------------------

    def prepare(self) -> dict:
        """Resolve atoms and tolerances, write the config, return it.

        Raises if any conserved position cannot be found in the topology: a
        construct missing conserved atoms cannot be projected on this basis,
        and that must surface before the first block is paid for, not after.
        """
        os.makedirs(self.frames_dir, exist_ok=True)
        cfg_path = os.path.join(self.analysis_dir, self.CONFIG)
        if self._atom_indices is None and os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            self.apply_config(cfg)
            self.printf(f'convergence panel: loaded {cfg_path}')
            return cfg

        if self._atom_indices is None:
            import mdtraj as md
            from chimpss.analysis.gpcr_pca import fetch_naming_info
            if not self.pdb_code:
                raise ValueError('a parent pdb_code is required to number the topology '
                                 'against GPCRdb')
            naming, _ = fetch_naming_info(self.pdb_code, self.stor_dir)
            top_traj = md.load(self.topology)
            idx, missing, stats = resolve_atoms(top_traj.topology, naming, self.model,
                                                chain_index=self.chain_index)
            if missing:
                lines = '; '.join(f'{lbl}: {why}' for lbl, _, why in missing[:6])
                raise RuntimeError(
                    f'{len(missing)} of {len(self.model.conserved_bw)} conserved positions '
                    f'cannot be resolved in {self.topology} ({lines}{" ..." if len(missing) > 6 else ""}). '
                    f'Build a basis without them (chimpss-generate-pca --bw_exclude) or '
                    f'run this construct with --convergence jsd.')
            self._atom_indices = idx
            self._n_atoms_total = int(top_traj.n_atoms)
            self.mapping = {k: stats[k] for k in ('identity', 'n_aligned', 'n_positions',
                                                   'n_resolved', 'offsets') if k in stats}
        if self._tol is None:
            receptor = self.receptor
            if receptor is None and self.pdb_code:
                receptor = self.model.meta.get('structure_meta', {}).get(
                    self.pdb_code.upper(), {}).get('protein')
            if receptor is None and self.pdb_code:
                # the parent need not be in the training set (8DZR, 6ZA8 are
                # not); the GPCRdb record fetch_naming_info cached still names
                # the receptor
                from chimpss.analysis.gpcr_pca import Structure_Analyzer
                sa = Structure_Analyzer(self.pdb_code.upper(), stor_dir=self.stor_dir)
                if sa.load_cache():
                    receptor = sa.meta.get('structure', {}).get('protein')
            self._tol = experimental_tolerances(self.model_prefix, receptor=receptor, n_pc=self.n_pc)
            self.receptor = receptor
        cfg = dict(atom_indices=self._atom_indices.tolist(), n_atoms_total=self._n_atoms_total,
                   mapping=self.mapping, tolerances=_tol_to_dict(self._tol), receptor=self.receptor,
                   pdb_code=self.pdb_code, model_prefix=self.model_prefix, topology=self.topology,
                   n_states=self.n_states, skip=self.skip, b0_fraction=self.b0_fraction,
                   width_fraction=self.width_fraction, consecutive=self.consecutive,
                   min_blocks=self.min_blocks, n_pc=self.n_pc)
        with open(cfg_path, 'w') as fh:
            json.dump(cfg, fh, indent=1)
        self.printf(f'convergence panel: {len(self._atom_indices)} conserved atoms resolved'
                    + (f" (identity {self.mapping.get('identity', float('nan')):.3f})" if self.mapping else '')
                    + f'; same-state tolerance {self._tol.same_state_rmsd:.2f} A'
                    + (f' ({self.receptor}, n={self._tol.receptor_n_structures})'
                       if self._tol.receptor_n_structures else ' (family median)')
                    + f'; config -> {cfg_path}')
        return cfg

    def apply_config(self, cfg: dict) -> None:
        """Adopt a config produced by :meth:`prepare`, e.g. one broadcast to other ranks."""
        self._atom_indices = np.asarray(cfg['atom_indices'], np.int64)
        self._n_atoms_total = cfg.get('n_atoms_total')
        self.mapping = cfg.get('mapping', {})
        self._tol = _tol_from_dict(cfg['tolerances'])
        self.receptor = cfg.get('receptor', self.receptor)
        for k in ('n_states', 'skip', 'b0_fraction', 'width_fraction', 'consecutive',
                  'min_blocks', 'n_pc'):
            if k in cfg:
                setattr(self, k, type(getattr(self, k))(cfg[k]))
        os.makedirs(self.frames_dir, exist_ok=True)

    @property
    def tolerances(self) -> Tolerances:
        if self._tol is None:
            raise RuntimeError('LivePanel.prepare() has not been run')
        return self._tol

    # -- frames ---------------------------------------------------------------

    def _block_path(self, block: int) -> str:
        return os.path.join(self.frames_dir, f'{int(block):04d}.npz')

    def ingest(self, block: int) -> bool:
        """Read one block's conserved atoms for the lowest rungs; False if cached."""
        path = self._block_path(block)
        if os.path.exists(path):
            return False
        if self._atom_indices is None:
            raise RuntimeError('LivePanel.prepare() has not been run')
        d = os.path.join(self.run_dir, 'saved_variables', str(int(block)))
        spath, ppath, tpath = (os.path.join(d, f) for f in ('states.npy', 'positions.npy',
                                                             'temperatures.npy'))
        st = np.load(spath)
        n_rep = int(np.load(tpath, allow_pickle=True).size) if os.path.exists(tpath) else st.shape[1]
        n_atoms = self._n_atoms_total
        if n_atoms is None:
            # raw float32 buffer: the shape lives only in the sibling arrays
            n_atoms = os.path.getsize(ppath) // (st.shape[0] * n_rep * 3 * 4)
        pos = _load_positions(ppath, st.shape[0], n_rep, int(n_atoms))
        if pos is None:
            self.printf(f'convergence panel: block {block} positions unreadable, skipped')
            return False
        # energies.npy is (iteration, replica, state) in reduced units, indexed
        # by replica like positions.npy; the frame at positions[it, r] pairs
        # with energies[it, r, :].
        epath = os.path.join(d, 'energies.npy')
        ener = np.load(epath, mmap_mode='r') if os.path.exists(epath) else None
        K = self.n_states
        frames, states, u = [], [], []
        for it in range(self.skip, min(st.shape[0], pos.shape[0])):
            for s in range(K):
                hit = np.where(st[it] == s)[0]
                if hit.size:
                    frames.append(np.asarray(pos[it, hit[0]][self._atom_indices], np.float32))
                    states.append(s)
                    if ener is not None and ener.shape[0] > it:
                        u.append(np.asarray(ener[it, hit[0], :K], np.float64))
        del pos
        coords = (np.stack(frames) if frames
                  else np.empty((0, self._atom_indices.size, 3), np.float32))
        states_a = np.asarray(states, np.int64)
        if len(u) == len(frames) and frames:
            u_a = np.stack(u)
            w, f_k, ok = block_mbar_weights(u_a, states_a, K, target=0)
        else:
            u_a = np.full((len(frames), K), np.nan)
            w, f_k, ok = block_mbar_weights(u_a, states_a, K, target=0)  # uniform over rung 0
            ok = False
        if not ok:
            self.printf(f'convergence panel: block {block}: MBAR weights unavailable, '
                        f'using the frames sampled at rung 0 uniformly')
        os.makedirs(self.frames_dir, exist_ok=True)
        # per-process temporary name: the live run and an offline builder can
        # ingest the same block concurrently, and a shared name would let one
        # replace the other's half-written file
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(tmp, coords=coords, states=states_a.astype(np.int8), u=u_a.astype(np.float32),
                 w=w.astype(np.float64), f_k=np.asarray(f_k, np.float64),
                 ess=np.float64(1.0 / np.sum(w ** 2)) if w.size else np.float64(0.0),
                 weights_ok=np.bool_(ok))
        os.replace(tmp, path)
        return True

    def _blocks_on_disk(self) -> List[int]:
        sv = os.path.join(self.run_dir, 'saved_variables')
        if not os.path.isdir(sv):
            return []
        out = []
        for name in os.listdir(sv):
            if name.isdigit() and all(os.path.exists(os.path.join(sv, name, f))
                                      for f in ('positions.npy', 'states.npy')):
                out.append(int(name))
        return sorted(out)

    def ingest_all(self) -> List[int]:
        """Ingest every complete block not yet cached (a resubmitted job catches up here)."""
        return [b for b in self._blocks_on_disk() if self.ingest(b)]

    def cached_blocks(self) -> List[int]:
        if not os.path.isdir(self.frames_dir):
            return []
        return sorted(int(f[:-4]) for f in os.listdir(self.frames_dir)
                      if f.endswith('.npz') and f[:-4].isdigit())

    def load_frames(self, max_block: Optional[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every cached block, or only those up to ``max_block`` (replays).

        Returns ``(coords_nm, blocks, states, weights)``; the weights are the
        per-block MBAR weights at rung 0, each block summing to one.
        """
        coords, blocks, states, weights = [], [], [], []
        for b in self.cached_blocks():
            if max_block is not None and b > max_block:
                continue
            d = np.load(self._block_path(b))
            c = d['coords']
            if c.shape[0] == 0:
                continue
            coords.append(c)
            blocks.append(np.full(c.shape[0], b, np.int64))
            states.append(d['states'].astype(np.int64))
            w = d['w'] if 'w' in d.files else np.full(c.shape[0], 1.0 / c.shape[0])
            weights.append(np.asarray(w, np.float64))
        if not coords:
            n = 0 if self._atom_indices is None else self._atom_indices.size
            return (np.empty((0, n, 3), np.float32), np.empty(0, np.int64),
                    np.empty(0, np.int64), np.empty(0, np.float64))
        return (np.concatenate(coords), np.concatenate(blocks), np.concatenate(states),
                np.concatenate(weights))

    def weights_report(self) -> dict:
        """Per-block MBAR health: ESS, rung shares of the weight, solver failures."""
        rows = []
        for b in self.cached_blocks():
            d = np.load(self._block_path(b))
            if 'w' not in d.files or d['coords'].shape[0] == 0:
                continue
            w, st = d['w'], d['states'].astype(int)
            rows.append(dict(block=int(b), ess=float(1.0 / np.sum(w ** 2)),
                             rung_share=np.bincount(st, weights=w, minlength=self.n_states)[:self.n_states].tolist(),
                             ok=bool(d['weights_ok']) if 'weights_ok' in d.files else True))
        if not rows:
            return {}
        share = np.mean([r['rung_share'] for r in rows], axis=0)
        return dict(n_blocks=len(rows), ess_median=float(np.median([r['ess'] for r in rows])),
                    ess_p10=float(np.quantile([r['ess'] for r in rows], .1)),
                    rung_share_mean=share.tolist(), n_failed=sum(not r['ok'] for r in rows))

    # -- evaluation -----------------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self.analysis_dir, self.STATE)

    def _load_state(self) -> dict:
        p = self._state_path()
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    return json.load(fh)
            except Exception as exc:                            # pragma: no cover
                self.printf(f'convergence panel: could not read {p} ({exc}); starting afresh')
        return {'history': []}

    def evaluate(self, sim_no: Optional[int] = None, past_minimum: bool = True,
                 max_block: Optional[int] = None, persist: bool = True) -> dict:
        """Run the panel on every cached block and update the consecutive-pass streak.

        Re-evaluating the same ``sim_no`` replaces that block's entry rather
        than extending the streak, so a restarted evaluation cannot stop a run
        that a single pass would not have.  ``max_block`` restricts the frames
        to blocks up to that index, which replays the rule as it would have
        read partway through a finished run; ``persist=False`` leaves the live
        state file alone, for offline use on a run directory.
        """
        coords, blocks, _, w = self.load_frames(max_block=max_block)
        tol = self.tolerances
        if sim_no is None:
            sim_no = int(blocks.max()) if blocks.size else -1
        if coords.shape[0] == 0:
            r = dict(verdict='INSUFFICIENT', reason='no frames cached',
                     n_blocks_total=0, n_blocks_equilibrated=0, b0=None, n_frames=0,
                     tolerances=dict(same_state_rmsd=tol.same_state_rmsd))
        else:
            X = superpose_frames(coords, self.model)
            P = project_superposed(X, self.model, n_pc=self.n_pc)
            ub = np.unique(blocks)
            b0 = int(ub[int(np.floor(self.b0_fraction * ub.size))])
            r = panel(X, P, blocks, tol, b0=b0, width_fraction=self.width_fraction,
                      train_sd=self.train_sd, min_blocks=self.min_blocks, w=w)

        converged = r['verdict'] == 'CONVERGED'
        state = self._load_state() if persist else {'history': []}
        hist = [h for h in state.get('history', []) if h.get('block') != int(sim_no)]
        g = r.get('gates', {})
        entry = dict(block=int(sim_no), verdict=r['verdict'], n_blocks=r['n_blocks_total'],
                     b0=r['b0'], past_minimum=bool(past_minimum),
                     mean_rmsd=g.get('mean_structure', {}).get('rmsd'),
                     mean_rmsd_q95=g.get('mean_structure', {}).get('boot_q95'),
                     tolerance=tol.same_state_rmsd,
                     state=g.get('state', {}).get('state'),
                     same_state=g.get('state', {}).get('same_state'),
                     pc1_shift=(g.get('state', {}).get('half_shift') or [None])[0],
                     pc1_tolerance=(g.get('state', {}).get('tolerance') or [None])[0],
                     w1=g.get('width', {}).get('w1'),
                     w1_tolerance=g.get('width', {}).get('tolerance'),
                     n_basins=g.get('basins', {}).get('n_basins'),
                     ess_median=r.get('ess_median'),
                     failed=r.get('failed', []))
        hist.append(entry)
        hist.sort(key=lambda h: h['block'])
        streak = 0
        for h in reversed(hist):
            if h['verdict'] == 'CONVERGED':
                streak += 1
            else:
                break
        entry['streak'] = streak
        entry['stop'] = bool(past_minimum and converged and streak >= self.consecutive)
        if persist:
            state = {'history': hist, 'streak': streak, 'consecutive': self.consecutive}
            os.makedirs(self.analysis_dir, exist_ok=True)
            with open(self._state_path(), 'w') as fh:
                json.dump(state, fh, indent=1)
            with open(os.path.join(self.frames_dir, f'eval_{int(sim_no):04d}.json'), 'w') as fh:
                json.dump(r, fh, indent=1, default=float)
        return dict(entry, full=r)

    def report(self, e: dict) -> None:
        """Print the verdict in the house ``[PASS]/[FAIL]`` style."""
        r = e['full']
        pf = lambda ok: 'PASS' if ok else 'FAIL'
        width = 60
        self.printf('=' * (width + 12))
        self.printf(f"  {'Convergence Panel (block ' + str(e['block']) + ')':^{width}}")
        self.printf('=' * (width + 12))
        self.printf(f"  [{pf(e['past_minimum'])}]  Past minimum simulation fraction")
        if e.get('ess_median') is not None:
            self.printf(f"          {r['n_blocks_equilibrated']} equilibrated blocks, per-block MBAR "
                        f"weights at rung 0, ESS/block median {e['ess_median']:.0f}")
        if r['verdict'] == 'INSUFFICIENT':
            self.printf(f"  [FAIL]  {r.get('reason', 'insufficient data')}")
        else:
            g = r['gates']
            ms, ss, ew, bc, pp = (g['mean_structure'], g['state'], g['width'],
                                  g['basins'], g['populations'])
            self.printf(f"  [{pf(ms['pass_'])}]  Mean-structure drift {ms['rmsd']:.3f} A "
                        f"(bootstrap q95 {ms['boot_q95']:.3f}) <= {ms['tolerance']:.2f} A "
                        f"[interleaved floor {ms['null_p95']:.3f}]")
            self.printf(f"  [{pf(ss['pass_'])}]  State stability: quarters "
                        f"{'/'.join(a[:3] for a in ss['quarter_assign'])}; PC1 shift "
                        f"{ss['half_shift'][0]:+.2f} A (tol {ss['tolerance'][0]:.2f}), PC2 "
                        f"{ss['half_shift'][1]:+.2f} A (tol {ss['tolerance'][1]:.2f})")
            self.printf(f"  [{pf(ew['pass_'])}]  Ensemble width W1 {ew['w1']:.3f} A <= "
                        f"{ew['tolerance']:.3f} A [floor {ew['null_w1_p95']:.3f}]")
            pops = ('not applicable' if pp.get('not_applicable') else ', '.join(
                f"basin {b['basin']} dG {b['dG']:+.2f}" for b in pp['basins']))
            self.printf(f"  [{pf(pp['pass_'])}]  Basin populations: {bc['n_basins']} basin(s); {pops}")
        self.printf(f"  [{pf(e['streak'] >= self.consecutive)}]  Held for {e['streak']} of "
                    f"{self.consecutive} consecutive blocks")
        self.printf('-' * (width + 12))
        self.printf(f"  verdict: {r['verdict']}"
                    + (f"  (failed: {', '.join(e['failed'])})" if e.get('failed') else '')
                    + f"  ->  {'STOP' if e['stop'] else 'continue'}")
        if 'diagnostics' in r:
            d = r['diagnostics']
            top = list(d.items())[:3]
            self.printf('  diagnostics (block-mean series): ' + '; '.join(
                f"{k} tau {v['tau']:.1f} n_eff {v['neff']:.0f} p_corr {v['p_corr']:.3f} "
                f"R-hat {v['rhat']:.2f}" for k, v in top))
        self.printf('=' * (width + 12))
