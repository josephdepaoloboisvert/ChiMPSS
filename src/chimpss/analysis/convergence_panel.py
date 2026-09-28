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


def collect_low_temperature_frames(run_dir: str, topology: str, model: PanGPCRModel,
                                   naming_info: Sequence[dict], n_states: int = 5,
                                   skip: int = 0, chain_index: int = 0,
                                   cache: bool = True, max_blocks: Optional[int] = None,
                                   _printf: Callable = print
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Conserved atoms of the ``n_states`` lowest rungs, every iteration.

    Returns ``(coords_nm, blocks, states, info)``.  ``skip`` discards the first
    ``skip`` iterations of each block (a restart from checkpoint carries
    velocities, so the default keeps all of them).  With the default ladder the
    first five rungs span 310.0-312.3 K, close enough that pooling them is a
    2 % change in beta; the state of every frame is returned so a caller can
    reweight or restrict to rung 0.
    """
    import mdtraj as md

    tag = f'lowT_s{n_states}_skip{skip}_{_fingerprint(model)}'
    out_path = os.path.join(run_dir, 'analysis', f'{tag}.npz')
    if cache and os.path.exists(out_path):
        d = np.load(out_path)
        info = json.loads(str(d['info']))
        available = {int(os.path.basename(p)) for p in _block_dirs(run_dir)
                     if os.path.exists(os.path.join(p, 'positions.npy'))}
        if max_blocks is not None:
            available = set(sorted(available)[:max_blocks])
        if not (available - set(info['blocks_read'])):
            return d['coords'], d['blocks'], d['states'], info
        _printf(f'low-T frame cache is stale; rebuilding')

    top_traj = md.load(topology)
    atom_indices, missing, stats_ = resolve_atoms(top_traj.topology, naming_info, model,
                                                  chain_index=chain_index)
    n_atoms_total = top_traj.n_atoms
    dirs = _block_dirs(run_dir)
    if max_blocks is not None:
        dirs = dirs[:max_blocks]

    frames, blocks, states_ = [], [], []
    read = []
    for d in dirs:
        ppath, spath = os.path.join(d, 'positions.npy'), os.path.join(d, 'states.npy')
        if not (os.path.exists(ppath) and os.path.exists(spath)):
            continue
        st = np.load(spath)
        tpath = os.path.join(d, 'temperatures.npy')
        n_rep = int(np.load(tpath, allow_pickle=True).size) if os.path.exists(tpath) else st.shape[1]
        pos = _load_positions(ppath, st.shape[0], n_rep, n_atoms_total)
        if pos is None:
            continue
        b = int(os.path.basename(d))
        read.append(b)
        for it in range(skip, min(st.shape[0], pos.shape[0])):
            for s in range(n_states):
                hit = np.where(st[it] == s)[0]
                if hit.size:
                    frames.append(np.asarray(pos[it, hit[0]][atom_indices], dtype=np.float32))
                    blocks.append(b)
                    states_.append(s)
        del pos

    coords = np.stack(frames) if frames else np.empty((0, atom_indices.size, 3), np.float32)
    blocks_a = np.asarray(blocks, np.int64)
    states_a = np.asarray(states_, np.int64)
    info = dict(stats_, atom_indices=atom_indices.tolist(), n_states=int(n_states),
                skip=int(skip), n_frames=int(coords.shape[0]), blocks_read=read,
                missing=[[lbl, int(i), why] for lbl, i, why in missing])
    if cache:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, coords=coords, blocks=blocks_a, states=states_a,
                            info=json.dumps(info))
    return coords, blocks_a, states_a, info


def superpose_frames(coords_nm: np.ndarray, model: PanGPCRModel) -> np.ndarray:
    """Every frame superposed on the model reference, in angstroms."""
    x = np.asarray(coords_nm, np.float64) * NM_TO_ANGSTROM
    return np.stack([kabsch_superpose(f, model.ref_positions, model.ref_masses) for f in x])


def project_superposed(X: np.ndarray, model: PanGPCRModel, n_pc: int = 10) -> np.ndarray:
    return np.asarray(model.pca.transform(X.reshape(X.shape[0], -1))[:, :n_pc], np.float64)


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
    """Alternate chunks of ``chunk`` blocks between two sets, random assignment.

    Two sets built this way sample the same time span, so any difference
    between them is finite-sample noise plus autocorrelation shorter than
    ``chunk`` -- the floor against which a contiguous-halves difference is read.
    """
    chunks = [ub[i:i + chunk] for i in range(0, ub.size, chunk)]
    lab = rng.permutation(len(chunks)) % 2
    a = np.concatenate([c for c, l in zip(chunks, lab) if l == 0])
    b = np.concatenate([c for c, l in zip(chunks, lab) if l == 1])
    return a, b


def _mean_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def _frame_rmsd_triangle(X: np.ndarray, n_atoms: int, sub: int,
                         rng: np.random.Generator) -> np.ndarray:
    Z = X[rng.choice(X.shape[0], min(sub, X.shape[0]), replace=False)].reshape(-1, n_atoms * 3)
    sq = (Z ** 2).sum(1)
    D2 = sq[:, None] + sq[None] - 2.0 * Z @ Z.T
    D = np.sqrt(np.clip(D2, 0, None) / n_atoms)
    return D[np.triu_indices(Z.shape[0], 1)]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def mean_structure_drift(X: np.ndarray, blocks: np.ndarray, ub: np.ndarray,
                         n_boot: int = 200, chunks: Sequence[int] = (8, 16),
                         n_null: int = 6, seed: int = 0) -> dict:
    """RMSD between the mean structures of the two contiguous halves.

    ``boot_q95`` resamples blocks within each half (block bootstrap), so it is
    the upper bound of the drift itself; ``null_p95`` is the same statistic on
    interleaved splits, the value two samples of the same ensemble give.
    """
    rng = np.random.default_rng(seed)
    h1, h2 = _halves(ub)
    by = {b: X[blocks == b] for b in ub}

    def mean_of(bl):
        return np.concatenate([by[b] for b in bl]).mean(0)

    obs = _mean_rmsd(mean_of(h1), mean_of(h2))
    boots = [_mean_rmsd(mean_of(rng.choice(h1, h1.size)), mean_of(rng.choice(h2, h2.size)))
             for _ in range(n_boot)]
    null = []
    for c in chunks:
        for _ in range(n_null):
            a, b = _interleaved(ub, c, rng)
            null.append(_mean_rmsd(mean_of(a), mean_of(b)))
    return dict(rmsd=obs, boot_q95=float(np.quantile(boots, 0.95)),
                boot_median=float(np.median(boots)),
                null_median=float(np.median(null)), null_p95=float(np.quantile(null, 0.95)),
                per_frame_rmsd_to_mean=float(np.sqrt(((X - X.mean(0)) ** 2).sum(-1).mean())))


def state_stability(P: np.ndarray, blocks: np.ndarray, ub: np.ndarray,
                    tol: Tolerances, n_pc: int = 2, min_state_n: int = 20) -> dict:
    """Quarter-wise state assignment and half-vs-half shift along PC1..n_pc.

    The tolerance along each component is the experimental spread of
    structures in the assigned state; states with fewer than ``min_state_n``
    structures (GPCRdb's *Intermediate*) borrow the smaller of the two
    well-populated states so a thin class cannot loosen the test.
    """
    quarters = np.array_split(ub, 4)
    names = tol.states
    C = np.stack([tol.centroids[s][:n_pc] for s in names])
    qmeans = np.array([P[np.isin(blocks, q), :n_pc].mean(0) for q in quarters])
    assign = [names[int(np.argmin(np.linalg.norm(C - m, axis=1)))] for m in qmeans]
    h1, h2 = _halves(ub)
    m1 = P[np.isin(blocks, h1), :n_pc].mean(0)
    m2 = P[np.isin(blocks, h2), :n_pc].mean(0)
    shift = m2 - m1
    state = max(set(assign), key=assign.count)
    # a thinly populated state (GPCRdb's Intermediate: 12 structures) borrows
    # the tighter of the well-populated spreads, so it cannot loosen the test
    if tol.state_counts.get(state, 0) >= min_state_n:
        sd = tol.within_state_sd[state][:n_pc]
    else:
        cands = [tol.within_state_sd[s][:n_pc] for s in names
                 if tol.state_counts.get(s, 0) >= min_state_n]
        sd = np.min(np.stack(cands), axis=0) if cands else tol.within_state_sd[state][:n_pc]
    overall = P[np.isin(blocks, ub), :n_pc].mean(0)
    d_all = np.linalg.norm(C - overall, axis=1)
    order = np.argsort(d_all)
    return dict(quarter_means=qmeans.tolist(), quarter_assign=assign, state=state,
                same_state=len(set(assign)) == 1, half_shift=shift.tolist(),
                tolerance=np.asarray(sd, float).tolist(),
                shift_over_tolerance=(np.abs(shift) / sd).tolist(),
                nearest_other_state_gap=float(d_all[order[1]]) if len(order) > 1 else float('nan'),
                pass_=bool(len(set(assign)) == 1 and np.all(np.abs(shift) <= sd)))



def ensemble_width(X: np.ndarray, blocks: np.ndarray, ub: np.ndarray, n_atoms: int,
                   sub: int = 400, chunks: Sequence[int] = (8, 16), n_null: int = 6,
                   seed: int = 0) -> dict:
    """Distance between the two halves' frame-frame RMSD distributions.

    This is the production observable (the distribution of inter-frame
    distances within an ensemble) with two changes: the two ensembles are
    independent halves of the raw low-temperature data rather than
    overlapping MBAR resamples, and the distance is reported as Wasserstein-1
    in angstroms -- how far the distribution moved -- beside the JSD.
    """
    rng = np.random.default_rng(seed)
    h1, h2 = _halves(ub)

    def dist(bl_a, bl_b):
        ta = _frame_rmsd_triangle(X[np.isin(blocks, bl_a)], n_atoms, sub, rng)
        tb = _frame_rmsd_triangle(X[np.isin(blocks, bl_b)], n_atoms, sub, rng)
        e = np.linspace(min(ta.min(), tb.min()), max(ta.max(), tb.max()), 101)
        return wasserstein1(ta, tb), jsd_1d(ta, tb, e), float(np.median(ta)), float(np.median(tb))

    w, j, ma, mb = dist(h1, h2)
    null = [dist(*_interleaved(ub, c, rng))[:2] for c in chunks for _ in range(n_null)]
    nw, nj = np.array(null).T
    return dict(w1=w, jsd=j, median_interframe_rmsd=[ma, mb],
                null_w1_median=float(np.median(nw)), null_w1_p95=float(np.quantile(nw, .95)),
                null_jsd_median=float(np.median(nj)), null_jsd_p95=float(np.quantile(nj, .95)))


def basin_count(P: np.ndarray, n_pc: int = 3, max_components: int = 4, seed: int = 0) -> dict:
    """How many basins the equilibrated ensemble occupies.

    Sarle's bimodality coefficient on each leading component is the first
    check; if any exceeds 0.555 a Gaussian mixture on the leading components
    is fitted and the BIC-optimal component count is returned.  One basin is
    the common case for a ligand-bound receptor, and population tests are then
    not applicable rather than failed.
    """
    bc = [bimodality_coefficient(P[:, k]) for k in range(min(n_pc, P.shape[1]))]
    if max(bc) < BIMODALITY_CUTOFF:
        return dict(n_basins=1, bimodality=bc, method='bimodality coefficient')
    from sklearn.mixture import GaussianMixture
    Z = P[:, :n_pc]
    bics = []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(k, n_init=3, random_state=seed).fit(Z)
        bics.append(gm.bic(Z))
    return dict(n_basins=int(np.argmin(bics) + 1), bimodality=bc, bic=[float(b) for b in bics],
                method='GMM BIC')


def population_free_energies(P: np.ndarray, blocks: np.ndarray, ub: np.ndarray,
                             n_basins: int, n_pc: int = 3, kT: float = KT_310_KCAL,
                             min_population: float = 0.05, n_boot: int = 300,
                             seed: int = 0) -> dict:
    """Lyman-Zuckerman population test on basins, not on arbitrary cells.

    Assigns every frame to one of ``n_basins`` mixture components fitted on the
    whole equilibrated ensemble, then compares each basin's population between
    the halves as a free-energy difference with a block-bootstrap interval.
    """
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    Z = P[np.isin(blocks, ub), :n_pc]
    gm = GaussianMixture(n_basins, n_init=3, random_state=seed).fit(Z)
    lab_all = gm.predict(P[:, :n_pc])
    h1, h2 = _halves(ub)

    def pops(bl):
        l = lab_all[np.isin(blocks, bl)]
        return np.bincount(l, minlength=n_basins) / l.size

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


def drift_diagnostics(P: np.ndarray, blocks: np.ndarray, ub: np.ndarray,
                      train_sd: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """Per-component autocorrelation, corrected trend test, R-hat, FP error.

    Reported, never gated: a significance test on drift becomes harder to pass
    the longer a converged run continues, so it cannot be a stopping rule.
    """
    out = {}
    bm = np.array([P[blocks == b].mean(0) for b in ub])
    for k in range(P.shape[1]):
        o = ols_corrected(bm[:, k])
        frames = P[np.isin(blocks, ub), k]
        _, _, fp = fp_block_se(frames)
        d = dict(o, total_drift=o['slope'] * (ub.size - 1), rhat=rhat_split(bm[:, k]),
                 fp_se=fp, between_block_sd=float(bm[:, k].std(ddof=1)),
                 within_block_sd=float(np.mean([P[blocks == b, k].std() for b in ub])))
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
          train_sd: Optional[np.ndarray] = None, seed: int = 0) -> dict:
    """Evaluate every gate on the equilibrated blocks and return one verdict.

    ``X`` superposed conserved-atom coordinates (n_frames, n_atoms, 3) in A;
    ``P`` their projections (n_frames, n_pc); ``blocks`` the block of each
    frame.  ``b0`` is the first equilibrated block (default: the first quarter
    is discarded).  The width gate's tolerance is ``width_fraction`` of the
    same-state experimental RMSD.
    """
    ub = equilibrated_blocks(blocks, b0)
    n_atoms = X.shape[1]
    res = dict(n_blocks_total=int(np.unique(blocks).size), b0=int(ub[0]) if ub.size else None,
               n_blocks_equilibrated=int(ub.size), n_frames=int(np.isin(blocks, ub).sum()),
               tolerances=dict(same_state_rmsd=tol.same_state_rmsd,
                               family_same_state_rmsd=tol.family_same_state_rmsd,
                               diff_state_rmsd=tol.diff_state_rmsd,
                               receptor=tol.receptor, receptor_n=tol.receptor_n_structures,
                               width_rmsd=width_fraction * tol.same_state_rmsd))
    if ub.size < 2 * MIN_BLOCKS_PER_HALF:
        res['verdict'] = 'INSUFFICIENT'
        res['reason'] = (f'{ub.size} equilibrated blocks; the panel needs at least '
                         f'{2 * MIN_BLOCKS_PER_HALF} ({MIN_BLOCKS_PER_HALF} per half)')
        return res

    eq = np.isin(blocks, ub)
    ms = mean_structure_drift(X, blocks, ub, seed=seed)
    ms['tolerance'] = tol.same_state_rmsd
    ms['pass_'] = bool(ms['boot_q95'] <= tol.same_state_rmsd)
    ss = state_stability(P, blocks, ub, tol)
    ew = ensemble_width(X, blocks, ub, n_atoms, seed=seed)
    ew['tolerance'] = width_fraction * tol.same_state_rmsd
    ew['pass_'] = bool(ew['w1'] <= ew['tolerance'])
    bc = basin_count(P[eq])
    if bc['n_basins'] > 1:
        pf = population_free_energies(P, blocks, ub, bc['n_basins'], seed=seed)
    else:
        pf = dict(basins=[], kT=KT_310_KCAL, pass_=True, not_applicable=True)
    gates = dict(mean_structure=ms, state=ss, width=ew, basins=bc, populations=pf)
    res['gates'] = gates
    res['diagnostics'] = drift_diagnostics(P, blocks, ub, train_sd)
    failed = [g for g in ('mean_structure', 'state', 'width', 'populations')
              if not gates[g]['pass_']]
    res['failed'] = failed
    res['verdict'] = 'CONVERGED' if not failed else 'NOT CONVERGED'
    return res


def prefix_curve(X: np.ndarray, P: np.ndarray, blocks: np.ndarray, tol: Tolerances,
                 prefixes: Sequence[int], b0_fraction: float = 0.25, **kw) -> List[dict]:
    """The panel as it would have read at each prefix length, in blocks.

    A metric that is fit to be a stopping rule should, once it passes, keep
    passing as the run continues; this is the check.
    """
    out = []
    for n in prefixes:
        sel = blocks < n
        if sel.sum() == 0:
            continue
        r = panel(X[sel], P[sel], blocks[sel], tol, b0=None, **kw) if b0_fraction is None else \
            panel(X[sel], P[sel], blocks[sel], tol,
                  b0=int(np.unique(blocks[sel])[int(np.floor(b0_fraction * np.unique(blocks[sel]).size))]), **kw)
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
             f"{r['n_frames']} frames",
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
        "diagnostics (reported, not gated):",
        f"  {'PC':>4} {'tau':>5} {'n_eff':>5} {'p_naive':>8} {'p_corr':>7} {'R-hat':>5} {'drift':>6} {'FP se':>6}"]
    for k, d in list(r['diagnostics'].items())[:5]:
        lines.append(f"  {k:>4} {d['tau']:>5.1f} {d['neff']:>5.0f} {d['p_naive']:>8.1e} {d['p_corr']:>7.3f} "
                     f"{d['rhat']:>5.2f} {d['total_drift']:>+6.2f} {d['fp_se']:>6.3f}")
    return '\n'.join(lines)
