"""Convergence metrics for PTRE production runs, replayable on cached blocks.

The production criterion declares convergence when all three distance matrices
agree, by Frobenius norm and Jensen-Shannon divergence, against *every* block in
a comparison window (``fultonmarket/retro_convergence.py:build_checks``).  Two
properties of that construction were established by replaying it over
NTSR1_ML-301.hmr.rep0's 81 blocks:

* The window is derived from the equilibration index, so it is wide whenever
  the discard is low and narrow whenever it is high -- and it is not monotone
  in block number, so a block that leaves the window can re-enter.  At its
  widest it spanned 53 blocks.
* At that width the criterion is unsatisfiable.  None of 48 combinations of
  equilibration detector, window policy, collapsed-block exclusion and ratchet
  produced a single passing block.  Restricted instead to a *trailing k blocks*
  the same data converge: k=1 passes at 31 blocks, k=3 at 6 blocks (first at
  block 39), k>=5 never.

So the width of the comparison, not the equilibration estimate, decides the
verdict.  This module makes that dependence explicit and measurable:

``PairwiseCache``          all-pairs Frobenius and JSD for every metric, once
``replay``                 the criterion over the cache under any window policy
``null_distribution``      what the statistic reads when nothing has changed
``calibrated_threshold``   a threshold derived from that null, not asserted
``ca_cache``               alpha-carbon coordinates per block, for the below
``cluster_populations``    Lyman-Zuckerman population convergence
``covariance_overlap``     Romo-Grossfield fluctuation-space overlap

The thresholds currently in use (0.05 Frobenius, 0.10 JSD) have no derivation
in the codebase.  Splitting a single block's 1000-frame resample in half and
comparing the halves with the production statistic -- a comparison in which
nothing has changed but the sample -- gives 0.032 (torsion), 0.043 (alpha
carbon) and 0.036 (contact).  The 0.10 threshold therefore sits about three
times the noise floor, which is defensible; :func:`null_distribution` replaces
that estimate with a bootstrap at the true sample size.

References
----------
Lyman, E.; Zuckerman, D. M. *Biophys. J.* 2006, **91**, 164-172.
Romo, T. D.; Grossfield, A. *JCTC* 2011, **7**, 2464-2472. doi:10.1021/ct2002754
Grossfield, A.; Zuckerman, D. M. *Annu. Rep. Comput. Chem.* 2009, **5**, 23-48.
Lindorff-Larsen, K.; Ferkinghoff-Borg, J. *PLoS ONE* 2009, **4**, e4203.
Tiberti, M. et al. *PLoS Comput. Biol.* 2015, **11**, e1004415.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    'METRICS', 'PairwiseCache', 'replay', 'ReplayResult',
    'null_distribution', 'calibrated_threshold',
    'ca_cache', 'realign_cache', 'cluster_populations', 'covariance_overlap',
]

METRICS = ('torsion', 'alpha_carbon', 'contact')


def _block_ids(run_dir: str) -> List[int]:
    base = os.path.join(run_dir, 'saved_variables')
    return sorted(int(d) for d in os.listdir(base) if d.isdigit())


def _matrix_path(run_dir: str, block: int, metric: str) -> str:
    return os.path.join(run_dir, 'saved_variables', str(block),
                        f'resampled_{metric}_matrix.npy')


# ---------------------------------------------------------------------------
# Tier 0 -- all-pairs cache
# ---------------------------------------------------------------------------

class PairwiseCache:
    """All-pairs Frobenius and JSD between every block's distance matrices.

    Built once (a few minutes for 77 blocks) and reused for every policy
    question, which turns each replay from hours into milliseconds.  The
    statistics come from :func:`~chimpss.fultonmarket.utils.frobenius_norm` and
    :func:`~chimpss.fultonmarket.utils.jsd_distance_matrices` so the numbers are
    identical to what the live criterion computes.

    The cache is written to ``<run_dir>/analysis/pairwise.npz`` and deliberately
    **not** into ``saved_variables/``: ``FultonMarketAnalysis.__init__`` globs
    that directory and calls ``int()`` on every basename, so any non-numeric
    entry there breaks all downstream analysis.
    """

    def __init__(self, blocks: Sequence[int], fro: Dict[str, np.ndarray],
                 jsd: Dict[str, np.ndarray]):
        self.blocks = list(blocks)
        self.fro = fro
        self.jsd = jsd
        self._have: Set[int] = set(self.blocks)

    # -- construction -------------------------------------------------------

    @classmethod
    def default_path(cls, run_dir: str) -> str:
        return os.path.join(run_dir, 'analysis', 'pairwise.npz')

    @classmethod
    def load(cls, path: str) -> 'PairwiseCache':
        d = np.load(path)
        blocks = d['have'].tolist()
        fro = {m: d[f'fro_{m}'] for m in METRICS if f'fro_{m}' in d}
        jsd = {m: d[f'jsd_{m}'] for m in METRICS if f'jsd_{m}' in d}
        return cls(blocks, fro, jsd)

    @classmethod
    def build(cls, run_dir: str, path: Optional[str] = None,
              overwrite: bool = False, _printf=print) -> 'PairwiseCache':
        """Load the cache if present, otherwise compute and save it."""
        path = path or cls.default_path(run_dir)
        if os.path.exists(path) and not overwrite:
            return cls.load(path)

        from chimpss.fultonmarket.utils import frobenius_norm, jsd_distance_matrices

        mats: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in METRICS}
        have: List[int] = []
        for b in _block_ids(run_dir):
            ok = True
            for m in METRICS:
                p = _matrix_path(run_dir, b, m)
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    mats[m][b] = np.load(p)
                else:
                    ok = False
            if ok:
                have.append(b)
        _printf(f'pairwise cache: {len(have)} blocks carry all {len(METRICS)} matrices')

        N = (max(have) + 1) if have else 0
        fro = {m: np.full((N, N), np.nan) for m in METRICS}
        jsd = {m: np.full((N, N), np.nan) for m in METRICS}
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                for m in METRICS:
                    f = frobenius_norm(mats[m][a], mats[m][b])
                    j = jsd_distance_matrices(mats[m][a], mats[m][b])
                    fro[m][a, b] = fro[m][b, a] = f
                    jsd[m][a, b] = jsd[m][b, a] = j

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, have=np.array(have),
                            **{f'fro_{m}': fro[m] for m in METRICS},
                            **{f'jsd_{m}': jsd[m] for m in METRICS})
        _printf(f'pairwise cache written to {path}')
        return cls(have, fro, jsd)

    # -- access -------------------------------------------------------------

    def pair(self, metric: str, a: int, b: int) -> Tuple[float, float]:
        return float(self.fro[metric][a, b]), float(self.jsd[metric][a, b])

    def has(self, block: int) -> bool:
        return block in self._have


# ---------------------------------------------------------------------------
# Tier 1b -- replay under an arbitrary window policy
# ---------------------------------------------------------------------------

class ReplayResult(dict):
    """Per-block verdicts plus the blocks at which every check passed."""

    @property
    def passing(self) -> List[int]:
        return self['passing']

    @property
    def first(self) -> Optional[int]:
        return self['passing'][0] if self['passing'] else None


def replay(cache: PairwiseCache,
           *,
           window_blocks: Optional[int] = None,
           window_fraction: Optional[float] = None,
           discard: Optional[Sequence[Optional[float]]] = None,
           max_equil_fraction: float = 0.75,
           total_n_sims: int = 80,
           minimum_fraction: float = 0.35,
           frobenius_thresh: float = 0.05,
           jsd_thresh: float = 0.10,
           exclude: Optional[Iterable[int]] = None,
           require_gate: bool = True) -> ReplayResult:
    """Re-evaluate the convergence criterion over the cache under one policy.

    Exactly one window policy applies, in precedence order:

    ``window_blocks=k``      compare against the preceding ``k`` blocks
    ``window_fraction=f``    a trailing fraction of completed blocks
    neither                  the production rule, derived from ``discard``

    ``discard`` is the per-block equilibration fraction (0-1) as a detector
    would report it; it is required for the production rule and for the gate.
    ``exclude`` drops blocks from the comparison entirely, which is what
    withholding their matrices would do in the live code.
    """
    excl = set(int(x) for x in (exclude or ()))
    per_block: Dict[int, dict] = {}
    passing: List[int] = []

    hi = max(cache.blocks) if cache.blocks else -1
    for sim_no in range(1, hi + 1):
        if not cache.has(sim_no) or sim_no in excl:
            continue

        d = None if discard is None else discard[sim_no - 1]
        if window_blocks is not None:
            first_valid = max(0, sim_no - int(window_blocks))
        elif window_fraction is not None:
            first_valid = int(sim_no * (1.0 - window_fraction))
        else:
            if d is None:
                continue
            eff = max(1.0 - d, 1.0 - max_equil_fraction)
            first_valid = int((1.0 - eff) * sim_no)

        checks = {
            'past_minimum': sim_no >= total_n_sims * minimum_fraction,
            'equil_ok': (not require_gate) or (d is not None and d < max_equil_fraction),
        }
        for m in METRICS:
            vals = []
            bad = False
            for prev in range(first_valid, sim_no):
                if prev in excl or not cache.has(prev):
                    continue
                f, j = cache.pair(m, sim_no, prev)
                if np.isnan(f) or np.isnan(j):
                    bad = True
                    break
                vals.append((f, j))
            checks[m] = (not bad) and bool(vals) and all(
                f < frobenius_thresh and j < jsd_thresh for f, j in vals)

        checks['STOP'] = all(checks.values())
        per_block[sim_no] = {'window': (first_valid, sim_no - 1), **checks}
        if checks['STOP']:
            passing.append(sim_no)

    return ReplayResult(per_block=per_block, passing=passing)


# ---------------------------------------------------------------------------
# Tier 1a -- null calibration
# ---------------------------------------------------------------------------

def null_distribution(run_dir: str, block: int, metric: str,
                      n_boot: int = 200, n_sub: Optional[int] = None,
                      seed: int = 0) -> np.ndarray:
    """JSD between two resamples drawn from the *same* block.

    Nothing has changed between the two draws but the sample, so the resulting
    distribution is what the statistic reads under the null hypothesis of a
    stationary ensemble.  Any threshold below this distribution's upper tail
    cannot distinguish drift from sampling noise.

    Bootstraps row/column subsets of the block's own distance matrix, which
    preserves the metric's construction exactly; ``n_sub`` defaults to half the
    frames so the two draws are disjoint in expectation.
    """
    from chimpss.fultonmarket.utils import jsd_distance_matrices

    M = np.load(_matrix_path(run_dir, block, metric))
    n = M.shape[0]
    n_sub = int(n_sub or n // 2)
    rng = np.random.default_rng(seed)

    out = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        perm = rng.permutation(n)
        ia, ib = np.sort(perm[:n_sub]), np.sort(perm[n_sub:2 * n_sub])
        out[k] = jsd_distance_matrices(M[np.ix_(ia, ia)], M[np.ix_(ib, ib)])
    return out


def calibrated_threshold(null: Sequence[float], q: float = 0.99,
                         safety: float = 1.0) -> float:
    """``q``-quantile of the null, times a safety factor."""
    return float(np.quantile(np.asarray(null, dtype=float), q) * safety)


# ---------------------------------------------------------------------------
# Tier 1c/1d -- coordinate-space metrics
# ---------------------------------------------------------------------------

def ca_cache(run_dir: str, path: Optional[str] = None, overwrite: bool = False,
             _printf=print) -> Dict[int, np.ndarray]:
    """Alpha-carbon coordinates for every block's resampled trajectory.

    Each ``resampled_trj.dcd`` is ~900 MB of all-atom coordinates, but the
    alpha carbons are a few hundred atoms: the whole run reduces to a few
    hundred MB, which makes clustering and covariance overlap instant
    thereafter.  One pass costs a few minutes of disk read.
    """
    import mdtraj as md

    path = path or os.path.join(run_dir, 'analysis', 'ca_coords.npz')
    if os.path.exists(path) and not overwrite:
        d = np.load(path)
        return {int(k): d[k] for k in d.files}

    out: Dict[int, np.ndarray] = {}
    reference = None          # shared alignment frame, set from the first block
    for b in _block_ids(run_dir):
        bd = os.path.join(run_dir, 'saved_variables', str(b))
        dcd, top = os.path.join(bd, 'resampled_trj.dcd'), os.path.join(bd, 'resampled_top.pdb')
        if not (os.path.exists(dcd) and os.path.exists(top)):
            continue
        ref = md.load(top)
        ca = ref.topology.select('name CA')
        traj = md.load(dcd, top=top, atom_indices=ca)
        # Superpose every block onto ONE shared reference, not onto its own
        # first frame.  Aligning each block internally leaves each in an
        # arbitrary global orientation, and any downstream metric then measures
        # rotation rather than conformation: clustering separates blocks
        # trivially (every block lands in its own cluster, adjacent-block
        # population JSD ~0.99) and covariance overlap is depressed throughout.
        if reference is None:
            reference = traj[0]
        traj.superpose(reference, frame=0)
        out[b] = traj.xyz.astype(np.float32)
        _printf(f'  block {b}: {traj.n_frames} frames x {len(ca)} CA')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in out.items()})
    return out


def realign_cache(ca: Dict[int, np.ndarray],
                  reference: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
    """Put every block's frames in one common orientation.

    Kabsch-superposes all frames onto a single reference (the first frame of
    the lowest-numbered block by default).  Needed whenever a cache was built
    with per-block alignment: without a shared frame, clustering and covariance
    overlap read the arbitrary global orientation of each block rather than its
    conformation.  Operates on the cached coordinates, so it costs seconds
    rather than a re-read of the trajectories.
    """
    keys = sorted(ca)
    if not keys:
        return ca
    if reference is None:
        reference = ca[keys[0]][0]
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref - ref.mean(axis=0)

    out: Dict[int, np.ndarray] = {}
    for b in keys:
        X = np.asarray(ca[b], dtype=np.float64)
        Xc = X - X.mean(axis=1, keepdims=True)
        # per-frame optimal rotation onto ref
        H = np.einsum('fij,ik->fjk', Xc, ref)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.einsum('fij,fjk->fik', U, Vt)))
        D = np.zeros((X.shape[0], 3, 3))
        D[:, 0, 0] = D[:, 1, 1] = 1.0
        D[:, 2, 2] = d
        R = np.einsum('fij,fjk,fkl->fil', U, D, Vt)
        out[b] = np.einsum('fij,fjk->fik', Xc, R).astype(np.float32)
    return out


def cluster_populations(ca: Dict[int, np.ndarray], n_clusters: int = 20,
                        max_frames_per_block: int = 200,
                        seed: int = 0) -> Dict[str, object]:
    """Lyman-Zuckerman cluster-population convergence.

    Cluster a pooled subsample of every block's conformations once, then
    classify each block by proximity to those reference structures.  Each block
    becomes a population histogram over a shared, fixed basis, and convergence
    is the block-to-block stability of that histogram -- a low-dimensional,
    directly interpretable statement about *which* conformers are sampled and
    in what proportion, rather than about a pairwise-distance distribution.

    A block whose resample collapsed shows nearly all its population in a
    single cluster, which makes this an independent detector for collapse.
    """
    from sklearn.cluster import KMeans

    blocks = sorted(ca)
    rng = np.random.default_rng(seed)
    pool, flat = [], {}
    for b in blocks:
        X = ca[b].reshape(ca[b].shape[0], -1)
        flat[b] = X
        take = min(max_frames_per_block, X.shape[0])
        pool.append(X[rng.choice(X.shape[0], take, replace=False)])

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(np.vstack(pool))
    hist = {}
    for b in blocks:
        lab = km.predict(flat[b])
        h = np.bincount(lab, minlength=n_clusters).astype(float)
        hist[b] = h / h.sum()

    # occupancy = clusters holding >1% of a block's population
    occupancy = {b: int((h > 0.01).sum()) for b, h in hist.items()}
    return {'histograms': hist, 'centers': km.cluster_centers_,
            'occupancy': occupancy, 'blocks': blocks, 'n_clusters': n_clusters}


def covariance_overlap(ca_a: np.ndarray, ca_b: np.ndarray) -> float:
    """Romo-Grossfield covariance overlap between two conformational blocks.

    Compares the *fluctuation space* two ensembles explore, which the pairwise
    distance statistics do not test: two blocks can agree on the distribution
    of interatomic distances while fluctuating along different collective
    directions.  Returns 1 for identical covariance and approaches 0 as the
    subspaces and their amplitudes diverge.
    """
    A = ca_a.reshape(ca_a.shape[0], -1)
    B = ca_b.reshape(ca_b.shape[0], -1)
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    CA_ = np.cov(A, rowvar=False)
    CB_ = np.cov(B, rowvar=False)

    la, va = np.linalg.eigh(CA_)
    lb, vb = np.linalg.eigh(CB_)
    la = np.clip(la, 0, None)
    lb = np.clip(lb, 0, None)

    num = np.sum(la) + np.sum(lb) - 2.0 * np.sum(
        np.sqrt(np.outer(la, lb)) * (va.T @ vb) ** 2)
    den = np.sum(la) + np.sum(lb)
    if den <= 0:
        return float('nan')
    return float(1.0 - np.sqrt(max(num, 0.0) / den))
