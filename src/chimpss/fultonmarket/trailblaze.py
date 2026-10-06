"""Thermodynamic trailblazing: build a temperature ladder by walking it.

The ladder FultonMarket samples is guessed up front and then only ever grows
by inserting rungs where exchange fails.  Trailblazing measures it instead.
Starting at ``T_min`` with an equilibrated configuration:

1. sample the enthalpy ``H = U + pV`` at the current rung and take its
   standard deviation ``sigma_H``, the square root of the Fisher information
   metric for a temperature ladder (``g(beta) = Var_beta(H)``; Shenfeld et
   al., *Phys. Rev. E* 2009, **80**, 046705);
2. step to the temperature a fixed thermodynamic length ``dL*`` away,
   ``beta_n = beta_o - dL* / sigma_H``, i.e. ``T_n = T_o / (1 - dL* k_B T_o / sigma_H)``;
3. seed a simulation at ``T_n`` from the previous rung's final configuration,
   sample it, and estimate the exchange acceptance between the two rungs
   from both sample sets with the exact Metropolis criterion;
4. if the acceptance is below ``min_acc`` reject the rung and shrink the step
   (by 1.25 per rejection, as AlGDock's adaptive protocol does); if it is far
   above target and the walk has not reached ``T_max``, reject it and stretch;
   otherwise accept, record, and continue from the new rung;
5. stop when ``T_max`` is reached.

Equidistant spacing in thermodynamic length maximises the total exchange
acceptance along a ladder (Shenfeld Thm. 2), and for Gaussian enthalpy
fluctuations the neighbour acceptance is ``erfc(dL / 2)``, which maps a target
acceptance onto ``dL*`` (``2 erfcinv(A*)``: 0.95 for 50 %, 1.19 for 40 %).
The rung count is an output.  This is the procedure Minh's AlGDock uses for
its temperature ("BC") process (``chimpss.algdock.initialization``), written
against the campaign engine's conventions.

Layout of the module: the metric and placement maths are plain numpy and
scipy (``next_temperature``, ``pair_acceptance``, ``ladder_from_sigma``...);
:class:`Trailblazer` runs the walk against any callable that samples one rung
and is resumable from its JSON record; :class:`OpenMMRungSampler` is that
callable for an OpenMM system, importing openmm/openmmtools lazily so the
rest of the module is testable without them.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from chimpss.analysis.ladder import (
    LADDER_DECIMALS,
    R_KJ,
    length_for_acceptance,
    pair_lengths,
    predicted_acceptance,
)

__all__ = [
    'KB', 'beta', 'target_spacing', 'predicted_acceptance', 'pair_spacing', 'sigma_from_cv',
    'next_temperature', 'pair_acceptance', 'stationary_window', 'ladder_from_sigma',
    'load_ladder_file', 'Seed', 'Rung', 'TrailblazeRecord', 'SeedStore', 'Trailblazer',
    'OpenMMRungSampler', 'RECORD_NAME', 'LADDER_NAME', 'SEED_DIR', 'LADDER_MODES',
    'DIRECTIONS', 'DIRECTION_SIGN', 'SAMPLER_KEYS', 'WALK_KEYS', 'split_kwargs',
]

#: Boltzmann constant in kJ/(mol K); equals ``openmmtools.constants.kB``.
KB = R_KJ

RECORD_NAME = 'trailblaze.json'
LADDER_NAME = 'temperatures.npy'
SEED_DIR = 'seeds'

#: ``dL*`` for a target acceptance: ``2 erfcinv(A*)``.
target_spacing = length_for_acceptance

LADDER_MODES = ('geometric', 'trailblaze', 'file')

#: Walk directions: ``'up'`` from ``T_min``, ``'down'`` from ``T_max``, ``'both'`` = an up walk,
#: a down walk seeded from its hottest rung, and a reconciled ladder (:func:`reconcile`).
DIRECTIONS = ('up', 'down', 'both')
DIRECTION_SIGN = {'up': 1, 'down': -1}

#: Keyword arguments of :class:`OpenMMRungSampler` and :class:`Trailblazer` that a
#: caller may pass together as one ``trailblaze_kwargs`` dict.
SAMPLER_KEYS = ('equil_ps', 'first_equil_ps', 'sample_ps', 'sample_interval_ps', 'max_sample_ps',
                'neff_min', 'collision_rate_per_ps', 'pressure_bar', 'ramp_K', 'ramp_ps')
WALK_KEYS = ('target_acc', 'min_acc', 'max_acc', 'shrink', 'max_retries', 'max_rungs', 'max_dT',
             'min_dT', 'sigma_window', 'sigma_source', 'force', 'direction')


def split_kwargs(kwargs: Optional[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Split one ``trailblaze_kwargs`` dict into sampler and walk keyword arguments."""
    kwargs = dict(kwargs or {})
    unknown = sorted(set(kwargs) - set(SAMPLER_KEYS) - set(WALK_KEYS))
    if unknown:
        raise ValueError(f'unknown trailblaze settings {unknown}; known: {SAMPLER_KEYS + WALK_KEYS}')
    return ({k: v for k, v in kwargs.items() if k in SAMPLER_KEYS},
            {k: v for k, v in kwargs.items() if k in WALK_KEYS})


def beta(T):
    """Inverse temperature in mol/kJ."""
    return 1.0 / (KB * np.asarray(T, dtype=np.float64))


def pair_spacing(T_o: float, T_n: float, sigma_o: float, sigma_n: float) -> float:
    """Thermodynamic length between two rungs from their enthalpy fluctuations.

    ``0.5 (sigma_o + sigma_n) |beta_o - beta_n|``: the mean of the two rungs'
    metric estimates times the inverse-temperature gap (Shenfeld's two-sided
    eq. 11 uses the root mean square of the two; the difference is second
    order in ``sigma_n - sigma_o``).  Unsigned, so a downward pair measures
    the same as the upward one; ``chimpss.analysis.ladder.pair_lengths``
    keeps the sign for ascending ladders.
    """
    return abs(float(pair_lengths(np.array([T_o, T_n]), np.array([sigma_o, sigma_n]))[0]))


def sigma_from_cv(T, cv):
    """``sigma_H`` for a heat capacity ``cv`` in units of k_B: ``k_B T sqrt(cv)``."""
    return KB * np.asarray(T, dtype=np.float64) * np.sqrt(cv)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------

def next_temperature(T_o: float, sigma_o: float, dL: float, T_end: Optional[float] = None,
                     step_scale: float = 1.0, max_dT: float = 10.0, min_dT: float = 0.1,
                     even_out_fraction: float = 0.5, direction: int = 1,
                     T_max: Optional[float] = None) -> Tuple[float, bool]:
    """The next rung a thermodynamic length ``dL * step_scale`` from ``T_o`` towards ``T_end``.

    Exact in ``beta``: ``beta_n = beta_o - direction dL step_scale / sigma_o``,
    i.e. ``T_n = T_o / (1 - direction dL step_scale k_B T_o / sigma_o)`` with
    ``direction = +1`` walking up in temperature and ``-1`` walking down.  The
    step is capped at ``max_dT`` kelvin (a huge ``sigma`` estimate from a
    transient must not launch the walk) and refused below ``min_dT`` (a tiny
    step means ``sigma`` is not a stationary fluctuation, or the ladder would
    collide at the two decimals the analysis layer uses).

    Endpoint handling: when the step reaches ``T_end`` the rung is placed
    there and ``crossed`` is True.  When it would leave a sliver shorter than
    ``even_out_fraction`` of a step, the remaining distance is split in two
    equal rungs instead, so the last pair is never longer than a step and
    never a stub.  ``T_max`` is accepted as a deprecated alias of ``T_end``.
    """
    if T_end is None:
        if T_max is None:
            raise TypeError('next_temperature() needs T_end')
        T_end = T_max
    if not np.isfinite(sigma_o) or sigma_o <= 0:
        raise ValueError(f'sigma_H must be finite and positive, got {sigma_o}')
    if dL <= 0 or step_scale <= 0:
        raise ValueError('dL and step_scale must be positive')
    d = 1 if direction >= 0 else -1
    if d * (T_end - T_o) <= 0:
        raise ValueError(f'T_o={T_o} is not {"below" if d > 0 else "above"} T_end={T_end}')
    x = dL * step_scale * KB * T_o / sigma_o
    denom = 1.0 - d * x
    T_n = T_o + d * max_dT if denom <= 0 else T_o / denom
    dT = min(abs(T_n - T_o), max_dT)
    if dT < min_dT:
        raise ValueError(f'step {dT:.4f} K below min_dT={min_dT} K: sigma_H={sigma_o:.1f} at {T_o:.2f} K '
                         f'is too large for dL={dL * step_scale:.3f}; the sample is not stationary')
    T_n = T_o + d * dT
    if d * (T_n - T_end) >= 0:
        return float(T_end), True
    if d * (T_end - T_n) < even_out_fraction * dT:
        return float(T_o + 0.5 * (T_end - T_o)), False
    return float(T_n), False


def _reweight(H: np.ndarray, T_target: float, T_src: Optional[float]) -> np.ndarray:
    """Normalised importance weights taking samples drawn at ``T_src`` to ``T_target``."""
    if T_src is None or float(T_src) == float(T_target):
        return np.full(H.size, 1.0 / H.size)
    logw = -(float(beta(T_target)) - float(beta(T_src))) * (H - H.mean())
    w = np.exp(logw - logw.max())
    return w / w.sum()


def pair_acceptance(H_o: np.ndarray, H_n: np.ndarray, T_o: float, T_n: float,
                    max_pairs: int = 250_000, rng: Optional[np.random.Generator] = None,
                    T_o_src: Optional[float] = None, T_n_src: Optional[float] = None,
                    return_ess: bool = False):
    """Exact Metropolis exchange acceptance between two rungs from their samples.

    Every configuration sampled at ``T_o`` is paired with every configuration
    sampled at ``T_n``: ``mean(min(1, exp(-(beta_o - beta_n) (H_n[j] - H_o[i]))))``,
    which is AlGDock's ``exp(-u01 - u10 + u00 + u11)`` over the full outer
    product.  Above ``max_pairs`` the pairs are subsampled at random.

    ``T_o_src`` / ``T_n_src`` say that a sample set was drawn at another
    temperature than the rung it stands for (a walked rung near a reconciled
    one): that set is then importance-reweighted, ``w ~ exp(-(beta_target -
    beta_src) H)``, the same exponential reweighting
    :func:`chimpss.analysis.ladder.virtual_pair_acceptance` applies to saved
    blocks.  ``return_ess=True`` also returns the smaller of the two sets'
    effective-sample fractions, 1 when nothing was reweighted.
    """
    H_o = np.asarray(H_o, dtype=np.float64)
    H_n = np.asarray(H_n, dtype=np.float64)
    H_o = H_o[np.isfinite(H_o)]
    H_n = H_n[np.isfinite(H_n)]
    if H_o.size == 0 or H_n.size == 0:
        raise ValueError('both sample sets must contain finite values')
    db = float(beta(T_o) - beta(T_n))
    w_o = _reweight(H_o, T_o, T_o_src)
    w_n = _reweight(H_n, T_n, T_n_src)
    ess = min(1.0 / (H_o.size * np.sum(w_o ** 2)), 1.0 / (H_n.size * np.sum(w_n ** 2)))
    total = H_o.size * H_n.size
    if total <= max_pairs:
        delta = db * (H_n[None, :] - H_o[:, None])
        acc = float(np.sum(w_o[:, None] * w_n[None, :] * np.minimum(1.0, np.exp(-delta))))
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        i = rng.choice(H_o.size, size=max_pairs, p=w_o)
        j = rng.choice(H_n.size, size=max_pairs, p=w_n)
        delta = db * (H_n[j] - H_o[i])
        acc = float(np.minimum(1.0, np.exp(-delta)).mean())
    acc = min(1.0, acc)                      # weights sum to one only to rounding
    return (acc, float(ess)) if return_ess else acc


def stationary_window(H: np.ndarray, min_keep: int = 20) -> Tuple[int, float, float]:
    """``(t0, g, Neff)`` for an enthalpy series: where the stationary part starts.

    Uses ``pymbar.timeseries.detect_equilibration`` when pymbar is importable
    (statistical inefficiency ``g`` and the maximal number of effective
    samples); otherwise a two-halves drift test with ``g = 1``.  Never keeps
    fewer than ``min_keep`` samples.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.size
    if n < 4:
        return 0, 1.0, float(n)
    try:
        from pymbar.timeseries import detect_equilibration
        t0, g, neff = detect_equilibration(H)
        t0, g, neff = int(t0), float(g), float(neff)
    except ImportError:
        half = n // 2
        a, b = H[:half], H[half:]
        se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        t0 = half if (se > 0 and abs(a.mean() - b.mean()) > 2.0 * se) else 0
        g = 1.0
        neff = float(n - t0)
    if n - t0 < min_keep:
        t0 = max(0, n - min_keep)
        neff = min(neff, float(n - t0) / max(g, 1.0))
    return t0, g, neff


def ladder_from_sigma(T_min: float, T_max: float, sigma_fn: Callable[[float], float], dL: float,
                      max_rungs: int = 10_000, direction: str = 'up', **step_kwargs) -> np.ndarray:
    """Deterministic walk on a known ``sigma_H(T)``: the ladder trailblazing converges to.

    ``direction='down'`` starts at ``T_max`` and steps down; the result is
    returned ascending either way.
    """
    if direction not in DIRECTION_SIGN:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    sign = DIRECTION_SIGN[direction]
    T_start, T_end = (float(T_min), float(T_max)) if sign > 0 else (float(T_max), float(T_min))
    T = [T_start]
    crossed = False
    while not crossed:
        if len(T) >= max_rungs:
            raise RuntimeError(f'more than {max_rungs} rungs; dL={dL} is too small for this sigma')
        T_n, crossed = next_temperature(T[-1], float(sigma_fn(T[-1])), dL, T_end, direction=sign, **step_kwargs)
        T.append(T_n)
    return np.asarray(T if sign > 0 else T[::-1])


def ramp_stages(T_from: float, T: float, ramp_K: float) -> List[float]:
    """Temperatures visited before ``T`` when taking a configuration from ``T_from`` in ``ramp_K`` stages.

    Empty when the two are within ``ramp_K`` (one jump, as the upward walk
    has always done); otherwise ``T_from, T_from + ramp_K, ...`` up to the
    last stage strictly before ``T``, so no jump exceeds ``ramp_K``.
    """
    T_from, T, ramp_K = float(T_from), float(T), float(ramp_K)
    if ramp_K <= 0:
        raise ValueError('ramp_K must be positive')
    if abs(T - T_from) <= ramp_K:
        return []
    step = ramp_K if T > T_from else -ramp_K
    stages = []
    x = T_from
    while (T - x) * step > 1e-9:
        stages.append(round(x, 6))
        x += step
    return stages


def load_ladder_file(path: str) -> np.ndarray:
    """A ladder from ``.npy`` or a text file, one temperature per line, validated."""
    T = np.load(path) if path.endswith('.npy') else np.loadtxt(path)
    T = np.atleast_1d(np.asarray(T, dtype=np.float64)).ravel()
    if T.size < 2:
        raise ValueError(f'{path}: a ladder needs at least two rungs')
    if not np.all(np.isfinite(T)) or np.any(T <= 0):
        raise ValueError(f'{path}: temperatures must be finite and positive')
    if np.any(np.diff(T) <= 0):
        raise ValueError(f'{path}: temperatures must be strictly increasing')
    if np.unique(np.round(T, LADDER_DECIMALS)).size != T.size:
        raise ValueError(f'{path}: two rungs coincide at {LADDER_DECIMALS} decimals')
    return T


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Seed:
    """A configuration handed from one rung to the next: positions and box in nm, velocities in nm/ps."""
    positions: np.ndarray
    box_vectors: np.ndarray
    velocities: Optional[np.ndarray] = None


@dataclass
class Rung:
    """One rung of a walked or reconciled ladder.

    ``acc_est`` and ``dL`` always describe the pair with the rung this one was
    seeded from: the previous rung in walk order, which is the rung below for
    an upward walk and the rung above for a downward one.
    :meth:`TrailblazeRecord.pair_acc` resolves that into ascending pairs.
    """
    T: float
    sigma: float            #: this rung's own estimate
    mean_H: float
    n_samples: int
    g: float
    neff: float
    acc_est: Optional[float]
    acc_pred: Optional[float]
    dL: Optional[float]
    step_scale: float
    rejects: int
    seed_path: Optional[str]
    wall_s: float
    sigma_step: Optional[float] = None   #: what the step from here used (pooled or slope-based)
    cv_slope: Optional[float] = None     #: d<H>/dT over the trailing rungs, kJ/mol/K, when available
    walk_index: Optional[int] = None     #: position in walk order (differs from the index for a down walk)
    source: Optional[str] = None         #: reconciled ladders: which walk the seed came from
    source_index: Optional[int] = None   #: ... and that walk's rung index
    source_T: Optional[float] = None     #: ... and its temperature
    sigma_fit: Optional[float] = None    #: reconciled ladders: the fitted metric at this rung


#: Keys a record written by an older version may lack, and what it meant then.
LEGACY_INPUT_DEFAULTS = {'sigma_source': 'pooled', 'direction': 'up'}


def inputs_match(on_disk: Dict[str, object], now: Dict[str, object]) -> bool:
    """Whether a record's inputs equal the current ones, reading legacy defaults for missing keys."""
    for k, v in now.items():
        if on_disk.get(k, LEGACY_INPUT_DEFAULTS.get(k)) != v:
            return False
    return all(k in now for k in on_disk)


@dataclass
class TrailblazeRecord:
    inputs: Dict[str, object]
    rungs: List[Rung] = field(default_factory=list)
    rejected: List[Dict[str, object]] = field(default_factory=list)
    complete: bool = False
    version: int = 2
    prepare: Dict[str, object] = field(default_factory=dict)   #: how the starting configuration was made
    consensus: Optional[Dict[str, object]] = None              #: set on a reconciled ('both') ladder

    @property
    def temperatures(self) -> np.ndarray:
        return np.array([r.T for r in self.rungs], dtype=np.float64)

    @property
    def direction(self) -> str:
        return str(self.inputs.get('direction', LEGACY_INPUT_DEFAULTS['direction']))

    def _pairs(self, attr: str) -> np.ndarray:
        """Per ascending pair ``(k, k+1)``: a rung attribute describing that pair."""
        K = len(self.rungs)
        if K < 2:
            return np.array([], dtype=np.float64)
        # up / both: rung k+1 was seeded from rung k; down: rung k was seeded from rung k+1
        owners = self.rungs[1:] if self.direction != 'down' else self.rungs[:-1]
        return np.array([np.nan if getattr(r, attr) is None else getattr(r, attr) for r in owners], dtype=np.float64)

    def pair_acc(self) -> np.ndarray:
        """Measured acceptance of each ascending neighbour pair."""
        return self._pairs('acc_est')

    def pair_dL(self) -> np.ndarray:
        """Realised thermodynamic length of each ascending neighbour pair."""
        return self._pairs('dL')

    @property
    def meta(self) -> Dict[str, object]:
        dLs = [r.dL for r in self.rungs if r.dL is not None]
        accs = [r.acc_est for r in self.rungs if r.acc_est is not None]
        m = {
            'ladder': 'trailblaze',
            'direction': self.direction,
            'target_acc': self.inputs.get('target_acc'),
            'min_acc': self.inputs.get('min_acc'),
            'max_acc': self.inputs.get('max_acc'),
            'n_rungs': len(self.rungs),
            'L_total': float(np.sum(dLs)) if dLs else 0.0,
            'acc_min': float(np.min(accs)) if accs else None,
            'acc_median': float(np.median(accs)) if accs else None,
            'complete': self.complete,
        }
        if self.consensus:
            m['K_up'] = self.consensus.get('K_up')
            m['K_down'] = self.consensus.get('K_down')
        return m

    def to_dict(self) -> Dict[str, object]:
        d = {'version': self.version, 'complete': self.complete, 'inputs': self.inputs,
             'rungs': [asdict(r) for r in self.rungs], 'rejected': self.rejected}
        if self.prepare:
            d['prepare'] = self.prepare
        if self.consensus is not None:
            d['consensus'] = self.consensus
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'TrailblazeRecord':
        return cls(inputs=dict(d['inputs']), rungs=[Rung(**r) for r in d['rungs']],
                   rejected=list(d.get('rejected', [])), complete=bool(d['complete']),
                   version=int(d.get('version', 1)), prepare=dict(d.get('prepare') or {}),
                   consensus=d.get('consensus'))

    def save(self, path: str) -> None:
        """Atomic: written through a per-process temporary name then renamed."""
        tmp = f'{path}.tmp.{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> 'TrailblazeRecord':
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load_if_complete(cls, path: str) -> Optional['TrailblazeRecord']:
        if not os.path.exists(path):
            return None
        rec = cls.load(path)
        return rec if rec.complete else None


#: File prefix of a downward walk's seeds in walk order, before they are re-indexed ascending.
WALK_PREFIX = 'walk_'


class SeedStore:
    """Per-rung seeds and samples under ``<out_dir>/seeds/<prefix>XXX.npz``.

    The default prefix ``rung_`` is the contract FultonMarket reads: index
    ``k`` seeds rung ``k`` of the ascending ladder.  A downward walk stores
    its seeds in walk order under :data:`WALK_PREFIX` and links them to
    ``rung_`` names, reversed, when it finishes.
    """

    def __init__(self, out_dir: str, prefix: str = 'rung_'):
        self.dir = os.path.join(out_dir, SEED_DIR)
        self.prefix = prefix

    def path(self, index: int) -> str:
        return os.path.join(self.dir, f'{self.prefix}{index:03d}.npz')

    def link(self, index: int, src_path: str, velocity_scale: Optional[float] = None) -> str:
        """Make ``path(index)`` hold the seed at ``src_path``: a hard link, or a copy.

        ``velocity_scale`` rescales the stored velocities (a seed handed to a
        rung at another temperature: ``sqrt(T_new / T_src)`` keeps the
        Maxwell-Boltzmann distribution exact), which forces a copy.  Written
        through a temporary name and renamed, so it can be re-run after a crash.
        """
        import shutil
        os.makedirs(self.dir, exist_ok=True)
        path = self.path(index)
        tmp = f'{path}.tmp.{os.getpid()}.npz'
        if velocity_scale is None or velocity_scale == 1.0:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
                os.link(src_path, tmp)
            except OSError:
                shutil.copyfile(src_path, tmp)
        else:
            with np.load(src_path) as z:
                arrays = {k: z[k] for k in z.files}
            if 'velocities' in arrays:
                arrays['velocities'] = (arrays['velocities'] * float(velocity_scale)).astype(np.float32)
            np.savez(tmp, **arrays)
        os.replace(tmp, path)
        return path

    def save(self, index: int, seed: Seed, H: np.ndarray) -> str:
        os.makedirs(self.dir, exist_ok=True)
        path = self.path(index)
        tmp = f'{path}.tmp.{os.getpid()}.npz'
        arrays = dict(positions=np.asarray(seed.positions, dtype=np.float32),
                      box_vectors=np.asarray(seed.box_vectors, dtype=np.float64),
                      H=np.asarray(H, dtype=np.float64))
        if seed.velocities is not None:
            arrays['velocities'] = np.asarray(seed.velocities, dtype=np.float32)
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        return path

    def load(self, index: int) -> Tuple[Seed, np.ndarray]:
        with np.load(self.path(index)) as z:
            seed = Seed(positions=z['positions'].astype(np.float64),
                        box_vectors=z['box_vectors'].astype(np.float64),
                        velocities=z['velocities'].astype(np.float64) if 'velocities' in z else None)
            return seed, z['H'].astype(np.float64)

    def has_all(self, n: int) -> bool:
        return n > 0 and all(os.path.exists(self.path(i)) for i in range(n))


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

#: A rung sampler: ``(T_kelvin, seed) -> (H_samples kJ/mol, final seed, meta)``.
RungSampler = Callable[[float, Seed], Tuple[np.ndarray, Seed, Dict[str, object]]]


class Trailblazer:
    """Walk the ladder from ``T_min`` to ``T_max`` with a rung sampler.

    Parameters
    ----------
    sampler
        Samples one rung; see :data:`RungSampler`.  :class:`OpenMMRungSampler`
        for a real system; any callable with the same signature for tests.
    target_acc, min_acc, max_acc
        The neighbour acceptance aimed for (sets ``dL* = 2 erfcinv``), the
        floor below which a proposed rung is rejected and the step shrunk,
        and the ceiling above which it is rejected and the step stretched.
    shrink, max_retries
        Step factor per rejection and the number of rejections after which a
        rung is accepted anyway, with a warning, so the walk terminates.
    max_rungs, max_dT, min_dT
        Guards: see :func:`next_temperature`.
    sigma_window
        The step from a rung uses ``sigma_H`` pooled (effective-sample
        weighted) over the last ``sigma_window`` accepted rungs rather than
        that rung's own estimate.  One rung's ~50 effective samples put ~10 %
        noise on its sigma and hence on the step; the metric changes by well
        under 1 % per kelvin, so five neighbouring rungs are a better
        estimate of it than one.  1 disables pooling.
    sigma_source
        ``'pooled'`` (default) steps on the pooled enthalpy fluctuation.
        ``'slope'`` steps on the fluctuation-dissipation estimate instead:
        ``C_V = d<H>/dT`` from a straight-line fit of the rung means over the
        last ``2 * sigma_window`` rungs, ``sigma = sqrt(k_B T^2 C_V)``.  A
        single trajectory sampled for 100-300 ps sees the fast fluctuations
        but not the slower configurational variance that a production block
        samples across many exchanging walkers, so the variance route ran
        5-10 % below the production metric on NTSR1 while the slope route
        tracked it within 1-4 %.  Falls back to the pooled value until four
        rungs exist or when the fit is implausible.
    out_dir
        Where ``trailblaze.json``, ``temperatures.npy`` and ``seeds/`` go.
        Required for resume; None keeps everything in memory.
    direction
        ``'up'`` walks from ``T_min`` to ``T_max`` (the default and AlGDock's
        own direction); ``'down'`` starts at ``T_max`` and walks to ``T_min``.
        Thermodynamic length is a distance, so both place the same ladder up
        to the per-rung noise of ``sigma_H``; a down walk hands every rung a
        configuration annealed from an equilibrated hot one.  Whatever the
        direction, the finished record, ``temperatures.npy`` and the
        ``seeds/rung_XXX.npz`` files are ascending in temperature.  A
        downward walk's starting configuration must already be at ``T_max``
        or be heated there by the sampler's ``prepare(T, seed, T_from=...)``.
    """

    def __init__(self, sampler: RungSampler, T_min: float, T_max: float, target_acc: float = 0.40,
                 min_acc: float = 0.30, max_acc: float = 0.75, shrink: float = 1.25,
                 max_retries: int = 6, max_rungs: int = 400, max_dT: float = 10.0,
                 min_dT: float = 0.1, sigma_window: int = 5, sigma_source: str = 'pooled',
                 out_dir: Optional[str] = None, printf: Callable = print, force: bool = False,
                 direction: str = 'up', seed_from: Optional[float] = None):
        if not T_max > T_min > 0:
            raise ValueError('need 0 < T_min < T_max')
        if not 0 < min_acc <= target_acc <= max_acc < 1:
            raise ValueError('need 0 < min_acc <= target_acc <= max_acc < 1')
        if direction not in DIRECTION_SIGN:
            raise ValueError(f"direction must be 'up' or 'down' for one walk, got {direction!r}; "
                             f"'both' is run by run_trailblaze()")
        self.sampler = sampler
        self.T_min, self.T_max = float(T_min), float(T_max)
        self.target_acc, self.min_acc, self.max_acc = float(target_acc), float(min_acc), float(max_acc)
        self.shrink, self.max_retries = float(shrink), int(max_retries)
        self.max_rungs, self.max_dT, self.min_dT = int(max_rungs), float(max_dT), float(min_dT)
        self.sigma_window = max(1, int(sigma_window))
        if sigma_source not in ('pooled', 'slope'):
            raise ValueError("sigma_source must be 'pooled' or 'slope'")
        self.sigma_source = sigma_source
        self.direction = direction
        self.sign = DIRECTION_SIGN[direction]
        self.T_start, self.T_end = (self.T_min, self.T_max) if self.sign > 0 else (self.T_max, self.T_min)
        #: temperature the initial seed sits at, when it is not T_start (the sampler then ramps);
        #: None means "already at T_start", which skips the ramp.
        self.seed_from = None if seed_from is None else float(seed_from)
        self.out_dir = out_dir
        self.printf = printf
        self.force = force
        self.dL = float(target_spacing(self.target_acc))
        self.store = SeedStore(out_dir) if out_dir else None
        self.walk_store = (SeedStore(out_dir, prefix=WALK_PREFIX if direction == 'down' else 'rung_')
                           if out_dir else None)

    # -- bookkeeping -------------------------------------------------------

    def inputs(self) -> Dict[str, object]:
        d = dict(T_min=self.T_min, T_max=self.T_max, target_acc=self.target_acc, min_acc=self.min_acc,
                 max_acc=self.max_acc, shrink=self.shrink, max_retries=self.max_retries,
                 max_rungs=self.max_rungs, max_dT=self.max_dT, min_dT=self.min_dT, dL=self.dL,
                 sigma_window=self.sigma_window, sigma_source=self.sigma_source, direction=self.direction)
        describe = getattr(self.sampler, 'describe', None)
        if callable(describe):
            d['sampler'] = describe()
        return d

    @property
    def record_path(self) -> Optional[str]:
        return os.path.join(self.out_dir, RECORD_NAME) if self.out_dir else None

    def _persist(self, rec: TrailblazeRecord) -> None:
        if self.record_path:
            os.makedirs(self.out_dir, exist_ok=True)
            rec.save(self.record_path)

    def _open_record(self) -> Tuple[TrailblazeRecord, bool]:
        """Existing record (resume) or a fresh one; the flag says whether it was resumed."""
        fresh = TrailblazeRecord(inputs=self.inputs())
        if not self.record_path or not os.path.exists(self.record_path):
            return fresh, False
        rec = TrailblazeRecord.load(self.record_path)
        if not inputs_match(rec.inputs, fresh.inputs) and not self.force:
            raise RuntimeError(
                f'{self.record_path} was written with different inputs; pass force=True to '
                f'resume anyway, or remove it to restart.\n  on disk: {rec.inputs}\n  now:     {fresh.inputs}')
        return rec, True

    # -- the walk ----------------------------------------------------------

    def pooled_sigma(self, rungs: List[Rung]) -> float:
        """Effective-sample-weighted pooled ``sigma_H`` of the last ``sigma_window`` rungs."""
        window = rungs[-self.sigma_window:]
        w = np.array([max(r.neff, 1.0) for r in window])
        var = np.array([r.sigma ** 2 for r in window])
        return float(np.sqrt((w * var).sum() / w.sum()))

    def slope_cv(self, rungs: List[Rung]) -> Optional[float]:
        """``d<H>/dT`` in kJ/mol/K from the last ``2 * sigma_window`` rung means, or None."""
        window = rungs[-2 * self.sigma_window:]
        if len(window) < 4:
            return None
        T = np.array([r.T for r in window])
        H = np.array([r.mean_H for r in window])
        cv = float(np.polyfit(T, H, 1)[0])
        return cv if cv > 0 else None

    def step_sigma(self, rungs: List[Rung]) -> Tuple[float, Optional[float]]:
        """``(sigma for the next step, C_V slope used or None)`` per ``sigma_source``."""
        pooled = self.pooled_sigma(rungs)
        cv = self.slope_cv(rungs)
        if self.sigma_source != 'slope' or cv is None:
            return pooled, cv
        sigma = float(np.sqrt(KB * rungs[-1].T ** 2 * cv))
        if not 0.5 < sigma / pooled < 2.0:        # implausible fit: lagging means or a transient
            return pooled, cv
        return sigma, cv

    def run(self, initial_seed: Seed) -> TrailblazeRecord:
        rec, resumed = self._open_record()
        if rec.complete:
            self.printf(f'trailblaze: complete ladder of {len(rec.rungs)} rungs found in {self.out_dir}')
            return rec
        # a record is kept in walk order until it is finalised; a finished-but-unfinalised
        # down record (killed between the last rung and the finalisation) is still in walk order
        if self.direction == 'down':
            rec.rungs = self._walk_order(rec.rungs)
        walk = rec.rungs

        if walk:
            last = walk[-1]
            last_index = last.walk_index if last.walk_index is not None else len(walk) - 1
            if self.walk_store is None or not os.path.exists(self.walk_store.path(last_index)):
                raise RuntimeError('cannot resume: the last accepted rung has no saved seed')
            seed, H_prev = self.walk_store.load(last_index)
            self.printf(f'trailblaze: resuming after rung {last_index} at {last.T:.2f} K')
        else:
            self.printf(f'trailblaze: {self.T_start:.2f} -> {self.T_end:.2f} K ({self.direction}), target '
                        f'acceptance {self.target_acc:.2f} (dL* = {self.dL:.3f}), reject below {self.min_acc:.2f}')
            t = time.time()
            initial_seed = self._prepare(rec, initial_seed)
            H_prev, seed, meta = self.sampler(self.T_start, initial_seed)
            rung = self._make_rung(self.T_start, H_prev, meta, acc=None, acc_pred=None, dL=None,
                                   step_scale=1.0, rejects=0, wall=time.time() - t)
            rung.walk_index = 0
            rung.seed_path = self._save_seed(0, seed, H_prev)
            walk.append(rung)
            rung.sigma_step, rung.cv_slope = self.step_sigma(walk)
            self._persist(rec)
            self._report(0, rung)

        stage = 0
        crossed = self.sign * (walk[-1].T - self.T_end) >= 0
        while not crossed:
            if len(walk) >= self.max_rungs:
                raise RuntimeError(f'trailblaze: {self.max_rungs} rungs without reaching T_end={self.T_end}; '
                                   f'raise max_rungs or target_acc')
            prev = walk[-1]
            sigma_o = prev.sigma_step if prev.sigma_step is not None else self.step_sigma(walk)[0]
            step_scale = self.shrink ** (-stage)
            T_n, crossed = next_temperature(prev.T, sigma_o, self.dL, self.T_end, step_scale=step_scale,
                                            max_dT=self.max_dT, min_dT=self.min_dT, direction=self.sign)
            t = time.time()
            H_n, seed_n, meta = self.sampler(T_n, seed)
            wall = time.time() - t
            sigma_n = float(np.std(H_n, ddof=1))
            acc = pair_acceptance(H_prev, H_n, prev.T, T_n)
            dL_real = pair_spacing(prev.T, T_n, sigma_o, sigma_n)
            acc_pred = float(predicted_acceptance(dL_real))
            n_rejects = abs(stage)

            if acc < self.min_acc and n_rejects < self.max_retries:
                stage += 1
                self._reject(rec, walk, T_n, acc, sigma_n, 'below min_acc, shrinking step')
                crossed = False
                continue
            if acc > self.max_acc and not crossed and n_rejects < self.max_retries:
                stage -= 1
                self._reject(rec, walk, T_n, acc, sigma_n, 'above max_acc, stretching step')
                continue
            if n_rejects >= self.max_retries:
                self.printf(f'trailblaze: WARNING accepting {T_n:.2f} K with acceptance {acc:.3f} after '
                            f'{n_rejects} rejections')

            rung = self._make_rung(T_n, H_n, meta, acc=acc, acc_pred=acc_pred, dL=dL_real,
                                   step_scale=step_scale, rejects=n_rejects, wall=wall)
            index = len(walk)
            rung.walk_index = index
            rung.seed_path = self._save_seed(index, seed_n, H_n)
            walk.append(rung)
            rung.sigma_step, rung.cv_slope = self.step_sigma(walk)
            self._persist(rec)
            self._report(index, rung)
            H_prev, seed, stage = H_n, seed_n, 0

        self._finalise(rec, walk)
        m = rec.meta
        self.printf(f"trailblaze: done, {m['n_rungs']} rungs ({self.direction}), total thermodynamic length "
                    f"{m['L_total']:.2f}, neighbour acceptance min {m['acc_min']:.3f} median {m['acc_median']:.3f}")
        return rec

    def _walk_order(self, rungs: List[Rung]) -> List[Rung]:
        """The rungs of a downward record in the order they were walked (hot to cold)."""
        if not rungs or rungs[0].walk_index is None:
            return rungs                      # in-progress records are stored in walk order
        return sorted(rungs, key=lambda r: r.walk_index)

    def _prepare(self, rec: TrailblazeRecord, initial_seed: Seed) -> Seed:
        """Equilibrate the starting configuration at ``T_start`` through the sampler's ``prepare``."""
        prepare = getattr(self.sampler, 'prepare', None)
        if not callable(prepare):
            return initial_seed
        T_from = self.seed_from
        if T_from is None and self.direction == 'down':
            T_from = self.T_min              # a burn-in configuration sits near the cold end
        t = time.time()
        if T_from is None or T_from == self.T_start:
            initial_seed = prepare(self.T_start, initial_seed)
        else:
            initial_seed = prepare(self.T_start, initial_seed, T_from=T_from)
        wall = time.time() - t
        rec.prepare = {'T': self.T_start, 'T_from': T_from, 'wall_s': float(wall)}
        rec.prepare.update(getattr(self.sampler, 'last_prepare', None) or {})
        what = f'heated from {T_from:.2f} K and equilibrated' if T_from not in (None, self.T_start) else 'equilibrated'
        self.printf(f'trailblaze: {what} the starting configuration at {self.T_start:.2f} K in {wall:.0f} s')
        return initial_seed

    def _finalise(self, rec: TrailblazeRecord, walk: List[Rung]) -> None:
        """Complete the record ascending in temperature with ``seeds/rung_XXX.npz`` matching."""
        if self.direction == 'down':
            rec.rungs = list(reversed(walk))
            K = len(rec.rungs)
            if self.store is not None:
                for k, r in enumerate(rec.rungs):
                    r.seed_path = self.store.link(k, self.walk_store.path(r.walk_index))
            for rej in rec.rejected:
                if 'after_walk_index' not in rej:
                    rej['after_walk_index'] = rej['after_rung']
                    rej['after_rung'] = K - 1 - rej['after_walk_index']
        else:
            rec.rungs = walk
        T = rec.temperatures
        if np.any(np.diff(T) <= 0):
            raise RuntimeError('the finished ladder is not strictly increasing')
        if np.unique(np.round(T, LADDER_DECIMALS)).size != T.size:
            raise RuntimeError('two rungs coincide at two decimals; lower min_dT is not allowed')
        rec.complete = True
        self._persist(rec)
        if self.out_dir:
            np.save(os.path.join(self.out_dir, LADDER_NAME), T)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _make_rung(T, H, meta, acc, acc_pred, dL, step_scale, rejects, wall) -> Rung:
        H = np.asarray(H, dtype=np.float64)
        return Rung(T=float(T), sigma=float(np.std(H, ddof=1)), mean_H=float(H.mean()),
                    n_samples=int(H.size), g=float(meta.get('g', 1.0)), neff=float(meta.get('neff', H.size)),
                    acc_est=acc, acc_pred=acc_pred, dL=dL, step_scale=float(step_scale), rejects=int(rejects),
                    seed_path=None, wall_s=float(wall))

    def _save_seed(self, index: int, seed: Seed, H: np.ndarray) -> Optional[str]:
        return self.walk_store.save(index, seed, H) if self.walk_store else None

    def _reject(self, rec: TrailblazeRecord, walk: List[Rung], T_n: float, acc: float, sigma_n: float,
                why: str) -> None:
        rec.rejected.append({'after_rung': len(walk) - 1, 'T': float(T_n), 'acc_est': float(acc),
                             'sigma': float(sigma_n), 'reason': why})
        self._persist(rec)
        self.printf(f'trailblaze: rejected {T_n:.2f} K (acceptance {acc:.3f}): {why}')

    def _report(self, index: int, r: Rung) -> None:
        acc = '' if r.acc_est is None else f' acc {r.acc_est:.3f} (pred {r.acc_pred:.3f}) dL {r.dL:.3f}'
        rej = f' [{r.rejects} rejected]' if r.rejects else ''
        pooled = '' if r.sigma_step is None or (self.sigma_window == 1 and self.sigma_source == 'pooled') \
            else f' (step {r.sigma_step:.0f})'
        self.printf(f'trailblaze: rung {index:3d}  {r.T:8.3f} K  sigma_H {r.sigma:7.1f}{pooled} kJ/mol  '
                    f'n {r.n_samples} neff {r.neff:.0f}{acc}{rej}  {r.wall_s:.0f} s')


# ---------------------------------------------------------------------------
# OpenMM rung sampler
# ---------------------------------------------------------------------------

class OpenMMRungSampler:
    """Sample one rung of an OpenMM system the way the production sampler does.

    The thermodynamic state is built exactly as ``FultonMarket._build_thermodynamic_states``
    does (NPT at ``pressure_bar``; openmmtools adds a Monte Carlo barostat when
    the system has none), the dynamics are ``LangevinDynamicsMove`` with the
    production timestep and collision rate, and the Context comes from the
    same global cache, so the GPU platform and precision match the run that
    will use the ladder.

    Per rung: ``equil_ps`` of dynamics with velocities redrawn at the new
    temperature, then enthalpy samples every ``sample_interval_ps`` for
    ``sample_ps``, extended in further ``sample_ps`` chunks until the
    stationary part holds ``neff_min`` effective samples or ``max_sample_ps``
    is reached.  ``H = U + pV`` comes from ``reduced_potential`` on the sampler
    state the move just updated, so no extra energy evaluation is needed.

    The first rung is different: its seed comes from the burn-in, which may
    sit at another temperature entirely (the campaign equilibrates at one
    temperature and starts the ladder at another), and the ~1 K steps the
    walk takes later equilibrate in picoseconds while a 10 K jump does not.
    :meth:`prepare` therefore runs ``first_equil_ps`` at the first rung
    before the walk samples it; :class:`Trailblazer` calls it when present.
    A downward walk starts at ``T_max`` from a configuration equilibrated
    near ``T_min``: ``prepare(T, seed, T_from=...)`` then heats it in stages
    of ``ramp_K`` kelvin, ``ramp_ps`` each with velocities redrawn, before
    the ``first_equil_ps`` at ``T``.
    """

    def __init__(self, system, dt_fs: float, pressure_bar: float = 1.0, collision_rate_per_ps: float = 1.0,
                 equil_ps: float = 20.0, sample_ps: float = 100.0, sample_interval_ps: float = 1.0,
                 max_sample_ps: float = 400.0, neff_min: int = 50, first_equil_ps: float = 200.0,
                 context_cache=None, ramp_K: float = 10.0, ramp_ps: Optional[float] = None):
        import openmm.unit as unit
        from openmmtools import cache, mcmc
        from openmmtools.states import ThermodynamicState

        self._unit = unit
        self.dt_fs = float(dt_fs)
        self.pressure_bar = float(pressure_bar)
        self.collision_rate_per_ps = float(collision_rate_per_ps)
        self.equil_ps, self.sample_ps = float(equil_ps), float(sample_ps)
        self.sample_interval_ps, self.max_sample_ps = float(sample_interval_ps), float(max_sample_ps)
        self.neff_min = int(neff_min)
        self.first_equil_ps = float(first_equil_ps)
        self.ramp_K = float(ramp_K)
        self.ramp_ps = float(equil_ps if ramp_ps is None else ramp_ps)
        self.last_prepare: Dict[str, object] = {}

        self._state = ThermodynamicState(system=system, temperature=300.0 * unit.kelvin,
                                         pressure=pressure_bar * unit.bar)
        self._cache = context_cache if context_cache is not None else cache.global_context_cache
        dt = self.dt_fs * unit.femtosecond
        gamma = self.collision_rate_per_ps / unit.picosecond
        self._equil = mcmc.LangevinDynamicsMove(timestep=dt, collision_rate=gamma,
                                                n_steps=max(1, int(round(self.equil_ps * 1000.0 / self.dt_fs))),
                                                reassign_velocities=True)
        self._chunk = mcmc.LangevinDynamicsMove(timestep=dt, collision_rate=gamma,
                                                n_steps=max(1, int(round(self.sample_interval_ps * 1000.0 / self.dt_fs))),
                                                reassign_velocities=False)
        self._first = mcmc.LangevinDynamicsMove(timestep=dt, collision_rate=gamma,
                                                n_steps=max(1, int(round(self.first_equil_ps * 1000.0 / self.dt_fs))),
                                                reassign_velocities=True)
        self._ramp = mcmc.LangevinDynamicsMove(timestep=dt, collision_rate=gamma,
                                               n_steps=max(1, int(round(self.ramp_ps * 1000.0 / self.dt_fs))),
                                               reassign_velocities=True)
        for move in (self._equil, self._chunk, self._first, self._ramp):
            move._global_citation_silence = True   # noqa: SLF001 -- same silencing Randolph uses
        self._n_chunks = max(1, int(round(self.sample_ps / self.sample_interval_ps)))
        self._max_chunks = max(self._n_chunks, int(round(self.max_sample_ps / self.sample_interval_ps)))

    def describe(self) -> Dict[str, object]:
        # the ramp settings are left out on purpose: preparation never recurs on resume, so they
        # must not make a resumed walk refuse its own record
        return dict(kind='openmm', dt_fs=self.dt_fs, pressure_bar=self.pressure_bar,
                    collision_rate_per_ps=self.collision_rate_per_ps, equil_ps=self.equil_ps,
                    sample_ps=self.sample_ps, sample_interval_ps=self.sample_interval_ps,
                    max_sample_ps=self.max_sample_ps, neff_min=self.neff_min,
                    first_equil_ps=self.first_equil_ps)

    def ramp_stages(self, T_from: float, T: float) -> List[float]:
        """Temperatures visited before ``T`` when heating or cooling from ``T_from`` in ``ramp_K`` stages."""
        return ramp_stages(T_from, T, self.ramp_K)

    def prepare(self, T: float, seed: Seed, T_from: Optional[float] = None) -> Seed:
        """Equilibrate the starting configuration at the first rung for ``first_equil_ps``.

        With ``T_from`` given and more than ``ramp_K`` away from ``T``, the
        configuration is first taken through intermediate temperatures in
        ``ramp_K`` steps, ``ramp_ps`` of dynamics each with velocities
        redrawn, so a cold burn-in reaches ``T_max`` without a thermal shock.
        """
        u = self._unit
        t = time.time()
        stages = [] if T_from is None else self.ramp_stages(T_from, T)
        ss = self.sampler_state(seed)
        for T_stage in stages:
            self._state.temperature = float(T_stage) * u.kelvin
            self._ramp.apply(self._state, ss, context_cache=self._cache)
        self._state.temperature = float(T) * u.kelvin
        self._first.apply(self._state, ss, context_cache=self._cache)
        self.last_prepare = {'stages': stages, 'ramp_K': self.ramp_K, 'ramp_ps': self.ramp_ps,
                             'ramp_total_ps': float(len(stages) * self.ramp_ps),
                             'first_equil_ps': self.first_equil_ps, 'prepare_wall_s': float(time.time() - t)}
        return self.seed_from_sampler_state(ss)

    # -- conversions -------------------------------------------------------

    def sampler_state(self, seed: Seed):
        """A ``SamplerState`` from a :class:`Seed` (velocities left out: the first move redraws them)."""
        from openmmtools.states import SamplerState
        u = self._unit
        return SamplerState(positions=np.asarray(seed.positions) * u.nanometer,
                            box_vectors=np.asarray(seed.box_vectors) * u.nanometer)

    @staticmethod
    def seed_from_sampler_state(ss) -> Seed:
        import openmm.unit as unit
        vel = None if ss.velocities is None else np.asarray(ss.velocities.value_in_unit(unit.nanometer / unit.picosecond))
        return Seed(positions=np.asarray(ss.positions.value_in_unit(unit.nanometer)),
                    box_vectors=np.asarray(ss.box_vectors.value_in_unit(unit.nanometer)),
                    velocities=vel)

    # -- sampling ----------------------------------------------------------

    def __call__(self, T: float, seed: Seed) -> Tuple[np.ndarray, Seed, Dict[str, object]]:
        u = self._unit
        state = self._state
        state.temperature = float(T) * u.kelvin
        ss = self.sampler_state(seed)
        kT = KB * float(T)

        self._equil.apply(state, ss, context_cache=self._cache)
        H: List[float] = []
        while True:
            for _ in range(self._n_chunks):
                self._chunk.apply(state, ss, context_cache=self._cache)
                H.append(state.reduced_potential(ss) * kT)
            t0, g, neff = stationary_window(np.asarray(H))
            if neff >= self.neff_min or len(H) >= self._max_chunks:
                break
        H_arr = np.asarray(H, dtype=np.float64)
        meta = {'t0': int(t0), 'g': float(g), 'neff': float(neff), 'n_total': int(H_arr.size),
                'sampled_ps': float(H_arr.size * self.sample_interval_ps)}
        return H_arr[t0:], self.seed_from_sampler_state(ss), meta
