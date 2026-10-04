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
    'SAMPLER_KEYS', 'WALK_KEYS', 'split_kwargs',
]

#: Boltzmann constant in kJ/(mol K); equals ``openmmtools.constants.kB``.
KB = R_KJ

RECORD_NAME = 'trailblaze.json'
LADDER_NAME = 'temperatures.npy'
SEED_DIR = 'seeds'

#: ``dL*`` for a target acceptance: ``2 erfcinv(A*)``.
target_spacing = length_for_acceptance

LADDER_MODES = ('geometric', 'trailblaze', 'file')

#: Keyword arguments of :class:`OpenMMRungSampler` and :class:`Trailblazer` that a
#: caller may pass together as one ``trailblaze_kwargs`` dict.
SAMPLER_KEYS = ('equil_ps', 'first_equil_ps', 'sample_ps', 'sample_interval_ps', 'max_sample_ps',
                'neff_min', 'collision_rate_per_ps', 'pressure_bar')
WALK_KEYS = ('target_acc', 'min_acc', 'max_acc', 'shrink', 'max_retries', 'max_rungs', 'max_dT',
             'min_dT', 'force')


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
    """Thermodynamic length between two rungs from their enthalpy fluctuations (Shenfeld eq. 11)."""
    return float(pair_lengths(np.array([T_o, T_n]), np.array([sigma_o, sigma_n]))[0])


def sigma_from_cv(T, cv):
    """``sigma_H`` for a heat capacity ``cv`` in units of k_B: ``k_B T sqrt(cv)``."""
    return KB * np.asarray(T, dtype=np.float64) * np.sqrt(cv)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------

def next_temperature(T_o: float, sigma_o: float, dL: float, T_max: float, step_scale: float = 1.0,
                     max_dT: float = 10.0, min_dT: float = 0.1,
                     even_out_fraction: float = 0.5) -> Tuple[float, bool]:
    """The next rung a thermodynamic length ``dL * step_scale`` above ``T_o``.

    Exact in ``beta``: ``beta_n = beta_o - dL step_scale / sigma_o``.  The
    step is capped at ``max_dT`` (a huge ``sigma`` estimate from a transient
    must not launch the walk) and refused below ``min_dT`` (a tiny step means
    ``sigma`` is not a stationary fluctuation, or the ladder would collide at
    the two decimals the analysis layer uses).

    Endpoint handling: when the step reaches ``T_max`` the rung is placed
    there and ``crossed`` is True.  When it would leave a sliver shorter than
    ``even_out_fraction`` of a step, the remaining distance is split in two
    equal rungs instead, so the last pair is never longer than a step and
    never a stub.
    """
    if not np.isfinite(sigma_o) or sigma_o <= 0:
        raise ValueError(f'sigma_H must be finite and positive, got {sigma_o}')
    if dL <= 0 or step_scale <= 0:
        raise ValueError('dL and step_scale must be positive')
    if T_o >= T_max:
        raise ValueError(f'T_o={T_o} is not below T_max={T_max}')
    denom = 1.0 - dL * step_scale * KB * T_o / sigma_o
    T_n = T_o + max_dT if denom <= 0 else T_o / denom
    dT = min(T_n - T_o, max_dT)
    if dT < min_dT:
        raise ValueError(f'step {dT:.4f} K below min_dT={min_dT} K: sigma_H={sigma_o:.1f} at {T_o:.2f} K '
                         f'is too large for dL={dL * step_scale:.3f}; the sample is not stationary')
    T_n = T_o + dT
    if T_n >= T_max:
        return float(T_max), True
    if (T_max - T_n) < even_out_fraction * dT:
        return float(T_o + 0.5 * (T_max - T_o)), False
    return float(T_n), False


def pair_acceptance(H_o: np.ndarray, H_n: np.ndarray, T_o: float, T_n: float,
                    max_pairs: int = 250_000, rng: Optional[np.random.Generator] = None) -> float:
    """Exact Metropolis exchange acceptance between two rungs from their samples.

    Every configuration sampled at ``T_o`` is paired with every configuration
    sampled at ``T_n``: ``mean(min(1, exp(-(beta_o - beta_n) (H_n[j] - H_o[i]))))``,
    which is AlGDock's ``exp(-u01 - u10 + u00 + u11)`` over the full outer
    product.  Above ``max_pairs`` the pairs are subsampled at random.
    """
    H_o = np.asarray(H_o, dtype=np.float64)
    H_n = np.asarray(H_n, dtype=np.float64)
    H_o = H_o[np.isfinite(H_o)]
    H_n = H_n[np.isfinite(H_n)]
    if H_o.size == 0 or H_n.size == 0:
        raise ValueError('both sample sets must contain finite values')
    db = float(beta(T_o) - beta(T_n))
    total = H_o.size * H_n.size
    if total <= max_pairs:
        delta = db * (H_n[None, :] - H_o[:, None])
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        i = rng.integers(0, H_o.size, size=max_pairs)
        j = rng.integers(0, H_n.size, size=max_pairs)
        delta = db * (H_n[j] - H_o[i])
    return float(np.minimum(1.0, np.exp(-delta)).mean())


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
                      max_rungs: int = 10_000, **step_kwargs) -> np.ndarray:
    """Deterministic walk on a known ``sigma_H(T)``: the ladder trailblazing converges to."""
    T = [float(T_min)]
    crossed = False
    while not crossed:
        if len(T) >= max_rungs:
            raise RuntimeError(f'more than {max_rungs} rungs; dL={dL} is too small for this sigma')
        T_n, crossed = next_temperature(T[-1], float(sigma_fn(T[-1])), dL, T_max, **step_kwargs)
        T.append(T_n)
    return np.asarray(T)


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
    T: float
    sigma: float
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


@dataclass
class TrailblazeRecord:
    inputs: Dict[str, object]
    rungs: List[Rung] = field(default_factory=list)
    rejected: List[Dict[str, object]] = field(default_factory=list)
    complete: bool = False
    version: int = 1

    @property
    def temperatures(self) -> np.ndarray:
        return np.array([r.T for r in self.rungs], dtype=np.float64)

    @property
    def meta(self) -> Dict[str, object]:
        dLs = [r.dL for r in self.rungs if r.dL is not None]
        accs = [r.acc_est for r in self.rungs if r.acc_est is not None]
        return {
            'ladder': 'trailblaze',
            'target_acc': self.inputs.get('target_acc'),
            'min_acc': self.inputs.get('min_acc'),
            'max_acc': self.inputs.get('max_acc'),
            'n_rungs': len(self.rungs),
            'L_total': float(np.sum(dLs)) if dLs else 0.0,
            'acc_min': float(np.min(accs)) if accs else None,
            'acc_median': float(np.median(accs)) if accs else None,
            'complete': self.complete,
        }

    def to_dict(self) -> Dict[str, object]:
        return {'version': self.version, 'complete': self.complete, 'inputs': self.inputs,
                'rungs': [asdict(r) for r in self.rungs], 'rejected': self.rejected}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'TrailblazeRecord':
        return cls(inputs=dict(d['inputs']), rungs=[Rung(**r) for r in d['rungs']],
                   rejected=list(d.get('rejected', [])), complete=bool(d['complete']),
                   version=int(d.get('version', 1)))

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


class SeedStore:
    """Per-rung seeds and samples under ``<out_dir>/seeds/rung_XXX.npz``."""

    def __init__(self, out_dir: str):
        self.dir = os.path.join(out_dir, SEED_DIR)

    def path(self, index: int) -> str:
        return os.path.join(self.dir, f'rung_{index:03d}.npz')

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
    out_dir
        Where ``trailblaze.json``, ``temperatures.npy`` and ``seeds/`` go.
        Required for resume; None keeps everything in memory.
    """

    def __init__(self, sampler: RungSampler, T_min: float, T_max: float, target_acc: float = 0.40,
                 min_acc: float = 0.30, max_acc: float = 0.75, shrink: float = 1.25,
                 max_retries: int = 6, max_rungs: int = 400, max_dT: float = 10.0,
                 min_dT: float = 0.1, out_dir: Optional[str] = None, printf: Callable = print,
                 force: bool = False):
        if not T_max > T_min > 0:
            raise ValueError('need 0 < T_min < T_max')
        if not 0 < min_acc <= target_acc <= max_acc < 1:
            raise ValueError('need 0 < min_acc <= target_acc <= max_acc < 1')
        self.sampler = sampler
        self.T_min, self.T_max = float(T_min), float(T_max)
        self.target_acc, self.min_acc, self.max_acc = float(target_acc), float(min_acc), float(max_acc)
        self.shrink, self.max_retries = float(shrink), int(max_retries)
        self.max_rungs, self.max_dT, self.min_dT = int(max_rungs), float(max_dT), float(min_dT)
        self.out_dir = out_dir
        self.printf = printf
        self.force = force
        self.dL = float(target_spacing(self.target_acc))
        self.store = SeedStore(out_dir) if out_dir else None

    # -- bookkeeping -------------------------------------------------------

    def inputs(self) -> Dict[str, object]:
        d = dict(T_min=self.T_min, T_max=self.T_max, target_acc=self.target_acc, min_acc=self.min_acc,
                 max_acc=self.max_acc, shrink=self.shrink, max_retries=self.max_retries,
                 max_rungs=self.max_rungs, max_dT=self.max_dT, min_dT=self.min_dT, dL=self.dL)
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
        if rec.inputs != fresh.inputs and not self.force:
            raise RuntimeError(
                f'{self.record_path} was written with different inputs; pass force=True to '
                f'resume anyway, or remove it to restart.\n  on disk: {rec.inputs}\n  now:     {fresh.inputs}')
        return rec, True

    # -- the walk ----------------------------------------------------------

    def run(self, initial_seed: Seed) -> TrailblazeRecord:
        rec, resumed = self._open_record()
        if rec.complete:
            self.printf(f'trailblaze: complete ladder of {len(rec.rungs)} rungs found in {self.out_dir}')
            return rec

        if rec.rungs:
            if self.store is None or not os.path.exists(self.store.path(len(rec.rungs) - 1)):
                raise RuntimeError('cannot resume: the last accepted rung has no saved seed')
            seed, H_prev = self.store.load(len(rec.rungs) - 1)
            self.printf(f'trailblaze: resuming after rung {len(rec.rungs) - 1} at {rec.rungs[-1].T:.2f} K')
        else:
            self.printf(f'trailblaze: {self.T_min:.2f} -> {self.T_max:.2f} K, target acceptance '
                        f'{self.target_acc:.2f} (dL* = {self.dL:.3f}), reject below {self.min_acc:.2f}')
            t = time.time()
            prepare = getattr(self.sampler, 'prepare', None)
            if callable(prepare):
                initial_seed = prepare(self.T_min, initial_seed)
                self.printf(f'trailblaze: equilibrated the starting configuration at {self.T_min:.2f} K '
                            f'in {time.time() - t:.0f} s')
            H_prev, seed, meta = self.sampler(self.T_min, initial_seed)
            rung = self._make_rung(self.T_min, H_prev, meta, acc=None, acc_pred=None, dL=None,
                                   step_scale=1.0, rejects=0, wall=time.time() - t)
            rung.seed_path = self._save_seed(0, seed, H_prev)
            rec.rungs.append(rung)
            self._persist(rec)
            self._report(0, rung)

        stage = 0
        crossed = rec.rungs[-1].T >= self.T_max
        while not crossed:
            if len(rec.rungs) >= self.max_rungs:
                raise RuntimeError(f'trailblaze: {self.max_rungs} rungs without reaching T_max; '
                                   f'raise max_rungs or target_acc')
            prev = rec.rungs[-1]
            step_scale = self.shrink ** (-stage)
            T_n, crossed = next_temperature(prev.T, prev.sigma, self.dL, self.T_max, step_scale=step_scale,
                                            max_dT=self.max_dT, min_dT=self.min_dT)
            t = time.time()
            H_n, seed_n, meta = self.sampler(T_n, seed)
            wall = time.time() - t
            sigma_n = float(np.std(H_n, ddof=1))
            acc = pair_acceptance(H_prev, H_n, prev.T, T_n)
            dL_real = pair_spacing(prev.T, T_n, prev.sigma, sigma_n)
            acc_pred = float(predicted_acceptance(dL_real))
            n_rejects = abs(stage)

            if acc < self.min_acc and n_rejects < self.max_retries:
                stage += 1
                self._reject(rec, T_n, acc, sigma_n, 'below min_acc, shrinking step')
                crossed = False
                continue
            if acc > self.max_acc and not crossed and n_rejects < self.max_retries:
                stage -= 1
                self._reject(rec, T_n, acc, sigma_n, 'above max_acc, stretching step')
                continue
            if n_rejects >= self.max_retries:
                self.printf(f'trailblaze: WARNING accepting {T_n:.2f} K with acceptance {acc:.3f} after '
                            f'{n_rejects} rejections')

            rung = self._make_rung(T_n, H_n, meta, acc=acc, acc_pred=acc_pred, dL=dL_real,
                                   step_scale=step_scale, rejects=n_rejects, wall=wall)
            index = len(rec.rungs)
            rung.seed_path = self._save_seed(index, seed_n, H_n)
            rec.rungs.append(rung)
            self._persist(rec)
            self._report(index, rung)
            H_prev, seed, stage = H_n, seed_n, 0

        T = rec.temperatures
        if np.unique(np.round(T, LADDER_DECIMALS)).size != T.size:
            raise RuntimeError('two rungs coincide at two decimals; lower min_dT is not allowed')
        rec.complete = True
        self._persist(rec)
        if self.out_dir:
            np.save(os.path.join(self.out_dir, LADDER_NAME), T)
        m = rec.meta
        self.printf(f"trailblaze: done, {m['n_rungs']} rungs, total thermodynamic length {m['L_total']:.2f}, "
                    f"neighbour acceptance min {m['acc_min']:.3f} median {m['acc_median']:.3f}")
        return rec

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _make_rung(T, H, meta, acc, acc_pred, dL, step_scale, rejects, wall) -> Rung:
        H = np.asarray(H, dtype=np.float64)
        return Rung(T=float(T), sigma=float(np.std(H, ddof=1)), mean_H=float(H.mean()),
                    n_samples=int(H.size), g=float(meta.get('g', 1.0)), neff=float(meta.get('neff', H.size)),
                    acc_est=acc, acc_pred=acc_pred, dL=dL, step_scale=float(step_scale), rejects=int(rejects),
                    seed_path=None, wall_s=float(wall))

    def _save_seed(self, index: int, seed: Seed, H: np.ndarray) -> Optional[str]:
        return self.store.save(index, seed, H) if self.store else None

    def _reject(self, rec: TrailblazeRecord, T_n: float, acc: float, sigma_n: float, why: str) -> None:
        rec.rejected.append({'after_rung': len(rec.rungs) - 1, 'T': float(T_n), 'acc_est': float(acc),
                             'sigma': float(sigma_n), 'reason': why})
        self._persist(rec)
        self.printf(f'trailblaze: rejected {T_n:.2f} K (acceptance {acc:.3f}): {why}')

    def _report(self, index: int, r: Rung) -> None:
        acc = '' if r.acc_est is None else f' acc {r.acc_est:.3f} (pred {r.acc_pred:.3f}) dL {r.dL:.3f}'
        rej = f' [{r.rejects} rejected]' if r.rejects else ''
        self.printf(f'trailblaze: rung {index:3d}  {r.T:8.3f} K  sigma_H {r.sigma:7.1f} kJ/mol  '
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
    :meth:`prepare` therefore runs ``first_equil_ps`` at ``T_min`` before the
    walk samples rung 0; :class:`Trailblazer` calls it when present.
    """

    def __init__(self, system, dt_fs: float, pressure_bar: float = 1.0, collision_rate_per_ps: float = 1.0,
                 equil_ps: float = 20.0, sample_ps: float = 100.0, sample_interval_ps: float = 1.0,
                 max_sample_ps: float = 400.0, neff_min: int = 50, first_equil_ps: float = 200.0,
                 context_cache=None):
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
        for move in (self._equil, self._chunk, self._first):
            move._global_citation_silence = True   # noqa: SLF001 -- same silencing Randolph uses
        self._n_chunks = max(1, int(round(self.sample_ps / self.sample_interval_ps)))
        self._max_chunks = max(self._n_chunks, int(round(self.max_sample_ps / self.sample_interval_ps)))

    def describe(self) -> Dict[str, object]:
        return dict(kind='openmm', dt_fs=self.dt_fs, pressure_bar=self.pressure_bar,
                    collision_rate_per_ps=self.collision_rate_per_ps, equil_ps=self.equil_ps,
                    sample_ps=self.sample_ps, sample_interval_ps=self.sample_interval_ps,
                    max_sample_ps=self.max_sample_ps, neff_min=self.neff_min,
                    first_equil_ps=self.first_equil_ps)

    def prepare(self, T: float, seed: Seed) -> Seed:
        """Equilibrate the burn-in configuration at the first rung for ``first_equil_ps``."""
        u = self._unit
        self._state.temperature = float(T) * u.kelvin
        ss = self.sampler_state(seed)
        self._first.apply(self._state, ss, context_cache=self._cache)
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
