"""Temperature-ladder diagnostics and thermodynamic-length placement from saved blocks.

Every FultonMarket block writes three small arrays that together describe the
ladder completely:

``temperatures.npy``
    ``(K,)`` rung temperatures in kelvin.
``energies.npy``
    ``(n_iter, K, K)`` reduced potentials ``u[t, r, s] = beta_s * H(x_{t,r})``
    for every replica ``r`` evaluated at every rung ``s``.  The parallel
    tempering sampler computes the enthalpy ``H = U + pV`` once per replica and
    rescales it by each rung's inverse temperature, so ``u[t, r, s] / beta_s``
    is the same number for every ``s`` (to float32 precision).
``states.npy``
    ``(n_iter, K)`` the rung occupied by each replica at each iteration.

From these, without any new simulation, this module recovers:

* the per-rung enthalpy fluctuation ``sigma_H(T)`` -- the square root of the
  Fisher information metric for a temperature ladder, ``g(beta) = Var_beta(H)``
  (Shenfeld et al., *Phys. Rev. E* 2009, **80**, 046705, eq. 3);
* the thermodynamic length of every neighbour pair,
  ``dL = sigma_H * (beta_k - beta_{k+1})`` (eq. 11 there, averaging the two
  one-sided estimates), and the cumulative length along the ladder;
* the *exact* nearest-neighbour exchange acceptance the engine would have seen,
  ``min(1, exp(-(beta_k - beta_{k+1}) (H_{k+1} - H_k)))``, which is the
  Metropolis criterion on the configurations that actually sat at the two
  rungs.  The engine prints this to stdout and then loses it.

Equidistant spacing in thermodynamic length maximises the total exchange
acceptance along the ladder (Shenfeld Thm. 2), and for Gaussian energy
fluctuations the neighbour acceptance is ``erfc(dL / 2)``, which maps a target
acceptance onto a target spacing.  :func:`respace` inverts the cumulative
length at equal arc length to produce such a ladder from a measured profile;
:func:`virtual_pair_acceptance` checks a proposed ladder by reweighting the
samples already on disk before any GPU time is spent.

Units: enthalpies in kJ/mol, temperatures in K, ``R_KJ`` in kJ/(mol K).
``C_V`` is reported in units of ``k_B`` (dimensionless), so that
``g(ln beta) = C_V / k_B`` is the number whose square root sets the geometric
ratio of an optimal ladder.

Precision: reduced potentials are stored as float32 at magnitudes of 1e5-1e6
kT, where the float32 resolution is ~0.03 kT.  Everything here casts to
float64 first and recovers ``H`` by averaging ``u / beta_s`` over all ``K``
columns, which divides the rounding noise by ``sqrt(K)``.  Differences such as
``H_{k+1} - H_k`` are then formed in enthalpy, never in reduced potential.

No openmm, openmmtools, pymbar or jax import: numpy and scipy only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import erfc, erfcinv

__all__ = [
    'R_KJ', 'betas',
    'predicted_acceptance', 'length_for_acceptance', 'acceptance_bounds',
    'enthalpy_from_reduced', 'by_state',
    'pair_lengths', 'cumulative_length', 'metric_profile', 'smooth_sigma',
    'pair_acceptance_from_energies', 'pair_acceptance_from_enthalpy',
    'respace', 'virtual_pair_acceptance',
    'LadderData', 'load_run', 'ladder_report', 'format_table', 'block_audit',
    'parse_blocks', 'insertion_candidates', 'INSERTION_RULES',
]

#: Gas constant in kJ/(mol K).  Equals ``openmmtools.constants.kB`` and
#: ``scipy.constants.R / 1000`` to nine significant figures.
R_KJ = 0.008314462618

#: Decimals at which ``FultonMarketAnalysis`` matches ladders across blocks
#: (``analysis.py``: ``np.round(temperatures, decimals=2)``).
LADDER_DECIMALS = 2

ArrayOrList = Union[np.ndarray, Sequence[np.ndarray]]


# ---------------------------------------------------------------------------
# Scalar mappings
# ---------------------------------------------------------------------------

def betas(temperatures: np.ndarray) -> np.ndarray:
    """Inverse temperatures ``1 / (R T)`` in mol/kJ."""
    return 1.0 / (R_KJ * np.asarray(temperatures, dtype=np.float64))


def predicted_acceptance(dL):
    """Gaussian-fluctuation neighbour acceptance ``erfc(dL / 2)``.

    Exact when both rungs have Gaussian enthalpy distributions of equal
    variance ``sigma**2`` and means that differ by ``sigma**2 * dbeta``, which
    is what a constant heat capacity gives.  It sits inside the Shenfeld bounds
    (:func:`acceptance_bounds`) for ``dL`` up to about 1.7 (acceptance above
    ~0.2); beyond that the bounds, which come from a second-order expansion in
    the spacing, fall below the exact Gaussian value.
    """
    return erfc(np.asarray(dL, dtype=np.float64) / 2.0)


def length_for_acceptance(acceptance):
    """Inverse of :func:`predicted_acceptance`: ``2 * erfcinv(A)``."""
    a = np.asarray(acceptance, dtype=np.float64)
    if np.any((a <= 0.0) | (a >= 1.0)):
        raise ValueError('target acceptance must lie strictly between 0 and 1')
    return 2.0 * erfcinv(a)


def acceptance_bounds(dL) -> Tuple[np.ndarray, np.ndarray]:
    """Shenfeld eq. 20 bounds on a neighbour pair's acceptance.

    ``-dL**2/2 - dL/sqrt(2) <= ln A <= -dL**2/2``.  Returns ``(lower, upper)``.
    Asymptotic in the spacing: trustworthy for ``dL`` below ~1.7, which covers
    every ladder this package produces.
    """
    dL = np.asarray(dL, dtype=np.float64)
    return np.exp(-dL ** 2 / 2.0 - dL / np.sqrt(2.0)), np.exp(-dL ** 2 / 2.0)


# ---------------------------------------------------------------------------
# From saved arrays to enthalpies
# ---------------------------------------------------------------------------

def enthalpy_from_reduced(energies: np.ndarray, temperatures: np.ndarray,
                          rtol: float = 1e-5) -> np.ndarray:
    """Recover ``H[t, r]`` in kJ/mol from a block's reduced potentials.

    Averages ``u[t, r, s] / beta_s`` over the ``K`` evaluation rungs.  Raises
    ``ValueError`` when the columns disagree beyond ``rtol``, which means the
    energies did not come from a pure-temperature sampler (a Hamiltonian ladder
    would give genuinely different values per rung).  Rows containing
    non-finite values are returned as NaN rather than raising.
    """
    u = np.asarray(energies, dtype=np.float64)
    if u.ndim != 3:
        raise ValueError('energies must be (n_iter, n_replica, n_state)')
    b = betas(temperatures)
    if u.shape[2] != b.shape[0]:
        raise ValueError(f'energies have {u.shape[2]} state columns but the ladder has {b.shape[0]} rungs')
    H_all = u / b[None, None, :]
    H = H_all.mean(axis=2)

    finite = np.isfinite(H_all).all(axis=(1, 2))
    if finite.any():
        scale = np.maximum(np.abs(H[finite]), 1.0)[:, :, None]
        dev = np.abs(H_all[finite] - H[finite][:, :, None]) / scale
        worst = float(dev.max())
        if worst > rtol:
            raise ValueError(
                f'u[t, r, s] / beta_s varies across s by up to {worst:.2e} (rtol {rtol:.0e}); '
                f'these are not pure-temperature reduced potentials')
    H[~finite] = np.nan
    return H


def by_state(H: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Reorder a per-replica array into per-rung order: ``Hk[t, k] = H[t, r]`` with ``states[t, r] == k``.

    Every row of ``states`` must be a permutation of ``0..K-1`` (it is, for
    replica exchange).  Works on any trailing shape of ``H``.
    """
    H = np.asarray(H)
    s = np.asarray(states)
    if s.ndim != 2 or s.shape != H.shape[:2]:
        raise ValueError('states must be (n_iter, n_replica) matching the first two axes of H')
    n, K = s.shape
    if not np.array_equal(np.sort(s, axis=1), np.broadcast_to(np.arange(K), (n, K))):
        raise ValueError('every row of states must be a permutation of 0..K-1')
    Hk = np.empty_like(H)
    Hk[np.arange(n)[:, None], s] = H
    return Hk


def _block_list(x: ArrayOrList) -> List[np.ndarray]:
    if isinstance(x, np.ndarray):
        return [x]
    return [np.asarray(a) for a in x]


def _finite_rows(Hk: np.ndarray) -> np.ndarray:
    return Hk[np.isfinite(Hk).all(axis=1)]


# ---------------------------------------------------------------------------
# Metric and lengths
# ---------------------------------------------------------------------------

def pair_lengths(temperatures: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Thermodynamic length of each neighbour pair, ``0.5 (sigma_k + sigma_{k+1}) (beta_k - beta_{k+1})``."""
    T = np.asarray(temperatures, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if T.shape != s.shape or T.ndim != 1:
        raise ValueError('temperatures and sigma must be 1-D arrays of the same length')
    b = betas(T)
    return 0.5 * (s[:-1] + s[1:]) * (b[:-1] - b[1:])


def cumulative_length(T_grid: np.ndarray, sigma_grid: np.ndarray) -> np.ndarray:
    """``L(T) = int_{T_0}^{T} sigma_H(T') / (R T'^2) dT'`` by the trapezoid rule, ``L(T_0) = 0``."""
    T = np.asarray(T_grid, dtype=np.float64)
    s = np.asarray(sigma_grid, dtype=np.float64)
    f = s / (R_KJ * T ** 2)
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def metric_profile(temperatures: np.ndarray, Hk_blocks: ArrayOrList,
                   variance: str = 'within') -> Dict[str, np.ndarray]:
    """Per-rung enthalpy statistics and the thermodynamic metric.

    ``variance='within'`` pools the per-block variances (sum of squared
    deviations from each block's own mean over the pooled degrees of freedom),
    which is immune to slow drift of the mean across a long run.
    ``'total'`` is the plain variance of the concatenated samples.  Both are
    returned; ``variance`` picks which one feeds ``sigma_H``.

    Returns ``mean_H``, ``sigma_H`` (kJ/mol), ``sigma_within``, ``sigma_total``,
    ``C_V`` (units of k_B), ``sqrt_g_T`` (1/K; ``dL = sqrt_g_T dT``) and
    ``n_samples`` per rung.
    """
    if variance not in ('within', 'total'):
        raise ValueError("variance must be 'within' or 'total'")
    T = np.asarray(temperatures, dtype=np.float64)
    blocks = [_finite_rows(np.asarray(h, dtype=np.float64)) for h in _block_list(Hk_blocks)]
    blocks = [h for h in blocks if h.shape[0] > 0]
    if not blocks:
        raise ValueError('no finite samples')
    K = T.shape[0]
    if any(h.shape[1] != K for h in blocks):
        raise ValueError('every block must have one column per rung')

    allH = np.concatenate(blocks, axis=0)
    n = allH.shape[0]
    mean_H = allH.mean(axis=0)
    var_total = allH.var(axis=0, ddof=1) if n > 1 else np.full(K, np.nan)

    ss = np.zeros(K)
    dof = 0
    for h in blocks:
        if h.shape[0] > 1:
            ss += ((h - h.mean(axis=0)) ** 2).sum(axis=0)
            dof += h.shape[0] - 1
    var_within = ss / dof if dof > 0 else np.full(K, np.nan)

    var = var_within if variance == 'within' else var_total
    sigma = np.sqrt(var)
    return {
        'temperatures': T,
        'mean_H': mean_H,
        'sigma_H': sigma,
        'sigma_within': np.sqrt(var_within),
        'sigma_total': np.sqrt(var_total),
        'C_V': var / (R_KJ * T) ** 2,
        'sqrt_g_T': sigma / (R_KJ * T ** 2),
        'n_samples': np.full(K, n, dtype=int),
    }


def smooth_sigma(temperatures: np.ndarray, sigma: np.ndarray, n_samples=None,
                 model: str = 'cv_linear') -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit ``C_V(T) = a + b T`` and return ``sigma_fit = R T sqrt(a + b T)``.

    A per-rung ``sigma`` from ``n`` samples carries a relative standard error of
    about ``1 / sqrt(2 (n - 1))``; the fit replaces that with the noise of two
    coefficients.  ``model='none'`` returns ``sigma`` unchanged.  The returned
    info holds the coefficients, the relative residual RMS and the noise level
    the raw estimate was expected to have, so a residual well above that flags
    a profile that is not linear in T.
    """
    T = np.asarray(temperatures, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if model == 'none':
        return s.copy(), {'model': 'none'}
    if model != 'cv_linear':
        raise ValueError("model must be 'cv_linear' or 'none'")
    ok = np.isfinite(s) & (s > 0)
    if ok.sum() < 2:
        return s.copy(), {'model': 'none', 'reason': 'fewer than two finite rungs'}
    cv = (s / (R_KJ * T)) ** 2
    w = None
    if n_samples is not None:
        w = np.sqrt(np.maximum(np.asarray(n_samples, dtype=np.float64), 1.0))
    A = np.stack([np.ones(ok.sum()), T[ok]], axis=1)
    y = cv[ok]
    if w is not None:
        A = A * w[ok][:, None]
        y = y * w[ok]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    cv_fit = a + b * T
    if np.any(cv_fit <= 0):
        return s.copy(), {'model': 'none', 'reason': 'linear C_V fit went non-positive', 'a': a, 'b': b}
    s_fit = R_KJ * T * np.sqrt(cv_fit)
    resid = (s[ok] - s_fit[ok]) / s_fit[ok]
    info = {
        'model': 'cv_linear', 'a': a, 'b': b,
        'residual_rms': float(np.sqrt(np.mean(resid ** 2))),
        'cv_change_fraction': float(cv_fit[-1] / cv_fit[0] - 1.0),
    }
    if n_samples is not None:
        nmean = float(np.mean(n_samples))
        info['expected_noise'] = float(1.0 / np.sqrt(2.0 * max(nmean - 1.0, 1.0)))
    return s_fit, info


# ---------------------------------------------------------------------------
# Measured acceptance
# ---------------------------------------------------------------------------

def _acc_from_delta(delta: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.exp(-delta))


def pair_acceptance_from_enthalpy(temperatures: np.ndarray, Hk_blocks: ArrayOrList,
                                  mode: str = 'simultaneous') -> Dict[str, np.ndarray]:
    """Exact nearest-neighbour exchange acceptance from per-rung enthalpies.

    For the configurations at rungs ``k`` and ``k+1`` the Metropolis
    probability is ``min(1, exp(-(beta_k - beta_{k+1}) (H_{k+1} - H_k)))``.

    ``mode='simultaneous'`` pairs the two configurations present at the same
    iteration -- what the engine actually proposed.  ``mode='pooled'`` pairs
    every iteration ``t`` of rung ``k`` with every iteration ``t'`` of rung
    ``k+1`` *within the same block*; the two rungs' samples are independent
    draws from their ensembles, so this is an unbiased estimate of the same
    probability with ``n**2`` instead of ``n`` terms per block.  Cross-block
    pairs are deliberately excluded so slow drift cannot leak in.

    Returns ``acceptance`` (K-1,) and ``n_samples`` (K-1,) the number of pairs
    averaged.
    """
    if mode not in ('simultaneous', 'pooled'):
        raise ValueError("mode must be 'simultaneous' or 'pooled'")
    b = betas(temperatures)
    db = b[:-1] - b[1:]
    num = None
    cnt = 0.0
    for h in _block_list(Hk_blocks):
        h = _finite_rows(np.asarray(h, dtype=np.float64))
        if h.shape[0] == 0:
            continue
        if h.shape[1] != b.shape[0]:
            raise ValueError('every block must have one column per rung')
        if mode == 'simultaneous':
            delta = db[None, :] * (h[:, 1:] - h[:, :-1])          # (n, K-1)
            acc = _acc_from_delta(delta).sum(axis=0)
            n_pairs = h.shape[0]
        else:
            delta = db[None, None, :] * (h[None, :, 1:] - h[:, None, :-1])   # (n, n, K-1)
            acc = _acc_from_delta(delta).sum(axis=(0, 1))
            n_pairs = h.shape[0] ** 2
        num = acc if num is None else num + acc
        cnt += n_pairs
    if num is None:
        raise ValueError('no finite samples')
    return {'acceptance': num / cnt, 'n_samples': np.full(b.shape[0] - 1, int(cnt))}


def pair_acceptance_from_energies(energies: ArrayOrList, states: ArrayOrList,
                                  skip: int = 1, mode: str = 'simultaneous') -> Dict[str, np.ndarray]:
    """The same acceptance straight from reduced potentials (the engine's own form).

    ``min(1, exp(-(u[a, k+1] + u[b, k] - u[a, k] - u[b, k+1])))`` with ``a``
    the replica at rung ``k`` and ``b`` at ``k+1``.  Needs no temperatures.
    Carries float32 rounding noise of order 0.06 in the exponent at the
    campaign's energy scale, so :func:`pair_acceptance_from_enthalpy` is the
    preferred estimator; this one exists to show the two agree.
    """
    if mode not in ('simultaneous', 'pooled'):
        raise ValueError("mode must be 'simultaneous' or 'pooled'")
    e_blocks = _block_list(energies)
    s_blocks = _block_list(states)
    if len(e_blocks) != len(s_blocks):
        raise ValueError('energies and states must have the same number of blocks')
    num = None
    cnt = 0.0
    K = None
    for u, s in zip(e_blocks, s_blocks):
        u = np.asarray(u, dtype=np.float64)[skip:]
        s = np.asarray(s)[skip:]
        if u.shape[0] == 0:
            continue
        n, K_b = s.shape
        if K is None:
            K = K_b
        elif K_b != K:
            raise ValueError('every block must have the same number of rungs')
        finite = np.isfinite(u).all(axis=(1, 2))
        u, s = u[finite], s[finite]
        n = u.shape[0]
        if n == 0:
            continue
        rep_at = np.argsort(s, axis=1)                      # rep_at[t, k] = replica at rung k
        t_idx = np.arange(n)[:, None]
        k_idx = np.arange(K - 1)[None, :]
        A, B = rep_at[:, :-1], rep_at[:, 1:]
        F = u[t_idx, A, k_idx + 1] - u[t_idx, A, k_idx]    # (n, K-1)
        G = u[t_idx, B, k_idx] - u[t_idx, B, k_idx + 1]
        if mode == 'simultaneous':
            acc = _acc_from_delta(F + G).sum(axis=0)
            n_pairs = n
        else:
            acc = _acc_from_delta(F[:, None, :] + G[None, :, :]).sum(axis=(0, 1))
            n_pairs = n ** 2
        num = acc if num is None else num + acc
        cnt += n_pairs
    if num is None:
        raise ValueError('no finite samples after skip')
    return {'acceptance': num / cnt, 'n_samples': np.full(K - 1, int(cnt))}


# ---------------------------------------------------------------------------
# Adaptive insertion
# ---------------------------------------------------------------------------

INSERTION_RULES = ('mean', 'ucb')


def insertion_candidates(per_iter_acc: np.ndarray, threshold: float, rule: str = 'mean',
                         min_iters: int = 10, z: float = 2.0) -> Dict[str, np.ndarray]:
    """Which neighbour pairs the engine should split, from per-iteration exchange statistics.

    ``per_iter_acc[t, i, j]`` is the acceptance of exchanges between rungs
    ``i`` and ``j`` observed at iteration ``t`` (accepted/proposed; a pair
    with no proposals counts 0).  Each iteration contributes one pair of
    configurations, so ``n`` iterations give an acceptance estimate with a
    standard error of roughly 0.1–0.2 for ``n`` of 5–18, which is the
    campaign's block.

    ``rule='mean'`` is the historical test: insert when the mean falls below
    ``threshold``.  Judged on a single five-iteration cycle it flags about
    one pair in five of a ladder whose true acceptance is 0.10 above the
    threshold, which is how the campaign's 60-rung ladders grew to 110–132.
    ``rule='ucb'`` inserts only when the upper confidence bound
    ``mean + z * SE`` (SE across iterations) is below ``threshold`` and at
    least ``min_iters`` iterations have accumulated: a pair has to be shown
    bad, not merely read low once.

    Returns ``acceptance`` (K-1,), ``upper`` (K-1,; the mean itself under
    ``'mean'``), ``n_iters`` and ``insert_after`` (rung indices ``k`` whose
    pair ``(k, k+1)`` should be split, i.e. the engine inserts at ``k+1``).
    """
    if rule not in INSERTION_RULES:
        raise ValueError(f"rule must be one of {INSERTION_RULES}")
    a = np.asarray(per_iter_acc, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError('per_iter_acc must be (n_iter, K, K)')
    n, K = a.shape[0], a.shape[1]
    idx = np.arange(K - 1)
    series = a[:, idx, idx + 1]                      # (n, K-1)
    mean = series.mean(axis=0) if n else np.full(K - 1, np.nan)
    if rule == 'mean':
        upper = mean.copy()
        flag = mean < threshold
    else:
        if n >= 2:
            se = series.std(axis=0, ddof=1) / np.sqrt(n)
            upper = mean + z * se
        else:
            upper = np.full(K - 1, np.inf)
        flag = (upper < threshold) & (n >= int(min_iters))
    return {'acceptance': mean, 'upper': upper, 'n_iters': int(n),
            'insert_after': np.where(flag)[0]}


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def respace(temperatures: np.ndarray, sigma: np.ndarray, n_rungs: Optional[int] = None,
            target_dL: Optional[float] = None, target_acceptance: Optional[float] = None,
            T_min: Optional[float] = None, T_max: Optional[float] = None,
            n_fine: int = 4000, decimals: int = LADDER_DECIMALS) -> Dict[str, object]:
    """Place rungs at equal thermodynamic length along a measured ``sigma_H(T)`` profile.

    Exactly one of ``n_rungs``, ``target_dL`` or ``target_acceptance`` sets the
    spacing.  ``sigma`` is interpolated linearly in ``T`` onto ``n_fine``
    points between ``T_min`` and ``T_max`` (defaults: the sampled range),
    ``L(T)`` is integrated, and rung ``j`` is placed where
    ``L = j * L_total / (K - 1)``.  The endpoints are forced exactly.

    Outside the sampled range ``sigma`` is held at the nearest sampled value
    and ``extrapolated`` is set.  Raises ``ValueError`` if two rungs would
    coincide after rounding to ``decimals`` (the analysis layer matches rungs
    at two decimals) or if the ladder is not strictly increasing.
    """
    given = [x is not None for x in (n_rungs, target_dL, target_acceptance)]
    if sum(given) != 1:
        raise ValueError('give exactly one of n_rungs, target_dL, target_acceptance')
    T = np.asarray(temperatures, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if T.ndim != 1 or T.shape != s.shape or T.shape[0] < 2:
        raise ValueError('temperatures and sigma must be 1-D of equal length >= 2')
    if np.any(np.diff(T) <= 0):
        raise ValueError('temperatures must be strictly increasing')
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError('sigma must be finite and positive')

    lo = float(T[0]) if T_min is None else float(T_min)
    hi = float(T[-1]) if T_max is None else float(T_max)
    if not hi > lo:
        raise ValueError('T_max must exceed T_min')
    extrapolated = lo < T[0] - 1e-9 or hi > T[-1] + 1e-9

    T_grid = np.linspace(lo, hi, int(n_fine))
    s_grid = np.interp(T_grid, T, s)
    L = cumulative_length(T_grid, s_grid)
    L_total = float(L[-1])

    if target_acceptance is not None:
        target_dL = float(length_for_acceptance(target_acceptance))
    if target_dL is not None:
        if target_dL <= 0:
            raise ValueError('target_dL must be positive')
        K = int(np.ceil(L_total / target_dL)) + 1
    else:
        K = int(n_rungs)
    if K < 2:
        raise ValueError('a ladder needs at least two rungs')
    dL_actual = L_total / (K - 1)

    T_new = np.interp(np.arange(K) * dL_actual, L, T_grid)
    T_new[0], T_new[-1] = lo, hi
    if np.any(np.diff(T_new) <= 0):
        raise ValueError('respaced ladder is not strictly increasing')
    if np.unique(np.round(T_new, decimals)).size != K:
        raise ValueError(
            f'{K} rungs over {hi - lo:.2f} K collide at {decimals} decimals; '
            f'the analysis layer could not tell them apart')

    s_new = np.interp(T_new, T, s)
    dL_pairs = pair_lengths(T_new, s_new)
    return {
        'temperatures': T_new,
        'sigma_H': s_new,
        'n_rungs': K,
        'dL_target': None if target_dL is None else float(target_dL),
        'dL_actual': float(dL_actual),
        'L_total': L_total,
        'pair_dL': dL_pairs,
        'predicted_acceptance': predicted_acceptance(dL_pairs),
        'extrapolated': bool(extrapolated),
        'T_min': lo, 'T_max': hi,
    }


def virtual_pair_acceptance(T_a: float, T_b: float, temperatures: np.ndarray,
                            Hk_blocks: ArrayOrList) -> Dict[str, float]:
    """Acceptance of a hypothetical pair ``(T_a, T_b)`` by reweighting sampled rungs.

    Each hypothetical rung borrows the samples of the nearest sampled rung and
    reweights them by ``exp(-(beta_new - beta_k) (H - <H>_k))``; the pair's
    acceptance is the weighted within-block pooled Metropolis average.  Returns
    the acceptance and the smaller of the two effective sample sizes (as a
    fraction of the raw count), which says how far the reweighting had to
    stretch.
    """
    T = np.asarray(temperatures, dtype=np.float64)
    blocks = [_finite_rows(np.asarray(h, dtype=np.float64)) for h in _block_list(Hk_blocks)]
    blocks = [h for h in blocks if h.shape[0] > 0]
    if not blocks:
        raise ValueError('no finite samples')
    b = betas(T)
    ka = int(np.argmin(np.abs(T - T_a)))
    kb = int(np.argmin(np.abs(T - T_b)))
    beta_a = 1.0 / (R_KJ * T_a)
    beta_b = 1.0 / (R_KJ * T_b)
    allH = np.concatenate(blocks, axis=0)
    mean_a, mean_b = allH[:, ka].mean(), allH[:, kb].mean()

    num = 0.0
    den = 0.0
    wa_all, wb_all = [], []
    for h in blocks:
        la = -(beta_a - b[ka]) * (h[:, ka] - mean_a)
        lb = -(beta_b - b[kb]) * (h[:, kb] - mean_b)
        wa = np.exp(la - la.max())
        wb = np.exp(lb - lb.max())
        delta = (beta_a - beta_b) * (h[None, :, kb] - h[:, None, ka])
        acc = _acc_from_delta(delta)
        num += float(wa @ acc @ wb)
        den += float(wa.sum() * wb.sum())
        wa_all.append(wa / wa.sum() * h.shape[0])
        wb_all.append(wb / wb.sum() * h.shape[0])
    wa_all = np.concatenate(wa_all)
    wb_all = np.concatenate(wb_all)

    def ess_fraction(w):
        return float(w.sum() ** 2 / (w ** 2).sum() / w.size)

    return {
        'acceptance': num / den,
        'ess_fraction': min(ess_fraction(wa_all), ess_fraction(wb_all)),
        'source_rungs': (ka, kb),
    }


# ---------------------------------------------------------------------------
# Loading a run
# ---------------------------------------------------------------------------

@dataclass
class LadderData:
    """The small arrays of a run, converted to per-rung enthalpies.

    ``Hk_blocks_raw`` keeps every iteration; ``Hk_blocks`` applies ``skip``.
    Iteration 0 of every block is the block's starting configurations, which
    the engine seeds identically on all replicas, so ``skip=1`` is the floor
    for any statistic.
    """
    run_dir: str
    temperatures: np.ndarray
    blocks_used: List[int]
    blocks_skipped: Dict[int, str]
    skip: int
    Hk_blocks_raw: List[np.ndarray] = field(default_factory=list)
    n_nonfinite: int = 0

    @property
    def Hk_blocks(self) -> List[np.ndarray]:
        return [h[self.skip:] for h in self.Hk_blocks_raw]

    @property
    def n_iterations(self) -> List[int]:
        return [int(h.shape[0]) for h in self.Hk_blocks_raw]

    @property
    def n_samples_per_rung(self) -> int:
        return int(sum(max(h.shape[0] - self.skip, 0) for h in self.Hk_blocks_raw))


def _block_dirs(run_dir: str) -> List[Tuple[int, str]]:
    base = os.path.join(run_dir, 'saved_variables')
    if not os.path.isdir(base):
        raise FileNotFoundError(f'{base} does not exist')
    names = sorted((d for d in os.listdir(base) if d.isdigit()), key=int)
    return [(int(d), os.path.join(base, d)) for d in names]


def parse_blocks(spec: Optional[Sequence[str]]) -> Optional[List[int]]:
    """Turn CLI tokens like ``['3', '7:12', '20']`` into ``[3, 7, 8, ..., 11, 20]``.

    A ``start:stop`` range is half-open like Python's.
    """
    if spec is None:
        return None
    out: List[int] = []
    for tok in spec:
        tok = str(tok)
        if ':' in tok:
            a, b = tok.split(':', 1)
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(tok))
    return out


def load_block(block_dir: str, temperatures: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """One block's ``(temperatures, Hk_raw, n_nonfinite)`` from its three small arrays."""
    T = np.load(os.path.join(block_dir, 'temperatures.npy'))
    u = np.load(os.path.join(block_dir, 'energies.npy'), mmap_mode='r')
    s = np.load(os.path.join(block_dir, 'states.npy'), mmap_mode='r')
    H = enthalpy_from_reduced(np.asarray(u), np.asarray(T))
    Hk = by_state(H, np.asarray(s))
    n_bad = int((~np.isfinite(Hk).all(axis=1)).sum())
    return np.asarray(T, dtype=np.float64), Hk, n_bad


def load_run(run_dir: str, blocks: Optional[Sequence[int]] = None, skip: int = 1) -> LadderData:
    """Read every (or the selected) block of a run into :class:`LadderData`.

    Blocks whose ladder differs from the final block's at two decimals are
    skipped with a reason, as ``mixing.concatenated_states`` does: the ladder
    grows by insertion and a padded column would fabricate samples.
    """
    dirs = _block_dirs(run_dir)
    if not dirs:
        raise FileNotFoundError(f'no numbered block directories under {run_dir}/saved_variables')
    if blocks is not None:
        keep = set(int(b) for b in blocks)
        dirs = [(b, d) for b, d in dirs if b in keep]
        if not dirs:
            raise ValueError('none of the requested blocks exist')
    final_T = np.load(os.path.join(dirs[-1][1], 'temperatures.npy')).astype(np.float64)
    final_key = np.round(final_T, LADDER_DECIMALS)

    used, skipped, Hk_blocks, n_bad = [], {}, [], 0
    for b, d in dirs:
        needed = [os.path.join(d, f) for f in ('temperatures.npy', 'energies.npy', 'states.npy')]
        if not all(os.path.exists(p) for p in needed):
            skipped[b] = 'missing arrays'
            continue
        T = np.load(needed[0]).astype(np.float64)
        if T.shape != final_T.shape or not np.array_equal(np.round(T, LADDER_DECIMALS), final_key):
            skipped[b] = f'ladder differs from final ({T.shape[0]} vs {final_T.shape[0]} rungs)'
            continue
        _, Hk, bad = load_block(d)
        if Hk.shape[0] > bad and (Hk.shape[0] - bad) * 2 < Hk.shape[0]:
            skipped[b] = f'{bad} of {Hk.shape[0]} iterations non-finite'
            continue
        used.append(b)
        Hk_blocks.append(Hk)
        n_bad += bad
    if not used:
        raise ValueError(f'no usable block under {run_dir}: {skipped}')
    return LadderData(run_dir=run_dir, temperatures=final_T, blocks_used=used,
                      blocks_skipped=skipped, skip=int(skip), Hk_blocks_raw=Hk_blocks,
                      n_nonfinite=n_bad)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _iteration_profile(Hk_blocks_raw: List[np.ndarray], mean_H: np.ndarray, sigma: np.ndarray) -> List[float]:
    """Mean standardised deviation of H per iteration index, over blocks and rungs."""
    n_max = max(h.shape[0] for h in Hk_blocks_raw)
    out = []
    for t in range(n_max):
        vals = [((h[t] - mean_H) / sigma) for h in Hk_blocks_raw if h.shape[0] > t]
        vals = np.concatenate(vals)
        vals = vals[np.isfinite(vals)]
        out.append(float(vals.mean()) if vals.size else float('nan'))
    return out


def _calibration_factor(dL: np.ndarray, A_meas: np.ndarray) -> float:
    """Least-squares ``c`` in ``A_meas = erfc(c dL / 2)``."""
    ok = np.isfinite(dL) & np.isfinite(A_meas)
    if ok.sum() < 2:
        return float('nan')
    grid = np.linspace(0.5, 1.5, 2001)
    sse = [((erfc(c * dL[ok] / 2.0) - A_meas[ok]) ** 2).sum() for c in grid]
    return float(grid[int(np.argmin(sse))])


def _f(x) -> float:
    return float(x)


def ladder_report(data: Union[LadderData, str], targets: Sequence[float] = (0.35, 0.40, 0.50),
                  skip: int = 1, smooth: str = 'cv_linear', blocks: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Everything the audit knows about one run, as a JSON-serialisable dict.

    ``rungs``: per rung ``T``, ``mean_H``, ``sigma_H``, ``sigma_fit``, ``C_V``.
    ``pairs``: per neighbour pair the spacing ``dT``, the length ``dL`` (from
    the fitted sigma) and ``dL_raw``, the Gaussian prediction, the Shenfeld
    bounds, and the measured acceptance in both modes.
    ``summary``: ladder-level numbers -- total length, spacing uniformity,
    acceptance range, how well the prediction tracks the measurement, the rung
    count an equidistant ladder would need at each target, the C_V fit, and
    the restart-transient diagnostics.
    """
    if isinstance(data, str):
        data = load_run(data, blocks=blocks, skip=skip)
    T = data.temperatures
    K = T.shape[0]
    Hk = data.Hk_blocks

    prof = metric_profile(T, Hk, variance='within')
    sigma_fit, fit = smooth_sigma(T, prof['sigma_H'], n_samples=prof['n_samples'], model=smooth)
    dL = pair_lengths(T, sigma_fit)
    dL_raw = pair_lengths(T, prof['sigma_H'])
    A_pred = predicted_acceptance(dL)
    lower, upper = acceptance_bounds(dL)
    sim = pair_acceptance_from_enthalpy(T, Hk, mode='simultaneous')
    pool = pair_acceptance_from_enthalpy(T, Hk, mode='pooled')
    A_sim, A_pool = sim['acceptance'], pool['acceptance']

    n_pairs = int(sim['n_samples'][0])
    se_sim = np.sqrt(np.maximum(A_sim * (1 - A_sim), 1e-12) / max(n_pairs, 1))
    violations = int(((A_pool < lower - 2 * se_sim) | (A_pool > upper + 2 * se_sim)).sum())

    T_grid = np.linspace(T[0], T[-1], 4000)
    L_total = float(cumulative_length(T_grid, np.interp(T_grid, T, sigma_fit))[-1])
    K_needed = {f'{a:.2f}': int(np.ceil(L_total / float(length_for_acceptance(a))) + 1) for a in targets}

    iter_profile = _iteration_profile(data.Hk_blocks_raw, prof['mean_H'], prof['sigma_H'])
    ratio = prof['sigma_total'] / prof['sigma_within']

    rungs = [{
        'rung': k, 'T': _f(T[k]), 'mean_H': _f(prof['mean_H'][k]),
        'sigma_H': _f(prof['sigma_H'][k]), 'sigma_fit': _f(sigma_fit[k]),
        'C_V': _f(prof['C_V'][k]), 'n_samples': int(prof['n_samples'][k]),
    } for k in range(K)]
    pairs = [{
        'pair': k, 'T_lo': _f(T[k]), 'T_hi': _f(T[k + 1]), 'dT': _f(T[k + 1] - T[k]),
        'dL': _f(dL[k]), 'dL_raw': _f(dL_raw[k]),
        'A_pred': _f(A_pred[k]), 'A_lower': _f(lower[k]), 'A_upper': _f(upper[k]),
        'A_meas_sim': _f(A_sim[k]), 'A_meas_pool': _f(A_pool[k]), 'n_pairs': n_pairs,
    } for k in range(K - 1)]

    finite = np.isfinite(A_pool) & np.isfinite(A_pred)
    corr = float(np.corrcoef(A_pred[finite], A_pool[finite])[0, 1]) if finite.sum() > 2 else float('nan')
    summary = {
        'run_dir': data.run_dir,
        'n_rungs': K,
        'T_min': _f(T[0]), 'T_max': _f(T[-1]),
        'blocks_used': list(data.blocks_used),
        'blocks_skipped': {int(k): v for k, v in data.blocks_skipped.items()},
        'n_iterations_per_block': data.n_iterations,
        'skip': data.skip,
        'samples_per_rung': data.n_samples_per_rung,
        'n_nonfinite_iterations': data.n_nonfinite,
        'L_total': L_total,
        'dL_mean': _f(np.mean(dL)), 'dL_min': _f(np.min(dL)), 'dL_max': _f(np.max(dL)),
        'dL_cv': _f(np.std(dL) / np.mean(dL)),
        'A_meas_min': _f(np.nanmin(A_pool)), 'A_meas_mean': _f(np.nanmean(A_pool)),
        'A_meas_max': _f(np.nanmax(A_pool)),
        'A_pred_min': _f(np.min(A_pred)), 'A_pred_mean': _f(np.mean(A_pred)),
        'A_pred_max': _f(np.max(A_pred)),
        'corr_pred_meas': corr,
        'mean_abs_dev': _f(np.nanmean(np.abs(A_pred - A_pool))),
        'calibration_c': _calibration_factor(dL, A_pool),
        'bounds_violations': violations,
        'K_needed': K_needed,
        'targets': [float(a) for a in targets],
        'cv_fit': fit,
        'sigma_H_bottom': _f(prof['sigma_H'][0]), 'sigma_H_top': _f(prof['sigma_H'][-1]),
        'C_V_bottom': _f(prof['C_V'][0]), 'C_V_top': _f(prof['C_V'][-1]),
        'sigma_total_over_within_median': _f(np.nanmedian(ratio)),
        'sigma_total_over_within_max': _f(np.nanmax(ratio)),
        'iteration_profile_sigma': iter_profile,
    }
    return {'summary': summary, 'rungs': rungs, 'pairs': pairs}


def format_table(report: Dict[str, object], max_rows: Optional[int] = None) -> str:
    """Console rendering of a report: the pair table and the summary."""
    s = report['summary']
    lines = [
        f"Ladder audit: {s['run_dir']}",
        f"  rungs {s['n_rungs']}  {s['T_min']:.2f}-{s['T_max']:.2f} K   blocks used "
        f"{len(s['blocks_used'])} (skipped {len(s['blocks_skipped'])})   samples/rung {s['samples_per_rung']}   skip {s['skip']}",
        f"  sigma_H {s['sigma_H_bottom']:.0f} -> {s['sigma_H_top']:.0f} kJ/mol   C_V/k_B {s['C_V_bottom']:.3g} -> {s['C_V_top']:.3g}"
        f"   total thermodynamic length {s['L_total']:.2f}",
        f"  dL per pair min/mean/max {s['dL_min']:.2f}/{s['dL_mean']:.2f}/{s['dL_max']:.2f}  (CV {s['dL_cv']:.2f})",
        f"  measured A min/mean/max {s['A_meas_min']:.3f}/{s['A_meas_mean']:.3f}/{s['A_meas_max']:.3f}"
        f"   predicted {s['A_pred_min']:.3f}/{s['A_pred_mean']:.3f}/{s['A_pred_max']:.3f}"
        f"   corr {s['corr_pred_meas']:.3f}  mean|dev| {s['mean_abs_dev']:.3f}  c {s['calibration_c']:.3f}"
        f"   bounds violations {s['bounds_violations']}",
        "  rungs needed at target acceptance: " + '  '.join(f"{a}: {k}" for a, k in s['K_needed'].items()),
        f"  restart transient (sigma units, by iteration): "
        + ' '.join(f"{v:+.2f}" for v in s['iteration_profile_sigma'][:10]),
        '',
        f"{'pair':>4} {'T_lo':>8} {'T_hi':>8} {'dT':>6} {'dL':>6} {'A_pred':>7} {'A_sim':>7} {'A_pool':>7}",
    ]
    rows = report['pairs'] if max_rows is None else report['pairs'][:max_rows]
    for p in rows:
        lines.append(f"{p['pair']:>4} {p['T_lo']:>8.2f} {p['T_hi']:>8.2f} {p['dT']:>6.2f} {p['dL']:>6.3f} "
                     f"{p['A_pred']:>7.3f} {p['A_meas_sim']:>7.3f} {p['A_meas_pool']:>7.3f}")
    if max_rows is not None and len(report['pairs']) > max_rows:
        lines.append(f"  ... {len(report['pairs']) - max_rows} more pairs")
    return '\n'.join(lines)


def block_audit(run_dir: str, block: int, trailing: int = 5, skip: int = 1,
                threshold: Optional[float] = None) -> Dict[str, object]:
    """Audit one freshly saved block, for the in-run hook.

    Reads only that block, plus the JSON records of up to ``trailing - 1``
    previous blocks under ``<run_dir>/analysis/ladder/`` to form a trailing
    pooled acceptance with less noise than a single block (+-0.07 per pair per
    block, +-0.03 over five).  Returns the dict to be written as
    ``<run_dir>/analysis/ladder/<block:04d>.json``.
    """
    bdir = os.path.join(run_dir, 'saved_variables', str(block))
    T, Hk_raw, n_bad = load_block(bdir)
    Hk = Hk_raw[skip:]
    prof = metric_profile(T, [Hk], variance='within')
    dL = pair_lengths(T, prof['sigma_H'])
    sim = pair_acceptance_from_enthalpy(T, [Hk], mode='simultaneous')['acceptance']
    pool = pair_acceptance_from_enthalpy(T, [Hk], mode='pooled')['acceptance']

    key = np.round(T, LADDER_DECIMALS).tolist()
    hist_dir = os.path.join(run_dir, 'analysis', 'ladder')
    trail_num = pool * Hk.shape[0] ** 2
    trail_den = float(Hk.shape[0] ** 2)
    trail_blocks = [int(block)]
    for b in range(int(block) - 1, max(int(block) - int(trailing), -1), -1):
        path = os.path.join(hist_dir, f'{b:04d}.json')
        if not os.path.exists(path):
            break
        with open(path) as fh:
            prev = json.load(fh)
        if prev.get('ladder_key') != key:
            break
        n2 = float(prev['n_iterations_used']) ** 2
        trail_num = trail_num + np.asarray(prev['A_pool'], dtype=np.float64) * n2
        trail_den += n2
        trail_blocks.append(b)
    trailing_A = trail_num / trail_den

    out = {
        'block': int(block),
        'n_iterations': int(Hk_raw.shape[0]),
        'n_iterations_used': int(Hk.shape[0]),
        'skip': int(skip),
        'n_rungs': int(T.shape[0]),
        'ladder_key': key,
        'temperatures': T.tolist(),
        'sigma_H': prof['sigma_H'].tolist(),
        'mean_H': prof['mean_H'].tolist(),
        'dL': dL.tolist(),
        'A_pred': predicted_acceptance(dL).tolist(),
        'A_sim': sim.tolist(),
        'A_pool': pool.tolist(),
        'trailing_blocks': trail_blocks,
        'A_trailing': trailing_A.tolist(),
        'A_min': _f(np.nanmin(pool)), 'A_mean': _f(np.nanmean(pool)), 'A_max': _f(np.nanmax(pool)),
        'iteration_profile_sigma': _iteration_profile([Hk_raw], prof['mean_H'], prof['sigma_H']),
        'n_nonfinite_iterations': int(n_bad),
    }
    if threshold is not None:
        out['threshold'] = float(threshold)
        out['pairs_below_threshold'] = [int(k) for k in np.where(trailing_A < threshold)[0]]
    return out
