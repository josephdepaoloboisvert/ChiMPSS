"""Replica-exchange mixing diagnostics, recovered from ``states.npy``.

Every block writes ``states.npy`` of shape ``(n_iterations, n_replicas)``
holding the thermodynamic state index occupied by each replica at each
iteration.  That is a complete record of how configurations moved through the
temperature ladder, and it is cheap to read -- a few kB per block against the
1.9 GB of ``positions.npy`` -- yet nothing in the package currently looks at
it.  These diagnostics cost seconds for a whole run.

Treating replica exchange as Gibbs sampling over the joint (configuration,
state) space (Chodera & Shirts, *J. Chem. Phys.* 2011, **135**, 194110) makes
the empirical state-transition matrix the natural object: its subdominant
eigenvalue gives the mixing time in temperature space, and round-trip counts
give the rate at which configurations traverse the full ladder (Rosta & Hummer,
*J. Chem. Phys.* 2009, **131**, 165102; Nymeyer, *JCTC* 2008, **4**, 626).

What these measurements established on NTSR1_ML-301.hmr.rep0
-----------------------------------------------------------
``tau_2`` is 16-19 iterations against blocks that are only 19 iterations long,
so each block completes roughly one decorrelation in temperature space.  That
looked like a promising explanation for the five blocks whose MBAR resample
collapsed to 1-10 distinct configurations: if few walkers reach the lowest
rung, the 310 K pool is fed by few independent sources.

**The hypothesis was tested and refuted.**  Blocks that collapsed had *more*
walkers reaching 310 K than healthy ones (mean 7.6 vs 6.0; Mann-Whitney for
collapsed < healthy p = 0.9996), and walker count correlates weakly and
negatively with the number of distinct frames drawn (Spearman rho = -0.204,
p = 0.09).

What does predict collapse is the size of the post-equilibration frame pool,
which ``t0`` sets:

==============  ==============  ================
group            pool (frames)   unique of 1000
==============  ==============  ================
collapsed (5)             11.6              4.6
healthy (72)             106.1            289.3
==============  ==============  ================

Spearman(pool, unique) = 0.798, p = 3.9e-18; Mann-Whitney pool(collapsed) <
pool(healthy) p = 1.6e-4.  A pool below ~35 frames flags all five collapses
with no false negatives.  So an effective-sample-size gate treats a symptom
whose cause is upstream, and a minimum-pool bound on ``t0`` is both more direct
and knowable *before* the MBAR solve is attempted.  See
:func:`chimpss.analysis.equilibration.bound_pool`.

The mixing numbers remain worth reporting -- a ladder that decorrelates once
per block is a real property of this setup -- they simply are not the cause of
the resampling failures.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    'transition_matrix', 'mixing_time', 'acceptance_from_states',
    'round_trips', 'state_occupancy', 'walkers_reaching_state',
    'block_states', 'concatenated_states', 'mixing_report',
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _block_dirs(run_dir: str) -> List[str]:
    base = os.path.join(run_dir, 'saved_variables')
    names = sorted((d for d in os.listdir(base) if d.isdigit()), key=int)
    return [os.path.join(base, d) for d in names]


def block_states(run_dir: str, block: int) -> np.ndarray:
    """``states.npy`` for one block, shape ``(n_iterations, n_replicas)``."""
    return np.load(os.path.join(run_dir, 'saved_variables', str(block), 'states.npy'))


def concatenated_states(run_dir: str, blocks: Optional[Sequence[int]] = None) -> np.ndarray:
    """Stack every block's states into one ``(total_iterations, n_replicas)`` array.

    Round trips are only meaningful across the whole run: a block is ~19
    iterations and the mixing time is of the same order, so no walker traverses
    the ladder within a single block.  Blocks whose replica count differs from
    the final ladder are skipped rather than padded -- the ladder grew by
    adaptive insertion, and a padded column would fabricate transitions.
    """
    dirs = _block_dirs(run_dir)
    if blocks is not None:
        keep = set(int(b) for b in blocks)
        dirs = [d for d in dirs if int(os.path.basename(d)) in keep]

    arrays = [np.load(os.path.join(d, 'states.npy')) for d in dirs
              if os.path.exists(os.path.join(d, 'states.npy'))]
    if not arrays:
        return np.empty((0, 0), dtype=int)

    width = arrays[-1].shape[1]
    kept = [a for a in arrays if a.shape[1] == width]
    return np.vstack(kept) if kept else np.empty((0, width), dtype=int)


# ---------------------------------------------------------------------------
# Transition matrix and mixing time
# ---------------------------------------------------------------------------

def transition_matrix(states: np.ndarray, n_states: Optional[int] = None) -> np.ndarray:
    """Row-stochastic empirical state-transition matrix.

    ``T[i, j]`` is the fraction of iteration-to-iteration moves in which a
    replica occupying state ``i`` was found in state ``j`` at the next
    iteration, pooled over replicas.  Rows with no observed departures are left
    as an absorbing self-transition rather than divided by zero.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError('states must be (n_iterations, n_replicas) with at least 2 iterations')

    K = int(n_states if n_states is not None else states.max() + 1)
    counts = np.zeros((K, K), dtype=np.float64)
    src = states[:-1].ravel()
    dst = states[1:].ravel()
    np.add.at(counts, (src, dst), 1.0)

    rows = counts.sum(axis=1)
    dead = rows == 0
    counts[dead, :] = 0.0
    counts[dead, np.where(dead)[0]] = 1.0
    rows[dead] = 1.0
    return counts / rows[:, None]


def mixing_time(T: np.ndarray) -> float:
    """Mixing time ``tau_2 = -1 / ln|lambda_2|`` in units of iterations.

    The subdominant eigenvalue of the transition matrix sets the slowest
    relaxation in temperature space (Chodera & Shirts 2011).  Returns ``inf``
    when the chain is reducible or the second eigenvalue is degenerate with the
    first, which is the honest answer: nothing decorrelates.
    """
    ev = np.sort(np.abs(np.linalg.eigvals(np.asarray(T, dtype=np.float64))))[::-1]
    if ev.size < 2:
        return float('inf')
    lam2 = float(ev[1])
    if not (0.0 < lam2 < 1.0):
        return float('inf')
    return -1.0 / np.log(lam2)


def acceptance_from_states(states: np.ndarray) -> float:
    """Fraction of iteration steps on which a replica changed state.

    A coarse companion to the per-pair acceptance the reporter logs; useful
    when only ``states.npy`` survives.
    """
    states = np.asarray(states)
    if states.shape[0] < 2:
        return float('nan')
    return float((states[1:] != states[:-1]).mean())


# ---------------------------------------------------------------------------
# Ladder traversal
# ---------------------------------------------------------------------------

def round_trips(states: np.ndarray, n_states: Optional[int] = None) -> Dict[str, object]:
    """Count full ladder traversals (bottom -> top -> bottom) per walker.

    A round trip is the canonical efficiency measure for parallel tempering:
    it is the rate at which a configuration carries information from the hot
    end of the ladder, where barriers are crossed, to the cold end, where the
    ensemble of interest lives.  Counting half-trips (one bottom-to-top or
    top-to-bottom leg) as well, because on a 110-rung ladder a full round trip
    may never occur within a run.

    Returns per-walker trip counts, the total, and the mean iterations per
    half-trip -- ``inf`` when no walker completes a leg.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError('states must be (n_iterations, n_replicas)')
    K = int(n_states if n_states is not None else states.max() + 1)
    top, bottom = K - 1, 0

    per_walker, half_legs, leg_lengths = [], 0, []
    for w in range(states.shape[1]):
        series = states[:, w]
        target, trips, last_hit = None, 0, None
        for it, s in enumerate(series):
            if target is None:
                if s == bottom:
                    target, last_hit = top, it
                elif s == top:
                    target, last_hit = bottom, it
                continue
            if s == target:
                half_legs += 1
                if last_hit is not None:
                    leg_lengths.append(it - last_hit)
                last_hit = it
                target = bottom if target == top else top
                if target == top:        # returned to the bottom: a full trip
                    trips += 1
        per_walker.append(trips)

    return {
        'per_walker': per_walker,
        'total_round_trips': int(sum(per_walker)),
        'half_legs': int(half_legs),
        'mean_iterations_per_leg': float(np.mean(leg_lengths)) if leg_lengths else float('inf'),
        'n_iterations': int(states.shape[0]),
        'n_states': K,
    }


def state_occupancy(states: np.ndarray, state: int = 0) -> np.ndarray:
    """Number of replicas occupying ``state`` at each iteration."""
    return (np.asarray(states) == int(state)).sum(axis=1)


def walkers_reaching_state(states: np.ndarray, state: int = 0) -> int:
    """Count of distinct walkers that visited ``state`` at least once.

    For ``state=0`` this is the number of independent sources feeding the
    lowest rung, which is the ensemble the analysis reweights to.  Tested
    against the five collapsed blocks of NTSR1 rep0 and found *not* to explain
    them -- see the module docstring.
    """
    return int(np.unique(np.where(np.asarray(states) == int(state))[1]).size)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def mixing_report(run_dir: str, blocks: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Per-block mixing statistics plus whole-run round trips.

    Per-block values use that block's own transition matrix; round trips are
    computed on the concatenated run, because ``tau_2`` is of the same order as
    a block's length and no traversal completes inside one block.
    """
    dirs = _block_dirs(run_dir)
    ids = [int(os.path.basename(d)) for d in dirs]
    if blocks is not None:
        keep = set(int(b) for b in blocks)
        ids = [b for b in ids if b in keep]

    per_block = {}
    for b in ids:
        path = os.path.join(run_dir, 'saved_variables', str(b), 'states.npy')
        if not os.path.exists(path):
            continue
        s = np.load(path)
        T = transition_matrix(s)
        per_block[b] = {
            'n_iterations': int(s.shape[0]),
            'n_replicas': int(s.shape[1]),
            'n_states': int(s.max() + 1),
            'tau_2': mixing_time(T),
            'acceptance': acceptance_from_states(s),
            'walkers_at_state0': walkers_reaching_state(s, 0),
            'mean_occupancy_state0': float(state_occupancy(s, 0).mean()),
        }

    allstates = concatenated_states(run_dir, ids)
    whole = round_trips(allstates) if allstates.size else {}
    taus = [v['tau_2'] for v in per_block.values() if np.isfinite(v['tau_2'])]
    return {
        'per_block': per_block,
        'round_trips': whole,
        'median_tau_2': float(np.median(taus)) if taus else float('inf'),
        'block_length_iterations': int(np.median([v['n_iterations'] for v in per_block.values()]))
        if per_block else 0,
    }
