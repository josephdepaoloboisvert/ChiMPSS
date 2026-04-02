"""
Geometry utilities for ChiMPSS — RMSD and PBC translation.

Consolidates from FultonMarketUtils.py:
  rmsd (def version), best_translation_by_unitcell (numpy),
  best_translation_by_unitcell_jax (JAX + vmap)
"""

import itertools

import numpy as np


# ---------------------------------------------------------------------------
#  RMSD  (numpy)
# ---------------------------------------------------------------------------

def rmsd(a, b):
    """
    Root-mean-square deviation between coordinate arrays.

    Handles both single-frame ``(N, 3)`` and batched ``(M, N, 3)`` inputs.
    """
    if len(a.shape) == 1:
        return np.sqrt(((a - b) ** 2).sum(-1).mean())
    else:
        return np.array([
            np.sqrt(((a[i] - b[i]) ** 2).sum(-1).mean())
            for i in range(a.shape[0])
        ])


# ---------------------------------------------------------------------------
#  PBC translation  (numpy)
# ---------------------------------------------------------------------------

_PERMS_NP = np.array([x for x in itertools.product([-1, 0, 1], repeat=3)])


def best_translation_by_unitcell(cell_lengths, mobile_coords, target_coords):
    """
    Find the periodic-image translation that minimises RMSD to *target_coords*.

    Parameters
    ----------
    cell_lengths : array-like, shape (3,)
    mobile_coords : array-like, shape (N, 3)
    target_coords : array-like, shape (N, 3)

    Returns
    -------
    best_translation : np.ndarray, shape (3,)
    best_rmsd : float
    """
    translations = cell_lengths * _PERMS_NP
    permuted = np.array([
        np.sum((translations[i], mobile_coords), axis=0)
        for i in range(translations.shape[0])
    ])
    rmsds = np.array([rmsd(permuted[i], target_coords) for i in range(permuted.shape[0])])
    idx = np.argmin(rmsds)
    return translations[idx], rmsds[idx]


# ---------------------------------------------------------------------------
#  JAX variants (lazy-loaded to avoid triggering JAX init on import)
# ---------------------------------------------------------------------------

_JAX_FUNCS = {}


def _init_jax():
    """Build JAX-accelerated helpers on first use."""
    if _JAX_FUNCS:
        return

    import jax
    import jax.numpy as jnp

    perms = jnp.array([x for x in itertools.product([-1, 0, 1], repeat=3)])
    jaxrmsd = lambda a, b: jnp.sqrt(jnp.mean(jnp.sum((b - a) ** 2, axis=-1), axis=-1))
    jax_add = jax.vmap(lambda a, b: a + b, in_axes=(0, None))
    rmsd_j = jax.vmap(jaxrmsd, in_axes=(0, None))

    def _btbu_jax(cell_lengths, mobile_coords, target_coords):
        translations = cell_lengths * perms
        permuted_positions = jax_add(translations, mobile_coords)
        rmsds_of_permutations = rmsd_j(permuted_positions, target_coords)
        idx = jnp.argmin(rmsds_of_permutations)
        return translations[idx], rmsds_of_permutations[idx]

    _JAX_FUNCS['perms'] = perms
    _JAX_FUNCS['jaxrmsd'] = jaxrmsd
    _JAX_FUNCS['jax_add'] = jax_add
    _JAX_FUNCS['rmsd_j'] = rmsd_j
    _JAX_FUNCS['btbu_jax'] = jax.vmap(_btbu_jax, in_axes=(0, 0, None))


def best_translation_by_unitcell_jax(cell_lengths, mobile_coords, target_coords):
    """
    JAX-accelerated, vmapped version of :func:`best_translation_by_unitcell`.

    Expects batched inputs: *cell_lengths* and *mobile_coords* have an extra
    leading axis, *target_coords* is broadcast.
    """
    _init_jax()
    return _JAX_FUNCS['btbu_jax'](cell_lengths, mobile_coords, target_coords)
