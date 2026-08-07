"""Assemble the manuscript's composition and production-statistics tables.

Nothing in ChiMPSS previously wrote Table 1 (system composition) or Table 2
(per-run production statistics); the numbers were transcribed by hand.  In
particular the mean nearest-neighbour exchange acceptance is computed inside
``Randolph._eval_acc_rates`` purely to decide state insertion and is never
persisted, so it has to be recovered from the reporter after the fact.

This module reads the artefacts a finished FultonMarket run leaves behind and
emits machine-readable rows:

* :func:`system_composition`  -> Table 1
* :func:`run_statistics`      -> Table 2
* :func:`hmr_report`          -> Table S4 (the HMR mass-transfer specification)

Everything works off ``numpy.load(..., mmap_mode='r')`` against
``saved_variables/<n>/`` rather than instantiating ``FultonMarketAnalysis``,
so it is cheap enough to run over all 30 runs in one job and does not need a
GPU or the JAX stack.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

WATER_RESNAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'TIP'}
POPC_RESNAMES = {'POP', 'POPC', 'PC', 'PA', 'OL'}
SODIUM_RESNAMES = {'NA', 'NA+', 'SOD'}
CHLORIDE_RESNAMES = {'CL', 'CL-', 'CLA'}

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'ASH', 'GLH', 'CYX', 'CYM', 'LYN',
}


# ---------------------------------------------------------------------------
# Table 1 -- system composition
# ---------------------------------------------------------------------------

def system_composition(topology_pdb, box_nm=None):
    """Atom, residue, water, lipid and ion counts from a Bridgeport topology PDB.

    ``box_nm`` is an optional ``(x, y, z)`` in nm -- normally the mean over the
    production trajectory from :func:`mean_box`.  When omitted the CRYST1
    record of the PDB is used, which is the pre-production box.
    """
    import mdtraj as md

    traj = md.load(topology_pdb)
    top = traj.topology

    counts = {
        'atoms': int(top.n_atoms),
        'protein_residues': 0,
        'waters': 0,
        'popc': 0,
        'na': 0,
        'cl': 0,
        'ligand_residues': 0,
    }

    for res in top.residues:
        name = res.name.upper()
        if name in WATER_RESNAMES:
            counts['waters'] += 1
        elif name in POPC_RESNAMES:
            counts['popc'] += 1
        elif name in SODIUM_RESNAMES:
            counts['na'] += 1
        elif name in CHLORIDE_RESNAMES:
            counts['cl'] += 1
        elif name in STANDARD_AA:
            counts['protein_residues'] += 1
        else:
            counts['ligand_residues'] += 1

    # POPC in Lipid17 is split into head/tail residues; report whole lipids.
    if counts['popc']:
        lipid_atoms = sum(r.n_atoms for r in top.residues
                          if r.name.upper() in POPC_RESNAMES)
        counts['popc_atoms'] = int(lipid_atoms)
        counts['popc_residue_fragments'] = counts['popc']

    if box_nm is None and traj.unitcell_lengths is not None:
        box_nm = tuple(float(x) for x in traj.unitcell_lengths[0])

    if box_nm is not None:
        counts['box_nm'] = [round(float(x), 3) for x in box_nm]
        counts['volume_nm3'] = round(float(np.prod(box_nm)), 1)
    else:
        counts['box_nm'] = None
        counts['volume_nm3'] = None

    counts['topology_pdb'] = os.path.abspath(topology_pdb)
    return counts


def mean_box(run_dir, skip_blocks=0):
    """Mean and SD of the production box lengths, in nm.

    Averages ``box_vectors.npy`` across every saved block.  Returns
    ``(mean_xyz, sd_xyz)`` or ``(None, None)`` when no box data is present.
    """
    dirs = _block_dirs(run_dir)[skip_blocks:]
    lengths = []

    for d in dirs:
        path = os.path.join(d, 'box_vectors.npy')
        if not os.path.exists(path):
            continue
        arr = np.load(path, mmap_mode='r')
        a = np.asarray(arr, dtype=float)
        # Stored as (..., 3, 3) box matrices; the diagonal holds the lengths.
        if a.ndim >= 2 and a.shape[-2:] == (3, 3):
            diag = np.stack([a[..., 0, 0], a[..., 1, 1], a[..., 2, 2]], axis=-1)
            lengths.append(diag.reshape(-1, 3))
        elif a.ndim >= 1 and a.shape[-1] == 3:
            lengths.append(a.reshape(-1, 3))

    if not lengths:
        return None, None

    stacked = np.concatenate(lengths, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ---------------------------------------------------------------------------
# Table 2 -- production statistics
# ---------------------------------------------------------------------------

def _block_dirs(run_dir):
    stor = os.path.join(run_dir, 'saved_variables')
    if not os.path.isdir(stor):
        return []
    dirs = [d for d in glob.glob(os.path.join(stor, '*')) if os.path.isdir(d)]
    return sorted(dirs, key=lambda p: int(os.path.basename(p)))


def run_statistics(run_dir, iter_length_ns, dt_fs, mass_scheme=None, label=None):
    """Per-run production statistics for one FultonMarket output directory.

    Returns a dict with the columns of manuscript Table 2, extended with the
    mass scheme and integration settings that the HMR arm makes necessary.
    """
    dirs = _block_dirs(run_dir)

    row = {
        'run': label or os.path.basename(run_dir.rstrip('/')),
        'run_dir': os.path.abspath(run_dir),
        'mass_scheme': mass_scheme,
        'dt_fs': dt_fs,
        'iter_length_ns': iter_length_ns,
        'blocks': len(dirs),
        'final_replicas': None,
        'replicas_per_block': [],
        'iters_per_block': [],
        'aggregate_ns': None,
        'ns_per_replica': None,
        'acceptance_mean': None,
        'acceptance_min_pair': None,
        'states_inserted': None,
    }

    if not dirs:
        row['error'] = f'no saved_variables blocks under {run_dir}'
        return row

    for d in dirs:
        temps = np.load(os.path.join(d, 'temperatures.npy'), mmap_mode='r')
        row['replicas_per_block'].append(int(np.asarray(temps).shape[0]))

        e_path = os.path.join(d, 'energies.npy')
        if os.path.exists(e_path):
            e = np.load(e_path, mmap_mode='r')
            row['iters_per_block'].append(int(e.shape[0]))
        else:
            row['iters_per_block'].append(0)

    row['final_replicas'] = row['replicas_per_block'][-1]
    row['states_inserted'] = row['replicas_per_block'][-1] - row['replicas_per_block'][0]

    # Aggregate simulation time = sum over blocks of iters * replicas * iter_length.
    agg = sum(n_iter * n_rep * iter_length_ns
              for n_iter, n_rep in zip(row['iters_per_block'],
                                       row['replicas_per_block']))
    row['aggregate_ns'] = round(float(agg), 1)
    row['ns_per_replica'] = (round(float(agg / row['final_replicas']), 2)
                             if row['final_replicas'] else None)

    acc = terminal_acceptance(run_dir)
    row.update(acc)
    return row


def terminal_acceptance(run_dir):
    """Mean and minimum nearest-neighbour exchange acceptance at termination.

    Reads the mixing statistics straight out of the reporter, mirroring
    ``Randolph._eval_acc_rates``:  ``mean(accepted[1:] / proposed[1:], axis=0)``
    then the ``(i, i+1)`` band.  Note ``truncate_ncdf`` keeps only the tail of
    the NetCDF, so this is acceptance *at termination*, which is exactly what
    the manuscript reports.
    """
    out = {'acceptance_mean': None, 'acceptance_min_pair': None,
           'acceptance_source': None}

    candidates = [p for p in glob.glob(os.path.join(run_dir, '*.ncdf'))
                  if 'checkpoint' not in os.path.basename(p)
                  and '_copy' not in os.path.basename(p)]
    if not candidates:
        out['acceptance_source'] = 'no trajectory ncdf found'
        return out

    ncdf = candidates[0]
    try:
        from openmmtools.multistate import MultiStateReporter

        reporter = MultiStateReporter(ncdf, open_mode='r')
        try:
            accepted, proposed = reporter.read_mixing_statistics()
        finally:
            reporter.close()

        accepted = np.asarray(accepted, dtype=float)
        proposed = np.asarray(proposed, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.nan_to_num(np.mean(accepted[1:] / proposed[1:], axis=0))

        n = rates.shape[0]
        neighbours = np.array([rates[i, i + 1] for i in range(n - 1)])
        out['acceptance_mean'] = round(float(neighbours.mean()), 4)
        out['acceptance_min_pair'] = round(float(neighbours.min()), 4)
        out['acceptance_source'] = os.path.basename(ncdf)
    except Exception as exc:
        out['acceptance_source'] = f'unreadable ({type(exc).__name__}: {exc})'

    return out


# ---------------------------------------------------------------------------
# Table S4 -- HMR specification
# ---------------------------------------------------------------------------

def hmr_report(base_system_xml, hmr_system_xml, topology_pdb):
    """Quantify exactly what HMR did to a system.

    Compares the force-group system XML against the force-group + HMR XML and
    reports the per-particle mass changes, so the published Methods sentence
    can be written from measurement instead of memory.
    """
    from openmm import XmlSerializer
    from openmm.app import PDBFile
    from openmm.unit import dalton

    with open(base_system_xml) as fh:
        base = XmlSerializer.deserialize(fh.read())
    with open(hmr_system_xml) as fh:
        hmr = XmlSerializer.deserialize(fh.read())

    n = base.getNumParticles()
    if hmr.getNumParticles() != n:
        raise ValueError(
            f'particle count mismatch: {n} in {base_system_xml} vs '
            f'{hmr.getNumParticles()} in {hmr_system_xml}')

    m0 = np.array([base.getParticleMass(i).value_in_unit(dalton) for i in range(n)])
    m1 = np.array([hmr.getParticleMass(i).value_in_unit(dalton) for i in range(n)])
    delta = m1 - m0

    top = PDBFile(topology_pdb).topology
    is_h = np.array([a.element is not None and a.element.symbol == 'H'
                     for a in top.atoms()])
    is_water = np.array([a.residue.name.upper() in WATER_RESNAMES
                         for a in top.atoms()])

    if len(is_h) != n:
        raise ValueError(
            f'topology has {len(is_h)} atoms but system has {n} particles')

    changed = np.abs(delta) > 1e-9
    solute_h = is_h & ~is_water
    water_any = is_water & changed

    final_h = m1[solute_h]
    uniq, counts = np.unique(np.round(final_h, 6), return_counts=True)

    return {
        'base_system_xml': os.path.abspath(base_system_xml),
        'hmr_system_xml': os.path.abspath(hmr_system_xml),
        'n_particles': int(n),
        'n_atoms_changed': int(changed.sum()),
        'n_solute_hydrogens': int(solute_h.sum()),
        'n_water_atoms_changed': int(water_any.sum()),
        'total_mass_before_da': round(float(m0.sum()), 6),
        'total_mass_after_da': round(float(m1.sum()), 6),
        'total_mass_delta_da': float(m1.sum() - m0.sum()),
        'hydrogen_mass_populations': [
            {'final_mass_da': round(float(u_), 6),
             'scale_vs_standard': round(float(u_ / 1.007947), 4),
             'count': int(c)}
            for u_, c in zip(uniq, counts)
        ],
        'min_particle_mass_after_da': round(float(m1.min()), 6),
        'heaviest_debit_da': round(float(-delta.min()), 6),
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def write_tsv(rows, path, columns=None):
    """Write a list of dicts as a TSV. Lists are rendered space-separated."""
    if not rows:
        with open(path, 'w') as fh:
            fh.write('')
        return path

    columns = columns or list(rows[0].keys())
    with open(path, 'w') as fh:
        fh.write('\t'.join(columns) + '\n')
        for row in rows:
            cells = []
            for c in columns:
                v = row.get(c)
                if isinstance(v, (list, tuple)):
                    v = ' '.join(str(x) for x in v)
                cells.append('' if v is None else str(v))
            fh.write('\t'.join(cells) + '\n')
    return path


def write_json(obj, path):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


__all__ = [
    'hmr_report', 'mean_box', 'run_statistics', 'system_composition',
    'terminal_acceptance', 'write_json', 'write_tsv',
]
