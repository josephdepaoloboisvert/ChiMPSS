"""Class-A GPCR activation-motif measurements, indexed by Ballesteros-Weinstein label.

The ChiMPSS manuscript reports five motif measurements that had no
implementation anywhere in the package:

==========================  =========================================================
Measurement                 Definition used here
==========================  =========================================================
``tm36_distance``           Ca-Ca distance between BW 3.50 and 6.34.  The canonical
                            scalar readout of cytoplasmic TM6 opening.
``npxxy_rmsd``              Backbone RMSD of the NPxxY motif (7.49-7.53) after
                            superposing the receptor on TM1-TM4 (an activation-
                            invariant frame), measured against a supplied reference.
``tm7_h8_distance``         Ca-Ca distance between BW 7.53 and 8.50, the TM7-helix 8
                            junction that restructures as TM6 swings outward.
``w648_chi1``               chi1 of the residue at BW 6.48 -- the classical "toggle
                            switch".  Returns NaN when 6.48 is not a Trp.
``dry_ionic_lock``          Minimum heavy-atom distance between the BW 3.50 arginine
                            guanidinium (NE/NH1/NH2) and the BW 6.30 acidic side-chain
                            carboxylate (OE1/OE2 or OD1/OD2).  NaN when 6.30 is not
                            Asp/Glu, which is the case in most receptors.
==========================  =========================================================

Everything is keyed off Ballesteros-Weinstein labels rather than raw residue
numbers so the same code runs unmodified on any class-A receptor and on the
GPCRdb reference structures.  BW->resid translation reuses
:func:`chimpss.analysis.gpcr_pca.fetch_bw_map`,
:func:`~chimpss.analysis.gpcr_pca.auto_detect_resid_offset` and
:func:`~chimpss.analysis.gpcr_pca.map_conserved_resids`, so the numbering
conventions match the PCA projection exactly.

Typical use::

    from chimpss.analysis.motifs import measure_trajectory

    df = measure_trajectory(pdb_code='6ZA8',
                            topology='NTSR1_ML-301.equil.pdb',
                            trajectory='resampled_trj.dcd',
                            chain=None,
                            stor_dir='./many_structures/')

Or from the command line via ``chimpss-motifs``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from chimpss.analysis.gpcr_pca import (
    Structure_Analyzer,
    auto_detect_resid_offset,
    fetch_bw_map,
    map_conserved_resids,
)

# Labels this module needs.  Kept as one list so a single map_conserved_resids
# call resolves all of them and reports exactly which are unavailable.
MOTIF_BW_LABELS = ['3.49', '3.50', '3.51', '6.30', '6.34', '6.48',
                   '7.49', '7.50', '7.51', '7.52', '7.53', '8.50']

# TM1-TM4 anchor used as the activation-invariant superposition frame for the
# NPxxY RMSD.  These helices barely move between active and inactive states, so
# residual NPxxY displacement is attributable to the motif itself.
NPXXY_FRAME_BW = ['1.46', '1.50', '1.54', '2.46', '2.50', '2.54',
                  '3.36', '3.40', '3.44', '4.50', '4.54']

NPXXY_BW = ['7.49', '7.50', '7.51', '7.52', '7.53']

ACIDIC = {'ASP', 'GLU', 'ASH', 'GLH'}
ARGININE = {'ARG'}
TRYPTOPHAN = {'TRP'}

MOTIF_COLUMNS = ['tm36_distance', 'npxxy_rmsd', 'tm7_h8_distance',
                 'w648_chi1', 'dry_ionic_lock']


# ---------------------------------------------------------------------------
# BW resolution
# ---------------------------------------------------------------------------

def resolve_bw_resids(u, pdb_code, stor_dir, chain=None, labels=None,
                      resid_offset=0, auto_offset=True):
    """Map BW labels to resids in universe ``u``.

    Returns ``(resid_by_label, missing)`` where ``resid_by_label`` maps each
    resolved label to an int resid and ``missing`` maps each unresolved label
    to the reason string produced by ``map_conserved_resids``.
    """
    labels = list(labels or (MOTIF_BW_LABELS + NPXXY_FRAME_BW))
    labels = sorted(set(labels))

    analyzer, bw_map = fetch_bw_map(pdb_code, stor_dir)

    if auto_offset:
        resid_offset = auto_detect_resid_offset(
            u, analyzer, chain=chain, user_offset=resid_offset)

    resids, missing_info = map_conserved_resids(
        bw_map, labels, u, chain=chain, resid_offset=resid_offset)

    resid_by_label = {}
    for label, entry in zip(labels, resids):
        if entry:
            resid_by_label[label] = int(entry[0])

    missing = {lbl: reason for lbl, _, reason in missing_info}
    return resid_by_label, missing


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _sel(u, resid, extra, chain=None):
    chain_filter = f"chainid {chain} and " if chain else ""
    return u.select_atoms(f"{chain_filter}resid {resid} and ({extra})")


def _ca(u, resid, chain=None):
    return _sel(u, resid, "name CA", chain=chain)


def _resname(u, resid, chain=None):
    ag = _sel(u, resid, "name CA", chain=chain)
    return ag.residues[0].resname.upper() if ag.n_atoms else None


# ---------------------------------------------------------------------------
# Individual measurements -- each returns a float (NaN when not measurable)
# ---------------------------------------------------------------------------

def tm36_distance(u, resid_by_label, chain=None):
    """Ca-Ca distance between BW 3.50 and 6.34, in Angstrom."""
    if '3.50' not in resid_by_label or '6.34' not in resid_by_label:
        return np.nan
    a = _ca(u, resid_by_label['3.50'], chain)
    b = _ca(u, resid_by_label['6.34'], chain)
    if a.n_atoms != 1 or b.n_atoms != 1:
        return np.nan
    return float(np.linalg.norm(a.positions[0] - b.positions[0]))


def tm7_h8_distance(u, resid_by_label, chain=None):
    """Ca-Ca distance between BW 7.53 and 8.50, in Angstrom."""
    if '7.53' not in resid_by_label or '8.50' not in resid_by_label:
        return np.nan
    a = _ca(u, resid_by_label['7.53'], chain)
    b = _ca(u, resid_by_label['8.50'], chain)
    if a.n_atoms != 1 or b.n_atoms != 1:
        return np.nan
    return float(np.linalg.norm(a.positions[0] - b.positions[0]))


def w648_chi1(u, resid_by_label, chain=None):
    """chi1 (N-CA-CB-CG) of the BW 6.48 residue, in degrees on (-180, 180].

    Returns NaN when 6.48 is unresolved or is not a tryptophan -- the toggle
    switch is only defined for Trp6.48.
    """
    if '6.48' not in resid_by_label:
        return np.nan
    resid = resid_by_label['6.48']
    if _resname(u, resid, chain) not in TRYPTOPHAN:
        return np.nan

    coords = []
    for name in ('N', 'CA', 'CB', 'CG'):
        ag = _sel(u, resid, f"name {name}", chain=chain)
        if ag.n_atoms != 1:
            return np.nan
        coords.append(ag.positions[0])
    return _dihedral(*coords)


def dry_ionic_lock(u, resid_by_label, chain=None):
    """Minimum guanidinium(3.50)-carboxylate(6.30) heavy-atom distance, Angstrom.

    NaN unless 3.50 is Arg and 6.30 is Asp/Glu.  Most class-A receptors do not
    carry an acidic residue at 6.30, so NaN is the expected result rather than
    a failure -- the manuscript notes the lock is measurable only in
    CB2R:AM630 and 5HT2AR:Serotonin.
    """
    if '3.50' not in resid_by_label or '6.30' not in resid_by_label:
        return np.nan

    arg_resid = resid_by_label['3.50']
    acid_resid = resid_by_label['6.30']

    if _resname(u, arg_resid, chain) not in ARGININE:
        return np.nan
    if _resname(u, acid_resid, chain) not in ACIDIC:
        return np.nan

    guan = _sel(u, arg_resid, "name NE NH1 NH2", chain=chain)
    carbox = _sel(u, acid_resid, "name OE1 OE2 OD1 OD2", chain=chain)
    if guan.n_atoms == 0 or carbox.n_atoms == 0:
        return np.nan

    d = np.linalg.norm(
        guan.positions[:, None, :] - carbox.positions[None, :, :], axis=-1)
    return float(d.min())


def npxxy_rmsd(u, resid_by_label, reference_frame, chain=None):
    """Backbone RMSD of NPxxY (7.49-7.53) against ``reference_frame``.

    ``reference_frame`` is the dict returned by :func:`build_npxxy_reference`.
    The receptor is superposed on the TM1-TM4 anchor first, so the number
    reports NPxxY displacement rather than whole-bundle motion.
    """
    if reference_frame is None:
        return np.nan

    frame_ag = _backbone_ag(u, reference_frame['frame_labels'],
                            resid_by_label, chain)
    motif_ag = _backbone_ag(u, reference_frame['motif_labels'],
                            resid_by_label, chain)
    if frame_ag is None or motif_ag is None:
        return np.nan
    if frame_ag.n_atoms != reference_frame['frame_xyz'].shape[0]:
        return np.nan
    if motif_ag.n_atoms != reference_frame['motif_xyz'].shape[0]:
        return np.nan

    rot, mob_com, ref_com = _kabsch(frame_ag.positions,
                                    reference_frame['frame_xyz'])
    moved = (motif_ag.positions - mob_com) @ rot.T + ref_com
    diff = moved - reference_frame['motif_xyz']
    return float(np.sqrt((diff ** 2).sum() / len(diff)))


def _backbone_ag(u, labels, resid_by_label, chain=None):
    """Backbone atoms for ``labels``, in a fixed atom order, or None if incomplete."""
    chain_filter = f"chainid {chain} and " if chain else ""
    groups = []
    for label in labels:
        if label not in resid_by_label:
            return None
        ag = u.select_atoms(
            f"{chain_filter}resid {resid_by_label[label]} and name N CA C O")
        if ag.n_atoms != 4:
            return None
        groups.append(ag)
    total = groups[0]
    for g in groups[1:]:
        total = total + g
    return total


def _kabsch(mobile, reference):
    """Return (rotation, mobile_centroid, reference_centroid) minimising RMSD."""
    mob_com = mobile.mean(axis=0)
    ref_com = reference.mean(axis=0)
    p = mobile - mob_com
    q = reference - ref_com
    v, _, wt = np.linalg.svd(p.T @ q)
    if np.linalg.det(v) * np.linalg.det(wt) < 0.0:
        v[:, -1] = -v[:, -1]
    return (v @ wt).T, mob_com, ref_com


def _dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


# ---------------------------------------------------------------------------
# Reference frame for the NPxxY RMSD
# ---------------------------------------------------------------------------

def build_npxxy_reference(u, resid_by_label, chain=None):
    """Freeze the current frame of ``u`` as the NPxxY RMSD reference.

    Returns None when either the TM1-TM4 anchor or the motif itself cannot be
    fully resolved -- callers should treat that as "npxxy_rmsd unavailable"
    rather than an error.
    """
    frame_labels = [lbl for lbl in NPXXY_FRAME_BW if lbl in resid_by_label]
    motif_labels = [lbl for lbl in NPXXY_BW if lbl in resid_by_label]

    if len(frame_labels) < 6 or len(motif_labels) < 4:
        return None

    frame_ag = _backbone_ag(u, frame_labels, resid_by_label, chain)
    motif_ag = _backbone_ag(u, motif_labels, resid_by_label, chain)
    if frame_ag is None or motif_ag is None:
        return None

    return {
        'frame_labels': frame_labels,
        'motif_labels': motif_labels,
        'frame_xyz': frame_ag.positions.copy(),
        'motif_xyz': motif_ag.positions.copy(),
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def measure_frame(u, resid_by_label, chain=None, npxxy_reference=None):
    """All five motifs for the current frame of ``u``, as a dict."""
    return {
        'tm36_distance':    tm36_distance(u, resid_by_label, chain),
        'npxxy_rmsd':       npxxy_rmsd(u, resid_by_label, npxxy_reference, chain),
        'tm7_h8_distance':  tm7_h8_distance(u, resid_by_label, chain),
        'w648_chi1':        w648_chi1(u, resid_by_label, chain),
        'dry_ionic_lock':   dry_ionic_lock(u, resid_by_label, chain),
    }


def measure_trajectory(pdb_code, topology, trajectory=None, chain=None,
                       stor_dir='./many_structures/', resid_offset=0,
                       auto_offset=True, npxxy_reference=None, verbose=True):
    """Measure all five motifs over every frame.

    Returns ``(array, meta)`` where ``array`` has shape
    ``(n_frames, len(MOTIF_COLUMNS))`` in the column order of
    :data:`MOTIF_COLUMNS`, and ``meta`` records the resolved resids, anything
    that could not be resolved, and why each NaN column is NaN.
    """
    import MDAnalysis as mda

    u = mda.Universe(topology, trajectory) if trajectory else mda.Universe(topology)

    resid_by_label, missing = resolve_bw_resids(
        u, pdb_code, stor_dir, chain=chain,
        resid_offset=resid_offset, auto_offset=auto_offset)

    if verbose:
        print(f"Resolved {len(resid_by_label)} BW positions; "
              f"{len(missing)} unavailable")
        for lbl, reason in sorted(missing.items()):
            print(f"    [{lbl}] {reason}")

    if npxxy_reference is None:
        npxxy_reference = build_npxxy_reference(u, resid_by_label, chain)
        if verbose and npxxy_reference is None:
            print("    npxxy_rmsd unavailable: TM1-TM4 anchor or motif incomplete")

    rows = []
    for _ in u.trajectory:
        row = measure_frame(u, resid_by_label, chain, npxxy_reference)
        rows.append([row[c] for c in MOTIF_COLUMNS])

    arr = np.asarray(rows, dtype=float)

    meta = {
        'pdb_code':      pdb_code.upper(),
        'topology':      os.path.abspath(topology),
        'trajectory':    os.path.abspath(trajectory) if trajectory else None,
        'chain':         chain,
        'n_frames':      int(arr.shape[0]),
        'columns':       list(MOTIF_COLUMNS),
        'resid_by_label': resid_by_label,
        'missing_bw':    missing,
        'resname_at': {
            lbl: _resname(u, resid, chain)
            for lbl, resid in resid_by_label.items()
            if lbl in ('3.50', '6.30', '6.48')
        },
        'all_nan_columns': [c for i, c in enumerate(MOTIF_COLUMNS)
                            if arr.shape[0] and np.all(np.isnan(arr[:, i]))],
    }
    return arr, meta


def summarize(arr):
    """Mean/SD/n per column, NaN-safe. Returns a dict keyed by column name."""
    out = {}
    for i, col in enumerate(MOTIF_COLUMNS):
        v = arr[:, i] if arr.size else np.array([])
        finite = v[np.isfinite(v)] if v.size else v
        out[col] = {
            'mean': float(np.mean(finite)) if finite.size else None,
            'std':  float(np.std(finite)) if finite.size else None,
            'n':    int(finite.size),
        }
    return out


def measure_reference_set(pca_meta_path, stor_dir='./many_structures/',
                          verbose=True):
    """Measure the motifs on every structure in a generated PCA training set.

    Reads ``<prefix>_meta.json`` from ``chimpss-generate-pca``, which lists the
    retained PDB codes and their GPCRdb activation-state annotation, and walks
    the cached PDB files in ``stor_dir``.  Returns ``(rows, by_state)`` where
    ``rows`` is a list of per-structure dicts and ``by_state`` gives the
    active/inactive/intermediate population means the simulation ensembles are
    compared against.
    """
    import MDAnalysis as mda

    with open(pca_meta_path) as fh:
        pca_meta = json.load(fh)

    codes = pca_meta['codes_retained']
    structure_meta = pca_meta.get('structure_meta', {})

    rows = []
    for n, code in enumerate(codes, 1):
        pdb_path = os.path.join(stor_dir, f"{code}.pdb")
        if not os.path.exists(pdb_path):
            if verbose:
                print(f"  [{n}/{len(codes)}] {code}: cached PDB missing, skipped")
            continue
        try:
            u = mda.Universe(pdb_path)
            analyzer = Structure_Analyzer(code.upper(), stor_dir=stor_dir)
            ok, err = analyzer.fetch_all()
            if not ok:
                raise RuntimeError(err)
            bw_map = analyzer.getSchemeNames()
            labels = sorted(set(MOTIF_BW_LABELS + NPXXY_FRAME_BW))
            resids, _ = map_conserved_resids(bw_map, labels, u)
            resid_by_label = {lbl: int(r[0])
                              for lbl, r in zip(labels, resids) if r}
            values = measure_frame(u, resid_by_label, None, None)
        except Exception as exc:
            if verbose:
                print(f"  [{n}/{len(codes)}] {code}: {exc}")
            continue

        values['pdb_code'] = code
        values['state'] = structure_meta.get(code, {}).get('state')
        rows.append(values)
        if verbose and n % 50 == 0:
            print(f"  [{n}/{len(codes)}] measured {len(rows)} structures so far")

    by_state = {}
    for state in ('Active', 'Inactive', 'Intermediate'):
        subset = [r for r in rows if (r.get('state') or '') == state]
        by_state[state] = {
            'n': len(subset),
            **{col: _nanstats([r[col] for r in subset]) for col in MOTIF_COLUMNS},
        }

    return rows, by_state


def _nanstats(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {'mean': None, 'std': None, 'n': 0}
    return {'mean': float(arr.mean()), 'std': float(arr.std()), 'n': int(arr.size)}


__all__ = [
    'MOTIF_BW_LABELS', 'MOTIF_COLUMNS', 'NPXXY_BW', 'NPXXY_FRAME_BW',
    'build_npxxy_reference', 'dry_ionic_lock', 'measure_frame',
    'measure_reference_set', 'measure_trajectory', 'npxxy_rmsd',
    'resolve_bw_resids', 'summarize', 'tm36_distance', 'tm7_h8_distance',
    'w648_chi1',
]
