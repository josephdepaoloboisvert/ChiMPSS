r"""Equilibration detection in an *external* pan-GPCR PCA basis.

Why an external basis
---------------------
:meth:`FultonMarketAnalysis.determine_equilibration` can already detect
equilibration in a PCA space, but it fits that space on the very trajectory it
is judging.  PC1 is then by construction the direction of largest variance in
that data, and for a relaxing simulation the largest variance *is* the
relaxation.  Finding a transient along PC1 is close to tautological, and on
``NTSR1_ML-301.hmr.rep0`` that branch duly returned the latest ``t0`` of
anything tried (530 frames, 72.7% discarded) off a strided sample of only 73
frames per component, 33 of whose 40 components returned ``t0 = 0``.

Projecting instead onto a basis fitted on experimental GPCR structures
(``chimpss-generate-pca``) fixes the axes *before* the simulation is seen.
That breaks the circularity, gives axes with physical meaning -- the training
set separates active from inactive states -- and puts every run and every
receptor in one common coordinate system, which a per-trajectory PCA can never
do because each run gets its own axes.

The honest caveat is that the basis is built from *inter-structure* variance
across many receptors, so it is an open question whether it resolves
*intra-trajectory* relaxation of a single receptor.  It is also restricted to
conserved Ballesteros-Weinstein positions in the transmembrane bundle, hence
blind to loops, ICL3 and the ligand pocket.  Both are measured rather than
assumed: see :func:`basis_comparison`, which reports the trajectory's spread
along each PC against the training spread.

Numbering is not a formality
----------------------------
Crystallisation constructs carry internal deletions, and when a topology is
renumbered contiguously the break becomes invisible to a single-offset search.
On ``NTSR1_ML-301`` the best single offset reaches 64.6% sequence identity and
silently mis-assigns all of TM6, TM7 and H8 by four residues -- 87 of the 262
conserved positions.  This module therefore maps residues through
:func:`~chimpss.analysis.gpcr_pca.align_sequence_to_gpcrdb`, a gap-aware global
alignment, which recovers 342/342 residues at 90.4% identity and resolves the
two offsets (+1 and +5) that the deletion creates.

Units
-----
``positions.npy`` is written by OpenMM in **nanometres**; the PCA is trained on
PDB coordinates in **angstroms**.  Every frame is scaled by 10 before
projection.  Getting this wrong does not raise -- it silently rescales every
component by an order of magnitude -- so :func:`project_coordinates` applies it
in one place and nowhere else.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from chimpss.analysis.equilibration import (DETECTORS, _block_dirs,
                                            _load_positions)
from chimpss.analysis.gpcr_pca import (align_sequence_to_gpcrdb,
                                       build_gpcrdb_sequence,
                                       map_conserved_resids_aligned,
                                       naming_from_convention)

__all__ = [
    'PanGPCRModel', 'load_model', 'topology_sequence', 'resolve_atoms',
    'kabsch_superpose', 'project_coordinates', 'collect_frames',
    'pangpcr_projection', 'self_basis_projection', 'detect_pangpcr',
    'per_pc_table', 'training_spread', 'basis_comparison', 'NM_TO_ANGSTROM',
]

#: ``positions.npy`` is nanometres; the crystal-structure PCA is angstroms.
NM_TO_ANGSTROM = 10.0

_THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
_NORM = {'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS', 'HSD': 'HIS',
         'HSE': 'HIS', 'HSP': 'HIS', 'CYX': 'CYS', 'CYM': 'CYS',
         'ASH': 'ASP', 'GLH': 'GLU', 'LYN': 'LYS'}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class PanGPCRModel:
    """A trained pan-GPCR PCA together with everything needed to project onto it."""
    pca: object
    meta: dict
    ref_positions: np.ndarray          # (n_atoms, 3) angstroms
    ref_masses: np.ndarray             # (n_atoms,)

    @property
    def conserved_bw(self) -> List[str]:
        return self.meta['conserved_bw']

    @property
    def selection(self) -> str:
        return self.meta['selection']

    @property
    def n_atoms(self) -> int:
        return int(self.meta['n_atoms'])

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component variance fractions.

        ``sklearn`` returns these already per-component.  They are *not* the
        cumulative curve that :func:`chimpss.fultonmarket.utils.get_traj_PCA`
        builds, and must never be used interchangeably -- doing so gives tail
        components more weight than PC1.
        """
        return np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64)


def load_model(prefix: str) -> PanGPCRModel:
    """Load the ``*_pca.joblib`` / ``*_meta.json`` / ``*_ref.pdb`` triple."""
    import joblib
    import mdtraj as md

    pca_path, meta_path, ref_path = (f'{prefix}_pca.joblib',
                                     f'{prefix}_meta.json',
                                     f'{prefix}_ref.pdb')
    for p in (pca_path, meta_path, ref_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' not found. Build the basis first:\n"
                f"  chimpss-generate-pca <pdb_list> --prefix {prefix}")

    pca = joblib.load(pca_path)
    with open(meta_path) as fh:
        meta = json.load(fh)

    ref = md.load(ref_path)
    # mdtraj reads PDB in nm; the training coordinates were angstroms.
    ref_xyz = np.asarray(ref.xyz[0], dtype=np.float64) * NM_TO_ANGSTROM
    masses = np.array([a.element.mass for a in ref.topology.atoms], dtype=np.float64)

    if ref_xyz.shape[0] != int(meta['n_atoms']):
        raise ValueError(
            f"reference '{ref_path}' has {ref_xyz.shape[0]} atoms but metadata "
            f"says {meta['n_atoms']}")
    return PanGPCRModel(pca=pca, meta=meta, ref_positions=ref_xyz, ref_masses=masses)


# ---------------------------------------------------------------------------
# Residue and atom resolution
# ---------------------------------------------------------------------------

def topology_sequence(topology, chain_index: int = 0) -> List[Tuple[int, str]]:
    """``[(resid, one_letter), ...]`` for one chain of an mdtraj topology.

    Non-standard names are normalised (``HID``/``HIE``/``HIP`` all map to
    histidine), and anything that is still not a canonical amino acid is
    dropped -- lipids, waters and ions must not enter the alignment.
    """
    chain = list(topology.chains)[chain_index]
    out = []
    for res in chain.residues:
        letter = _THREE2ONE.get(_NORM.get(res.name.upper(), res.name.upper()))
        if letter:
            out.append((int(res.resSeq), letter))
    return out


def resolve_atoms(topology, naming_info: Sequence[dict], model: PanGPCRModel,
                  chain_index: int = 0,
                  scheme: str = 'BW') -> Tuple[np.ndarray, list, dict]:
    """Resolve the model's conserved BW positions to atom indices in ``topology``.

    ``naming_info`` is the raw GPCRdb ``residues/extended`` payload for this
    receptor; both the sequence and the BW labels are derived from it, so the
    two can never disagree about a sequence number.

    Returns ``(atom_indices, missing_info, stats)``.  ``atom_indices`` indexes
    the *full* system, so it slices ``positions.npy`` directly.  ``stats``
    carries the alignment diagnostics -- in particular ``offsets``, whose
    length exceeds one exactly when the construct has an internal indel.
    """
    gpcrdb_seq = build_gpcrdb_sequence(naming_info)
    bw_map = naming_from_convention(naming_info, scheme=scheme)

    traj_seq = topology_sequence(topology, chain_index)
    seq_map, stats = align_sequence_to_gpcrdb(traj_seq, gpcrdb_seq)

    chain = list(topology.chains)[chain_index]
    by_resid = {int(r.resSeq): r for r in chain.residues}
    resids, missing_info = map_conserved_resids_aligned(
        bw_map, model.conserved_bw, seq_map, available=set(by_resid))

    names = _selection_atom_names(model.selection)
    atom_indices: List[int] = []
    resolved: List[List[int]] = []
    for idx, (label, rid) in enumerate(zip(model.conserved_bw, resids)):
        if not rid:
            resolved.append([])
            continue
        res = by_resid[rid[0]]
        hit = [int(a.index) for a in res.atoms if a.name in names]
        if len(hit) != len(names):
            missing_info.append(
                (label, idx,
                 f'resid {rid[0]} carries {len(hit)} of {len(names)} atoms '
                 f'for selection {model.selection!r}'))
            resolved.append([])
            continue
        # keep the training order of atom names within a residue
        order = {n: k for k, n in enumerate(names)}
        hit.sort(key=lambda i: order[topology.atom(i).name])
        atom_indices.extend(hit)
        resolved.append(hit)

    stats['n_positions'] = len(model.conserved_bw)
    stats['n_resolved'] = sum(1 for r in resolved if r)
    stats['atoms_per_residue'] = len(names)
    return np.asarray(atom_indices, dtype=np.int64), missing_info, stats


def _selection_atom_names(selection: str) -> List[str]:
    """Atom names implied by the training selection string.

    Only the selections ``chimpss-generate-pca`` realistically uses are
    supported, and an unknown one raises rather than guessing -- a silent
    mismatch in atom count or ordering would corrupt every projection.
    """
    sel = ' '.join(selection.lower().split())
    if sel in ('name ca', 'protein and name ca'):
        return ['CA']
    if sel == 'backbone':
        return ['N', 'CA', 'C', 'O']
    if sel.startswith('name '):
        return [tok.upper() for tok in sel[5:].split()]
    raise ValueError(
        f"unsupported training selection {selection!r}; pass a 'name ...' "
        f"selection or 'backbone'")


# ---------------------------------------------------------------------------
# Superposition and projection
# ---------------------------------------------------------------------------

def kabsch_superpose(mobile: np.ndarray, ref: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Mass-weighted optimal superposition of ``mobile`` onto ``ref``.

    Reproduces ``MDAnalysis.analysis.align.alignto(..., weights='mass')``,
    which is what ``chimpss-generate-pca`` used to build the training frame.
    For a pure alpha-carbon selection every mass is equal and this reduces to
    plain Kabsch, but the backbone selection has four distinct masses and the
    weighting then matters.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if mobile.shape != ref.shape:
        raise ValueError(f'shape mismatch: {mobile.shape} vs {ref.shape}')

    w = (np.ones(mobile.shape[0]) if weights is None
         else np.asarray(weights, dtype=np.float64))
    w = w / w.sum()

    com_m = (w[:, None] * mobile).sum(axis=0)
    com_r = (w[:, None] * ref).sum(axis=0)
    p = mobile - com_m
    q = ref - com_r

    h = (p * w[:, None]).T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return p @ rot.T + com_r


def project_coordinates(coords_nm: np.ndarray, model: PanGPCRModel,
                        imputation: Optional[dict] = None) -> np.ndarray:
    """Project frames of selected atoms onto the pan-GPCR space.

    ``coords_nm`` is ``(n_frames, n_selected, 3)`` in nanometres, already
    sliced to the model's conserved atoms and in the model's atom order.  The
    scale conversion to angstroms happens here and only here.

    When ``imputation`` is given (from
    :func:`~chimpss.analysis.gpcr_pca._imputation_setup`) the missing features
    are filled with the training mean, which contributes zero to every
    component, and the superposition uses only the present atoms.
    """
    x = np.asarray(coords_nm, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    x = x * NM_TO_ANGSTROM

    if imputation is None:
        ref = model.ref_positions
        w = model.ref_masses
        vec = np.empty((x.shape[0], model.n_atoms * 3), dtype=np.float64)
    else:
        keep = imputation['trimmed_ref_atom_idx']
        ref = model.ref_positions[keep]
        w = model.ref_masses[keep]
        present = imputation['present_feat_idx']
        mean = np.asarray(model.pca.mean_, dtype=np.float64)
        vec = np.repeat(mean[None], x.shape[0], axis=0)

    if x.shape[1] != ref.shape[0]:
        raise ValueError(
            f'got {x.shape[1]} atoms per frame, expected {ref.shape[0]}')

    for i in range(x.shape[0]):
        fitted = kabsch_superpose(x[i], ref, w).ravel()
        if imputation is None:
            vec[i] = fitted
        else:
            vec[i, present] = fitted

    return model.pca.transform(vec).astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory projection
# ---------------------------------------------------------------------------

def model_fingerprint(model: PanGPCRModel) -> str:
    """Short stable digest of the atom set a model selects.

    The cached coordinate array holds only the atoms one basis asked for, so it
    is not interchangeable between bases.  Keying the cache on state and stride
    alone lets a 190-atom model silently load a 204-atom array built by a
    different basis; the projection then fails on an atom-count mismatch, or
    worse, succeeds against the wrong atoms if the counts happen to agree.
    """
    import hashlib
    key = f"{model.selection}|{model.n_atoms}|{'|'.join(model.conserved_bw)}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def collect_frames(run_dir: str, topology: str, model: PanGPCRModel,
                   naming_info: Sequence[dict], state: int = 0,
                   skip: int = 10, chain_index: int = 0,
                   cache: bool = True, max_blocks: Optional[int] = None,
                   _printf: Callable = print) -> Tuple[np.ndarray, dict]:
    """Read the conserved atoms of one state's time-ordered trajectory.

    ``states.npy`` records which replica occupies each state at each iteration,
    so the lowest rung -- the ensemble the analysis reweights to -- can be
    followed even though replicas permute.  This mirrors
    :func:`~chimpss.analysis.equilibration.structural_series` exactly, so the
    two series are frame-for-frame comparable; ``skip`` likewise discards the
    first ``skip`` iterations *of each block*, it is not a stride.

    Only the conserved atoms are read out of each ``positions.npy`` memmap --
    a few hundred of 76,003 -- so the pass is cheap despite the 1.9 GB blocks.

    Returns ``(coords_nm, info)`` with ``coords_nm`` of shape
    ``(n_frames, n_selected, 3)`` in nanometres.  Both the external projection
    and the self-fitted control are built from this one array, which is what
    makes them a controlled comparison rather than two different experiments.
    """
    import mdtraj as md

    tag = f'state{state}_skip{skip}_{model_fingerprint(model)}'
    out_path = os.path.join(run_dir, 'analysis', f'pangpcr_coords_{tag}.npy')
    info_path = os.path.join(run_dir, 'analysis', f'pangpcr_coords_{tag}.json')
    if cache and os.path.exists(out_path) and os.path.exists(info_path):
        with open(info_path) as fh:
            return np.load(out_path), json.load(fh)

    top_traj = md.load(topology)
    top = top_traj.topology
    n_atoms_total = top_traj.n_atoms

    atom_indices, missing_info, stats = resolve_atoms(
        top, naming_info, model, chain_index=chain_index)

    stats['missing'] = [[lbl, int(i), reason] for lbl, i, reason in missing_info]
    stats['atom_indices'] = atom_indices.tolist()
    stats['state'] = int(state)
    stats['skip'] = int(skip)

    dirs = _block_dirs(run_dir)
    if max_blocks is not None:
        dirs = dirs[:max_blocks]

    frames: List[np.ndarray] = []
    block_of: List[int] = []
    for d in dirs:
        ppath = os.path.join(d, 'positions.npy')
        spath = os.path.join(d, 'states.npy')
        if not (os.path.exists(ppath) and os.path.exists(spath)):
            continue
        st = np.load(spath)
        tpath = os.path.join(d, 'temperatures.npy')
        n_replicas = int(np.load(tpath).size) if os.path.exists(tpath) else st.shape[1]
        pos = _load_positions(ppath, st.shape[0], n_replicas, n_atoms_total)
        if pos is None:
            _printf(f'  block {os.path.basename(d)}: positions unreadable, skipped')
            continue
        n_it = min(st.shape[0], pos.shape[0])
        for it in range(skip, n_it):
            hit = np.where(st[it] == state)[0]
            if hit.size == 0:
                continue
            frames.append(np.asarray(pos[it, hit[0]][atom_indices], dtype=np.float32))
            block_of.append(int(os.path.basename(d)))
        del pos

    coords = (np.stack(frames) if frames
              else np.empty((0, atom_indices.size, 3), dtype=np.float32))
    stats['n_frames'] = int(coords.shape[0])
    stats['blocks'] = block_of

    if cache:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, coords)
        with open(info_path, 'w') as fh:
            json.dump(stats, fh, indent=2)
    return coords, stats


def _imputation_for(info: dict, model: PanGPCRModel) -> Optional[dict]:
    """Rebuild the imputation indices from cached ``collect_frames`` metadata."""
    missing = [(lbl, int(i), reason) for lbl, i, reason in info.get('missing', [])]
    if not missing:
        return None
    from chimpss.analysis.gpcr_pca import _imputation_setup
    return _imputation_setup(missing, model.conserved_bw, model.n_atoms, model.pca)


def pangpcr_projection(run_dir: str, topology: str, model: PanGPCRModel,
                       naming_info: Sequence[dict], state: int = 0,
                       skip: int = 10, chain_index: int = 0,
                       cache: bool = True, max_blocks: Optional[int] = None,
                       _printf: Callable = print) -> Tuple[np.ndarray, dict]:
    """Project the time-ordered trajectory of one state onto the external basis.

    Returns ``(projection, info)`` with ``projection`` of shape
    ``(n_frames, n_components)``.
    """
    coords, info = collect_frames(
        run_dir, topology, model, naming_info, state=state, skip=skip,
        chain_index=chain_index, cache=cache, max_blocks=max_blocks,
        _printf=_printf)
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32), info

    imputation = _imputation_for(info, model)
    if imputation is not None:
        info['pc1_imputed_frac'] = float(imputation['pc1_imputed_frac'])
        info['pc2_imputed_frac'] = float(imputation['pc2_imputed_frac'])

    projection = project_coordinates(coords, model, imputation=imputation)
    info['n_components'] = int(projection.shape[1])
    return projection, info


def self_basis_projection(coords_nm: np.ndarray, model: PanGPCRModel,
                          imputation: Optional[dict] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a PCA on these very frames and project them onto it.

    This is the circularity control.  It uses the *same atoms*, the *same
    frames* and the *same superposition* as the external basis, so the only
    thing that differs is whether the axes were chosen before or after seeing
    the data.  Any gap between the two ``t0`` estimates is attributable to that
    and nothing else.

    Returns ``(projection, explained_variance_ratio)``.
    """
    from sklearn.decomposition import PCA

    x = np.asarray(coords_nm, dtype=np.float64) * NM_TO_ANGSTROM
    if imputation is None:
        ref, w = model.ref_positions, model.ref_masses
    else:
        keep = imputation['trimmed_ref_atom_idx']
        ref, w = model.ref_positions[keep], model.ref_masses[keep]

    fitted = np.stack([kabsch_superpose(f, ref, w).ravel() for f in x])
    pca = PCA()
    proj = pca.fit_transform(fitted).astype(np.float32)
    return proj, np.asarray(pca.explained_variance_ratio_, dtype=np.float64)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def per_pc_table(projection: np.ndarray, evr: np.ndarray,
                 detector: Callable[..., int], n_pcs: int) -> List[dict]:
    """Run one detector on each of the leading ``n_pcs`` component series."""
    rows = []
    n = projection.shape[0]
    for k in range(min(n_pcs, projection.shape[1])):
        series = np.asarray(projection[:, k], dtype=np.float64)
        try:
            t0 = int(detector(series))
            err = None
        except Exception as exc:
            t0, err = None, f'{type(exc).__name__}: {exc}'
        rows.append({
            'pc': k + 1,
            'evr': float(evr[k]),
            't0': t0,
            'discard': (t0 / n) if (t0 is not None and n) else None,
            'spread': float(series.std()),
            'error': err,
        })
    return rows


def detect_pangpcr(projection: np.ndarray, evr: np.ndarray,
                   detector: str = 'max_neff',
                   n_pcs: Optional[int] = None,
                   cumulative_variance: float = 0.90) -> dict:
    """Consensus ``t0`` over the leading components, weighted by variance.

    Components are weighted by their **per-component** explained-variance
    fraction, renormalised over the components actually used.  This is the
    distinction that the trajectory-fitted branch originally got wrong, where
    the cumulative curve was passed as if it were per-component weights and
    tail components consequently outvoted PC1.

    ``n_pcs`` defaults to however many leading components are needed to reach
    ``cumulative_variance`` of the training variance.
    """
    evr = np.asarray(evr, dtype=np.float64)
    if projection.size == 0:
        raise ValueError('empty projection')

    if n_pcs is None:
        cum = np.cumsum(evr)
        n_pcs = int(np.searchsorted(cum, cumulative_variance) + 1)
        n_pcs = max(1, min(n_pcs, projection.shape[1]))

    fn = DETECTORS[detector] if isinstance(detector, str) else detector
    rows = per_pc_table(projection, evr, fn, n_pcs)

    ok = [r for r in rows if r['t0'] is not None]
    if not ok:
        raise RuntimeError(f'detector {detector!r} failed on every component')

    w = np.array([r['evr'] for r in ok], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(ok), 1.0 / len(ok))
    t0 = int(round(float(np.sum(np.array([r['t0'] for r in ok], dtype=np.float64) * w))))

    return {
        'detector': detector if isinstance(detector, str) else 'custom',
        'n_pcs': int(n_pcs),
        'n_frames': int(projection.shape[0]),
        't0': t0,
        'discard': t0 / projection.shape[0],
        't0_pc1': ok[0]['t0'] if ok and ok[0]['pc'] == 1 else None,
        'per_pc': rows,
    }


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def training_spread(model: PanGPCRModel, prefix: str,
                    n_pcs: int = 5) -> Optional[dict]:
    """Spread of the training structures along each PC, and state separation.

    Two things are worth knowing before trusting a ``t0`` from this basis.
    First, how much of the pan-GPCR space the trajectory actually explores: if
    the run occupies a sliver of the training spread, the axes are dominated by
    receptor-to-receptor differences rather than by anything this simulation
    does.  Second, whether PC1/PC2 separate active from inactive structures at
    all, which is the claim that makes the axes physically meaningful.

    Returns ``None`` when the training projections were not saved.
    """
    proj_path = f'{prefix}_train_projections.npy'
    if not os.path.exists(proj_path):
        return None
    train = np.load(proj_path)
    codes = model.meta.get('codes_retained', [])
    smeta = model.meta.get('structure_meta', {})
    states = [smeta.get(c, {}).get('state') or 'Unknown' for c in codes]

    k = min(n_pcs, train.shape[1])
    out = {
        'n_structures': int(train.shape[0]),
        'std': train[:, :k].std(axis=0).tolist(),
        'range': (train[:, :k].max(axis=0) - train[:, :k].min(axis=0)).tolist(),
        'by_state': {},
    }
    for st in sorted(set(states)):
        mask = np.array([s == st for s in states])
        if mask.sum() == 0:
            continue
        out['by_state'][st] = {
            'n': int(mask.sum()),
            'mean': train[mask, :k].mean(axis=0).tolist(),
            'std': train[mask, :k].std(axis=0).tolist(),
        }
    act, inact = out['by_state'].get('Active'), out['by_state'].get('Inactive')
    if act and inact:
        d = np.array(act['mean'][:2]) - np.array(inact['mean'][:2])
        pooled = np.sqrt((np.array(act['std'][:2]) ** 2
                          + np.array(inact['std'][:2]) ** 2) / 2.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            out['active_inactive_cohens_d'] = np.where(
                pooled > 0, np.abs(d) / pooled, np.nan).tolist()
    return out


def basis_comparison(run_dir: str, topology: str, model: PanGPCRModel,
                     naming_info: Sequence[dict], prefix: str,
                     state: int = 0, skip: int = 10, chain_index: int = 0,
                     detectors: Optional[Sequence[str]] = None,
                     n_pcs: Optional[int] = None,
                     energy_pdb: Optional[str] = None,
                     cache: bool = True,
                     _printf: Callable = print) -> dict:
    """Run every detector against every available observable and tabulate.

    The observables are, in increasing distance from the incumbent:

    ``energy``
        replica-averaged reduced potential -- what production actually uses.
    ``structural``
        alpha-carbon RMSD to the final frame, the observable the convergence
        criterion implicitly cares about.
    ``self_pca``
        PCA fitted on the conserved atoms of these very frames: the circular
        basis, retained as a control.
    ``pangpcr``
        the external crystal-structure basis.

    ``self_pca`` and ``pangpcr`` share atoms, frames and superposition, so the
    difference between their ``t0`` values isolates the effect of fitting the
    axes on the data being judged.
    """
    detectors = list(detectors or ('max_neff', 'mser', 'mser_sqrt', 'mser_w50'))
    report: dict = {'run_dir': run_dir, 'state': state, 'skip': skip,
                    'detectors': detectors, 'observables': {}}

    coords, info = collect_frames(
        run_dir, topology, model, naming_info, state=state, skip=skip,
        chain_index=chain_index, cache=cache, _printf=_printf)
    report['mapping'] = {k: v for k, v in info.items()
                         if k not in ('atom_indices', 'blocks')}
    if coords.size == 0:
        return report

    imputation = _imputation_for(info, model)
    ext_proj = project_coordinates(coords, model, imputation=imputation)
    self_proj, self_evr = self_basis_projection(coords, model, imputation)

    bases = {
        'pangpcr': (ext_proj, model.explained_variance_ratio),
        'self_pca': (self_proj, self_evr),
    }
    for name, (proj, evr) in bases.items():
        entry: dict = {'n_frames': int(proj.shape[0]),
                       'n_components': int(proj.shape[1]),
                       'by_detector': {}}
        for det in detectors:
            try:
                entry['by_detector'][det] = detect_pangpcr(
                    proj, evr, detector=det, n_pcs=n_pcs)
            except Exception as exc:
                entry['by_detector'][det] = {'error': f'{type(exc).__name__}: {exc}'}
        entry['trajectory_std'] = proj[:, :5].std(axis=0).tolist()
        report['observables'][name] = entry

    # scalar observables, for continuity with the existing sweep
    from chimpss.analysis.equilibration import (compare_detectors,
                                                energy_series,
                                                structural_series)
    scalar = {}
    if energy_pdb:
        try:
            scalar['energy'] = energy_series(run_dir, energy_pdb, skip=skip)
        except Exception as exc:
            _printf(f'  energy series unavailable: {exc}')
        try:
            scalar['structural'] = structural_series(
                run_dir, energy_pdb, skip=skip, state=state, cache=cache)
        except Exception as exc:
            _printf(f'  structural series unavailable: {exc}')
    for name, series in scalar.items():
        if series is None or np.asarray(series).size == 0:
            continue
        rows = compare_detectors(series, {d: DETECTORS[d] for d in detectors})
        report['observables'][name] = {
            'n_frames': int(np.asarray(series).size),
            'by_detector': rows,
        }

    spread = training_spread(model, prefix)
    if spread is not None:
        report['training'] = spread
        traj_std = np.asarray(report['observables']['pangpcr']['trajectory_std'])
        train_std = np.asarray(spread['std'][:traj_std.size])
        with np.errstate(divide='ignore', invalid='ignore'):
            report['spread_ratio'] = np.where(
                train_std > 0, traj_std / train_std, np.nan).tolist()
    return report
