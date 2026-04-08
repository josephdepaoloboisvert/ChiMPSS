#!/usr/bin/env python
"""
Project MD trajectory (or single structure) frames onto a pre-computed GPCR PCA space.

The protein's BW mapping is always fetched fresh from GPCRdb — it is never
assumed to be part of the training set.

The key assumption is that residue numbers in the topology/trajectory match
the UniProt sequence numbers used by GPCRdb (this is typically true when the
MD system was built directly from an RCSB PDB file without renumbering).
If your system uses different residue numbering, see --resid_offset.

Outputs:
  <out>   — numpy .npy array, shape (n_frames, n_pcs)

Usage examples:
  # MD trajectory
  python project_pca_gpcrs.py \\
      --pdb_code 6CMO --topology system.pdb --trajectory md.xtc \\
      --chain A --prefix gpcr_pca

  # Single crystal structure (no trajectory)
  python project_pca_gpcrs.py \\
      --pdb_code 6CMO --topology structure.pdb --prefix gpcr_pca

  # With a residue number offset (e.g. trajectory resids are 1-based, PDB is 34-based)
  python project_pca_gpcrs.py \\
      --pdb_code 6CMO --topology system.pdb --trajectory md.xtc \\
      --resid_offset -33 --prefix gpcr_pca
"""

import os
import sys
import json
import argparse

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import joblib

from gpcr_pca_utils import (
    Structure_Analyzer,
    naming_from_convention,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--pdb_code', required=True,
        help='PDB code of the parent crystal structure used to look up BW '
             'numbering via GPCRdb (e.g. 6CMO).  Does not need to be in the '
             'training set.')
    p.add_argument(
        '--topology', required=True,
        help='Topology file for the MD system (PDB, GRO, PSF, …)')
    p.add_argument(
        '--trajectory', default=None,
        help='Trajectory file (XTC, DCD, …).  Omit to project a single frame.')
    p.add_argument(
        '--chain', default=None,
        help='Chain ID of the GPCR in the trajectory.  '
             'If omitted, no chain filter is applied (use for single-chain systems).')
    p.add_argument(
        '--prefix', default='gpcr_pca',
        help='Prefix of the PCA files produced by generate_pca_gpcrs.py '
             '(default: gpcr_pca)')
    p.add_argument(
        '--stor_dir', default='./many_structures/',
        help='Cache directory for GPCRdb metadata (default: ./many_structures/)')
    p.add_argument(
        '--resid_offset', type=int, default=0,
        help='Integer added to GPCRdb sequence numbers to obtain resids in '
             'your topology (default: 0).  '
             'Example: if GPCRdb says resid 34 but your topology starts at 1, '
             'use --resid_offset -33.')
    p.add_argument(
        '--out', default=None,
        help='Output .npy file (default: <pdb_code>_projections.npy)')
    return p.parse_args()


# ── BW → trajectory resid mapping ─────────────────────────────────────────────

def fetch_bw_map(pdb_code, stor_dir):
    """
    Fetch GPCRdb BW map for pdb_code, using the per-structure disk cache.

    Returns:
        (Structure_Analyzer, bw_map {sequence_number: bw_label})
    """
    analyzer = Structure_Analyzer(pdb_code.upper(), stor_dir=stor_dir)
    print(f"Fetching GPCRdb data for {pdb_code.upper()}...")
    ok, err = analyzer.fetch_all()
    if not ok:
        sys.exit(f"ERROR: could not fetch data for {pdb_code}: {err}")
    bw_map = analyzer.getSchemeNames()
    protein = analyzer.meta.get('structure', {}).get('protein', '?')
    print(f"  Protein slug: {protein}  |  {len(bw_map)} residues in BW map")
    return analyzer, bw_map


def map_conserved_resids(bw_map, conserved_bw, u, chain=None, resid_offset=0):
    """
    Translate conserved BW labels → resids present in universe u.

    Steps:
      1. Build reverse map {bw_label: sequence_number} from bw_map.
      2. Apply resid_offset: trajectory_resid = sequence_number + resid_offset.
      3. Check each expected resid against the universe (optionally within chain).

    Args:
        bw_map:        {sequence_number (int): bw_label (str)} from GPCRdb
        conserved_bw:  ordered list of BW labels from PCA metadata
        u:             MDAnalysis Universe of the MD system
        chain:         chain ID string, or None for no chain filter
        resid_offset:  integer offset applied to GPCRdb sequence numbers

    Returns:
        resids:       list of [[resid], ...] in conserved_bw order; [] for missing
        missing_info: list of (bw_label, bw_index, reason_string) for every gap
    """
    label_to_seqnum = {lbl: seqnum for seqnum, lbl in bw_map.items()
                       if lbl != '-1'}

    chain_sel = f"chainid {chain} and " if chain else ""
    available = set(u.select_atoms(f"{chain_sel}backbone").residues.resids)

    resids      = []
    missing_info = []
    for idx, label in enumerate(conserved_bw):
        if label not in label_to_seqnum:
            reason = f"not in GPCRdb BW map for this protein"
            missing_info.append((label, idx, reason))
            resids.append([])
        else:
            traj_resid = label_to_seqnum[label] + resid_offset
            if traj_resid not in available:
                chain_note = f" in chain {chain}" if chain else ""
                reason = (f"GPCRdb seqnum {label_to_seqnum[label]} "
                          f"→ trajectory resid {traj_resid} not found{chain_note}")
                missing_info.append((label, idx, reason))
                resids.append([])
            else:
                resids.append([traj_resid])

    return resids, missing_info


# ── atom selection and trajectory projection ───────────────────────────────────

def build_mobile_ag(u, resids, selection, chain=None):
    """
    Select atoms from universe u matching conserved resids and selection string.
    Silently skips empty resid entries (missing positions).
    """
    present = [r for r in resids if r]
    resid_str = " or resid ".join(str(r[0]) for r in present)
    chain_filter = f"chainid {chain} and " if chain else ""
    return u.select_atoms(f"{chain_filter}(resid {resid_str}) and ({selection})")


def _imputation_setup(missing_info, conserved_bw, expected_n, pca):
    """
    Pre-compute everything needed for mean imputation.

    The feature vector has `expected_n * 3` elements, ordered by residue
    position in conserved_bw. Each residue contributes atoms_per_res atoms
    (and atoms_per_res * 3 features). atoms_per_res is derived from the
    expected atom count and is identical for every residue (guaranteed by the
    modal atom-count filter in generate_pca_gpcrs.py).

    Args:
        missing_info:  list of (bw_label, bw_index, reason) from map_conserved_resids
        conserved_bw:  ordered list of all BW labels
        expected_n:    total atom count from PCA metadata
        pca:           fitted sklearn PCA

    Returns dict with keys:
        present_feat_idx:     1-D int array of feature indices for present residues
        missing_feat_idx:     1-D int array of feature indices to impute
        trimmed_ref_atom_idx: 1-D int array indexing ref_ag for the present residues
        pc1_imputed_frac:     fraction of PC1 loading norm covered by imputed features
        pc2_imputed_frac:     fraction of PC2 loading norm covered by imputed features
    """
    n_residues    = len(conserved_bw)
    atoms_per_res = expected_n // n_residues
    feats_per_res = atoms_per_res * 3

    if expected_n % n_residues != 0:
        raise ValueError(
            f"expected_n ({expected_n}) is not divisible by n_residues "
            f"({n_residues}). Cannot determine atoms-per-residue for imputation.")

    missing_bw_indices = {bw_idx for _, bw_idx, _ in missing_info}
    present_bw_indices = [i for i in range(n_residues) if i not in missing_bw_indices]

    def feat_range(bw_i):
        return np.arange(bw_i * feats_per_res, (bw_i + 1) * feats_per_res)

    def atom_range(bw_i):
        return np.arange(bw_i * atoms_per_res, (bw_i + 1) * atoms_per_res)

    present_feat_idx     = np.concatenate([feat_range(i) for i in present_bw_indices])
    missing_feat_idx     = np.concatenate([feat_range(i) for i in missing_bw_indices])
    trimmed_ref_atom_idx = np.concatenate([atom_range(i) for i in present_bw_indices])

    # Fraction of each PC's loading L2-norm that lives in the imputed features.
    # After mean-centering, imputed features contribute zero → this fraction of
    # information is lost. Print as a diagnostic.
    def loading_frac(pc_idx):
        comps = pca.components_[pc_idx]
        total = np.sum(comps ** 2)          # = 1.0 by PCA construction
        imputed = np.sum(comps[missing_feat_idx] ** 2)
        return imputed / total if total > 0 else 0.0

    return {
        'present_feat_idx':     present_feat_idx,
        'missing_feat_idx':     missing_feat_idx,
        'trimmed_ref_atom_idx': trimmed_ref_atom_idx,
        'pc1_imputed_frac':     loading_frac(0),
        'pc2_imputed_frac':     loading_frac(1),
    }


def project_trajectory(u, mobile_ag, ref_ag, pca,
                       imputation=None, report_every=100):
    """
    Iterate all frames, align mobile_ag to ref_ag, vectorize, then
    project onto the PCA space with pca.transform (no re-fitting).

    Args:
        u:            MDAnalysis Universe (trajectory already loaded)
        mobile_ag:    AtomGroup of conserved residue atoms in the trajectory
                      (may be smaller than ref_ag when imputation is active)
        ref_ag:       AtomGroup of the training reference atoms (full set)
        pca:          fitted sklearn PCA object
        imputation:   None for normal projection, or the dict returned by
                      _imputation_setup for mean-imputed projection
        report_every: print progress every N frames

    Returns:
        np.ndarray of shape (n_frames, n_components), dtype float32
    """
    n_frames = len(u.trajectory)

    if imputation is None:
        # ── normal path ───────────────────────────────────────────────────────
        n_features = mobile_ag.n_atoms * 3
        vectorized = np.empty((n_frames, n_features), dtype=np.float32)
        print(f"Projecting {n_frames} frame(s)...")
        for i, _ts in enumerate(u.trajectory):
            align.alignto(mobile_ag, ref_ag, select='all', weights='mass')
            vectorized[i] = mobile_ag.positions.flatten()
            if (i + 1) % report_every == 0 or (i + 1) == n_frames:
                print(f"  {i + 1}/{n_frames}", end='\r', flush=True)
    else:
        # ── imputation path ───────────────────────────────────────────────────
        present_feat_idx     = imputation['present_feat_idx']
        missing_feat_idx     = imputation['missing_feat_idx']
        trimmed_ref_atom_idx = imputation['trimmed_ref_atom_idx']

        trimmed_ref_ag = ref_ag[trimmed_ref_atom_idx]
        n_features     = pca.mean_.shape[0]
        # Build template with mean values pre-filled; only present features
        # will be overwritten per frame.
        mean_template = pca.mean_.astype(np.float32)

        vectorized = np.empty((n_frames, n_features), dtype=np.float32)
        print(f"Projecting {n_frames} frame(s) [mean imputation active]...")
        for i, _ts in enumerate(u.trajectory):
            align.alignto(mobile_ag, trimmed_ref_ag, select='all', weights='mass')
            vec = mean_template.copy()
            vec[present_feat_idx] = mobile_ag.positions.flatten()
            vectorized[i] = vec
            if (i + 1) % report_every == 0 or (i + 1) == n_frames:
                print(f"  {i + 1}/{n_frames}", end='\r', flush=True)

    print()
    return pca.transform(vectorized).astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.out is None:
        args.out = f"{args.pdb_code.upper()}_projections.npy"

    # ── Load PCA artifacts ────────────────────────────────────────────────────
    pca_path  = f"{args.prefix}_pca.joblib"
    meta_path = f"{args.prefix}_meta.json"
    ref_path  = f"{args.prefix}_ref.pdb"

    for path in (pca_path, meta_path, ref_path):
        if not os.path.exists(path):
            sys.exit(f"ERROR: '{path}' not found. "
                     f"Run generate_pca_gpcrs.py --prefix {args.prefix} first.")

    pca = joblib.load(pca_path)
    with open(meta_path, 'r') as fh:
        meta = json.load(fh)

    selection    = meta['selection']
    conserved_bw = meta['conserved_bw']
    expected_n   = meta['n_atoms']

    print(f"Loaded PCA: {pca.n_components_} components, "
          f"trained on {len(meta['codes_retained'])} structures")
    print(f"Selection : '{selection}'  |  Expected atoms: {expected_n}")

    # ── Load reference AtomGroup for alignment ────────────────────────────────
    ref_u  = mda.Universe(ref_path)
    ref_ag = ref_u.select_atoms('all')
    if ref_ag.n_atoms != expected_n:
        sys.exit(
            f"ERROR: reference file '{ref_path}' has {ref_ag.n_atoms} atoms "
            f"but metadata says {expected_n}.  Regenerate with the same "
            f"--selection '{selection}'.")

    # ── Fetch BW map for the query protein ───────────────────────────────────
    os.makedirs(args.stor_dir, exist_ok=True)
    analyzer, bw_map = fetch_bw_map(args.pdb_code, args.stor_dir)

    # ── Load MD universe ──────────────────────────────────────────────────────
    if args.trajectory:
        u = mda.Universe(args.topology, args.trajectory)
    else:
        u = mda.Universe(args.topology)
    print(f"Universe  : {len(u.trajectory)} frame(s), {u.atoms.n_atoms} atoms")
    if args.chain:
        print(f"Chain     : {args.chain}")
    if args.resid_offset:
        print(f"Resid offset: {args.resid_offset:+d}")

    # ── Map conserved BW positions → trajectory resids ────────────────────────
    print("Mapping conserved BW positions to trajectory residues...")
    resids, missing_info = map_conserved_resids(
        bw_map, conserved_bw, u,
        chain=args.chain,
        resid_offset=args.resid_offset)

    imputation = None

    if missing_info:
        n_missing = len(missing_info)
        if n_missing > 2:
            lines = "\n".join(f"  [{lbl}] {reason}"
                              for lbl, _, reason in missing_info)
            raise Exception(
                f"{n_missing} conserved BW positions are missing from this "
                f"structure — too many for mean imputation (limit: 2).\n"
                f"{lines}\n\n"
                f"Tips:\n"
                f"  • Exclude these positions with --bw_exclude in "
                f"generate_pca_gpcrs.py and retrain\n"
                f"  • Check --resid_offset if numbering differs from UniProt\n"
                f"  • Check --chain if the GPCR is not the only chain")

        # 1–2 missing: warn and prepare imputation
        print(f"\n  WARNING: {n_missing} residue(s) missing — "
              f"will be mean-imputed (contribute zero to all PCs):")
        for lbl, _, reason in missing_info:
            print(f"    [{lbl}] {reason}")

        imputation = _imputation_setup(
            missing_info, conserved_bw, expected_n, pca)

        print(f"  PC1 loading norm from imputed features: "
              f"{imputation['pc1_imputed_frac']:.1%}")
        print(f"  PC2 loading norm from imputed features: "
              f"{imputation['pc2_imputed_frac']:.1%}")
        if imputation['pc1_imputed_frac'] > 0.10 or imputation['pc2_imputed_frac'] > 0.10:
            print(f"  WARNING: imputed features carry >10% of a PC's loading — "
                  f"interpret projections with caution.")
        print()
    else:
        print(f"  All {len(conserved_bw)} conserved positions mapped successfully")

    # ── Select and verify atom count ──────────────────────────────────────────
    mobile_ag = build_mobile_ag(u, resids, selection, chain=args.chain)
    expected_mobile = expected_n if imputation is None else len(imputation['present_feat_idx']) // 3
    if mobile_ag.n_atoms != expected_mobile:
        sys.exit(
            f"ERROR: selected {mobile_ag.n_atoms} atoms, expected {expected_mobile}.\n"
            f"  • Check --chain (currently: {args.chain})\n"
            f"  • Check that all conserved residues are present and complete\n"
            f"  • The training selection was '{selection}'")
    print(f"Selected  : {mobile_ag.n_atoms} atoms "
          f"({len(conserved_bw) - len(missing_info)} residues present"
          + (f", {len(missing_info)} imputed" if missing_info else "")
          + ")")

    # ── Project all frames ────────────────────────────────────────────────────
    projections = project_trajectory(u, mobile_ag, ref_ag, pca,
                                     imputation=imputation)

    np.save(args.out, projections)
    print(f"\nSaved projections → {args.out}  shape: {projections.shape}")
    print(f"  PC1  [{projections[:, 0].min():.2f}, {projections[:, 0].max():.2f}]")
    print(f"  PC2  [{projections[:, 1].min():.2f}, {projections[:, 1].max():.2f}]")
    if projections.shape[1] > 2:
        print(f"  PC3  [{projections[:, 2].min():.2f}, {projections[:, 2].max():.2f}]")


if __name__ == '__main__':
    main()
