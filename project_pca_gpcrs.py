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
      3. Verify each expected resid exists in u (optionally within chain).

    Args:
        bw_map:        {sequence_number (int): bw_label (str)} from GPCRdb
        conserved_bw:  ordered list of BW labels from PCA metadata
        u:             MDAnalysis Universe of the MD system
        chain:         chain ID string, or None for no chain filter
        resid_offset:  integer offset applied to GPCRdb sequence numbers

    Returns:
        list of [[resid], ...] in the same order as conserved_bw

    Raises:
        ValueError if any conserved position cannot be mapped
    """
    # Reverse: bw_label → sequence_number
    label_to_seqnum = {lbl: seqnum for seqnum, lbl in bw_map.items()
                       if lbl != '-1'}

    # Available resids in the relevant part of the universe
    chain_sel = f"chainid {chain} and " if chain else ""
    available = set(u.select_atoms(f"{chain_sel}backbone").residues.resids)

    result = []
    missing = []
    for label in conserved_bw:
        if label not in label_to_seqnum:
            missing.append(f"{label}: not in GPCRdb BW map for this protein")
            result.append([])
            continue
        traj_resid = label_to_seqnum[label] + resid_offset
        if traj_resid not in available:
            missing.append(
                f"{label}: GPCRdb seqnum {label_to_seqnum[label]} "
                f"→ trajectory resid {traj_resid} not found"
                + (f" in chain {chain}" if chain else ""))
            result.append([])
        else:
            result.append([traj_resid])

    if missing:
        msg = (
            f"{len(missing)} conserved BW positions could not be mapped:\n"
            + "\n".join(f"  {m}" for m in missing[:20])
            + (f"\n  ... and {len(missing)-20} more" if len(missing) > 20 else "")
            + "\n\nTips:"
            + "\n  • If residue numbering differs from UniProt, try --resid_offset N"
            + "\n  • If the GPCR is in a specific chain, use --chain <ID>"
            + "\n  • The PDB code you supply should be the crystal structure "
              "the MD was modelled on"
        )
        raise ValueError(msg)

    return result


# ── atom selection and trajectory projection ───────────────────────────────────

def build_mobile_ag(u, resids, selection, chain=None):
    """
    Select atoms from universe u matching conserved resids and selection string.
    """
    resid_str = " or resid ".join(str(r[0]) for r in resids)
    chain_filter = f"chainid {chain} and " if chain else ""
    return u.select_atoms(f"{chain_filter}(resid {resid_str}) and ({selection})")


def project_trajectory(u, mobile_ag, ref_ag, pca, report_every=100):
    """
    Iterate all frames, align mobile_ag to ref_ag, vectorize, then
    project onto the PCA space with pca.transform (no re-fitting).

    Args:
        u:            MDAnalysis Universe (trajectory already loaded)
        mobile_ag:    AtomGroup of conserved residue atoms in the trajectory
        ref_ag:       AtomGroup of the same atoms from the training reference
        pca:          fitted sklearn PCA object
        report_every: print progress every N frames

    Returns:
        np.ndarray of shape (n_frames, n_components), dtype float32
    """
    n_frames   = len(u.trajectory)
    n_features = mobile_ag.n_atoms * 3
    vectorized = np.empty((n_frames, n_features), dtype=np.float32)

    print(f"Projecting {n_frames} frame(s)...")
    for i, _ts in enumerate(u.trajectory):
        align.alignto(mobile_ag, ref_ag, select='all', weights='mass')
        # .flatten() always returns a copy — positions are overwritten next frame
        vectorized[i] = mobile_ag.positions.flatten()
        if (i + 1) % report_every == 0 or (i + 1) == n_frames:
            print(f"  {i + 1}/{n_frames}", end='\r', flush=True)
    print()  # newline after \r progress

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
    try:
        resids = map_conserved_resids(
            bw_map, conserved_bw, u,
            chain=args.chain,
            resid_offset=args.resid_offset)
    except ValueError as exc:
        sys.exit(f"ERROR:\n{exc}")

    # ── Select and verify atom count ──────────────────────────────────────────
    mobile_ag = build_mobile_ag(u, resids, selection, chain=args.chain)
    if mobile_ag.n_atoms != expected_n:
        sys.exit(
            f"ERROR: selected {mobile_ag.n_atoms} atoms, expected {expected_n}.\n"
            f"  • Check --chain (currently: {args.chain})\n"
            f"  • Check that all conserved residues are present and complete\n"
            f"  • The training selection was '{selection}'")
    print(f"Selected  : {mobile_ag.n_atoms} atoms ({len(conserved_bw)} residues)")

    # ── Project all frames ────────────────────────────────────────────────────
    projections = project_trajectory(u, mobile_ag, ref_ag, pca)

    np.save(args.out, projections)
    print(f"\nSaved projections → {args.out}  shape: {projections.shape}")
    print(f"  PC1  [{projections[:, 0].min():.2f}, {projections[:, 0].max():.2f}]")
    print(f"  PC2  [{projections[:, 1].min():.2f}, {projections[:, 1].max():.2f}]")
    if projections.shape[1] > 2:
        print(f"  PC3  [{projections[:, 2].min():.2f}, {projections[:, 2].max():.2f}]")


if __name__ == '__main__':
    main()
