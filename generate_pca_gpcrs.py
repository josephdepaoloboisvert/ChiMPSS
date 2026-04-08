#!/usr/bin/env python
"""
Generate a PCA space from GPCR crystal structures.

Fetches GPCRdb metadata and RCSB PDB files in parallel, uses
Ballesteros-Weinstein numbering to identify conserved residues, then
trains a PCA on the selected atom positions across all retained structures.

Outputs (all share the same <prefix>):
  <prefix>_pca.joblib   — trained sklearn PCA model (use pca.transform for new data)
  <prefix>_ref.pdb      — reference structure used for all alignments
  <prefix>_meta.json    — metadata needed by project_pca_gpcrs.py

Usage examples:
  python generate_pca_gpcrs.py 612_gpcrs.txt
  python generate_pca_gpcrs.py 612_gpcrs.txt --selection backbone --prefix gpcr_pca_bb
  python generate_pca_gpcrs.py 612_gpcrs.txt --threshold 0.80 --workers 24
  python generate_pca_gpcrs.py 612_gpcrs.txt --bw_positions "1.50,2.50,3.50,4.50,5.50,6.50,7.50"
  python generate_pca_gpcrs.py 612_gpcrs.txt --bw_positions my_positions.txt
"""

import os
import sys
import json
import argparse

import numpy as np
from MDAnalysis.analysis import align
from sklearn.decomposition import PCA
import joblib

from gpcr_pca_utils import (
    Structure_Analyzer,
    fetch_all_parallel,
    build_bw_assignments,
    conservation_filter,
    build_resids_copopulated,
    select_conserved_atoms,
)


def load_bw_positions(spec):
    """
    Parse a user-supplied BW position list.

    Args:
        spec: either a file path (one label per line, or space/comma separated)
              or an inline string of comma/space-separated labels
              e.g. "1.50,1.51,2.50" or "1.50 1.51 2.50"

    Returns:
        list of BW label strings, sorted by helix then position

    Raises:
        ValueError for malformed labels
    """
    from gpcr_pca_utils import _bw_sort_key

    if os.path.exists(spec):
        with open(spec, 'r') as fh:
            text = fh.read()
    else:
        text = spec

    labels = [tok.strip() for tok in text.replace(',', ' ').split()
              if tok.strip()]

    bad = [l for l in labels if '.' not in l or len(l.split('.')) != 2]
    if bad:
        raise ValueError(
            f"Malformed BW labels (expected 'H.PP' format): {bad}")

    return sorted(labels, key=_bw_sort_key)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        'pdb_list',
        help='Text file with PDB IDs (whitespace or newline separated)')
    p.add_argument(
        '--stor_dir', default='./many_structures/',
        help='Directory for PDB files and per-structure metadata cache '
             '(default: ./many_structures/)')
    p.add_argument(
        '--selection', default='name CA',
        help='MDAnalysis selection applied within conserved residues '
             '(default: "name CA").  '
             'Examples: "backbone", "name CA", "name CA C N O"')
    p.add_argument(
        '--prefix', default='gpcr_pca',
        help='Prefix for all output files (default: gpcr_pca)')
    p.add_argument(
        '--bw_positions', default=None,
        help='Explicit BW positions to use, bypassing the conservation filter. '
             'Accepts a comma/space-separated string ("1.50,1.51,2.50,...") '
             'or a path to a text file with one label per line. '
             'When omitted, positions are chosen by --threshold.')
    p.add_argument(
        '--threshold', type=float, default=0.90,
        help='BW conservation threshold 0–1; used only when --bw_positions is '
             'not given (default: 0.90)')
    p.add_argument(
        '--outlier_cutoff', type=float, default=100.0,
        help='Structures with |PC1| > this value (in the initial PCA) are '
             'dropped as outliers before the final PCA (default: 100.0)')
    p.add_argument(
        '--workers', type=int, default=16,
        help='Parallel HTTP fetch workers (default: 16).  '
             'Keep ≤ 20 to be respectful to RCSB/GPCRdb.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.stor_dir, exist_ok=True)

    # ── 1. Load PDB ID list ───────────────────────────────────────────────────
    with open(args.pdb_list, 'r') as fh:
        pdb_ids = sorted({tok.strip().upper()
                          for tok in fh.read().split()
                          if tok.strip()})
    print(f"Loaded {len(pdb_ids)} PDB IDs from '{args.pdb_list}'")

    # ── 2. Fetch GPCRdb metadata + RCSB PDB files in parallel ────────────────
    tests = [Structure_Analyzer(pid, stor_dir=args.stor_dir) for pid in pdb_ids]
    print(f"Fetching data ({args.workers} parallel workers)...")
    fetch_results = fetch_all_parallel(tests, max_workers=args.workers)
    tests = [t for t in tests if fetch_results.get(t.pdb_code, (False,))[0]]
    print(f"  {len(tests)} / {len(pdb_ids)} structures fetched successfully")

    # ── 3. Load PDB atoms into MDAnalysis ────────────────────────────────────
    print("Loading PDB structures...")
    failed_load = []
    for test in tests:
        try:
            test.load_pdb()
        except Exception as exc:
            print(f"  Load error {test.pdb_code}: {exc}")
            failed_load.append(test.pdb_code)
    tests = [t for t in tests if t.pdb_code not in failed_load]
    print(f"  {len(tests)} structures loaded")

    # ── 4. BW assignments ─────────────────────────────────────────────────────
    print("Building Ballesteros-Weinstein assignments...")
    bw_assignments = build_bw_assignments(tests)
    print(f"  Assignments built for {len(bw_assignments)} structures")

    if args.bw_positions is not None:
        try:
            conserved_bw = load_bw_positions(args.bw_positions)
        except ValueError as exc:
            sys.exit(f"ERROR in --bw_positions: {exc}")
        print(f"  Using {len(conserved_bw)} user-specified BW positions "
              f"(conservation filter bypassed)")
    else:
        conserved_bw = conservation_filter(
            bw_assignments, n_structures=len(tests), threshold=args.threshold)
        print(f"  {len(conserved_bw)} BW positions at "
              f">={args.threshold:.0%} conservation")

    resids_copopulated = build_resids_copopulated(
        tests, bw_assignments, conserved_bw)
    print(f"  {len(resids_copopulated)} structures have all "
          f"conserved positions unambiguously mapped")

    if not resids_copopulated:
        hint = ("Try lowering --threshold."
                if args.bw_positions is None
                else "Check that the supplied BW labels exist in this GPCR family.")
        sys.exit(f"ERROR: no structures survived the BW mapping filter. {hint}")

    # ── 5. Atom selection + modal atom-count filter ───────────────────────────
    test_by_code = {t.pdb_code: t for t in tests}

    selected = []
    for pdb_code, resids in resids_copopulated.items():
        test = test_by_code[pdb_code]
        ag = select_conserved_atoms(test.u_pdb, resids, args.selection)
        selected.append((pdb_code, ag))

    atom_counts = np.array([ag.n_atoms for _, ag in selected])
    values, counts = np.unique(atom_counts, return_counts=True)
    modal_count = int(values[counts.argmax()])
    count_dist = dict(zip(values.tolist(), counts.tolist()))
    print(f"  Atom count distribution: {count_dist}")
    print(f"  Keeping {modal_count} atoms/structure "
          f"({(atom_counts == modal_count).sum()}/{len(selected)} pass)")

    selected = [(code, ag) for code, ag in selected
                if ag.n_atoms == modal_count]

    if not selected:
        sys.exit("ERROR: no structures survived the atom-count filter.")

    # ── 6. Structural alignment to first structure ────────────────────────────
    ref_code, ref_ag = selected[0]
    print(f"Aligning {len(selected)} structures to reference {ref_code}...")
    for _, ag in selected:
        align.alignto(ag, ref_ag, select='all', weights='mass')

    # ── 7. Initial PCA for outlier detection ─────────────────────────────────
    all_positions = np.array([ag.positions for _, ag in selected])
    vectorized = all_positions.reshape(len(selected), -1)

    print("Initial PCA for outlier detection...")
    pca_init = PCA()
    comp_init = pca_init.fit_transform(vectorized)
    not_outlier = np.abs(comp_init[:, 0]) <= args.outlier_cutoff
    n_dropped = int((~not_outlier).sum())

    if n_dropped:
        dropped = [code for (code, _), flag in zip(selected, ~not_outlier)
                   if flag]
        print(f"  Dropping {n_dropped} outliers (|PC1| > {args.outlier_cutoff}): "
              f"{dropped}")
    else:
        print(f"  No outliers found")

    selected_clean = [s for s, keep in zip(selected, not_outlier) if keep]
    vectorized_clean = vectorized[not_outlier]
    codes_retained = [code for code, _ in selected_clean]

    # ── 8. Final PCA ──────────────────────────────────────────────────────────
    print(f"Running final PCA on {len(selected_clean)} structures...")
    pca = PCA()
    train_projections = pca.fit_transform(vectorized_clean)
    evr = pca.explained_variance_ratio_
    print(f"  PC1 {evr[0]:.1%}  PC2 {evr[1]:.1%}  PC3 {evr[2]:.1%}  "
          f"(cumulative top-3: {evr[:3].sum():.1%})")

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    pca_path   = f"{args.prefix}_pca.joblib"
    ref_path   = f"{args.prefix}_ref.pdb"
    meta_path  = f"{args.prefix}_meta.json"
    proj_path  = f"{args.prefix}_train_projections.npy"

    joblib.dump(pca, pca_path)
    print(f"Saved PCA model       → {pca_path}")

    np.save(proj_path, train_projections)
    print(f"Saved train coords    → {proj_path}")

    ref_ag.write(ref_path)
    print(f"Saved reference atoms → {ref_path}")

    # Collect per-structure metadata for downstream coloring / filtering
    structure_meta = {
        code: test_by_code[code].meta.get('structure', {})
        for code in codes_retained
    }

    meta = {
        'selection':     args.selection,
        'conserved_bw':  conserved_bw,
        'n_atoms':       modal_count,
        'ref_pdb_code':  ref_code,
        'ref_pdb_path':  ref_path,
        'codes_retained': codes_retained,
        'threshold':     args.threshold,
        'stor_dir':      args.stor_dir,
        'structure_meta': structure_meta,   # species, state, etc. for plotting
    }
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved metadata        → {meta_path}")

    print(f"\nDone. {len(codes_retained)} structures in the PCA space.")
    print(f"To project a new trajectory:\n"
          f"  python project_pca_gpcrs.py "
          f"--pdb_code <CODE> --topology <top> --trajectory <traj> "
          f"--prefix {args.prefix}")


if __name__ == '__main__':
    main()
