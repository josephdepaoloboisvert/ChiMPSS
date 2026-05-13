"""
CLI entry point for generating a GPCR PCA space from crystal structures.

Console script: chimpss-generate-pca

Usage examples:
  chimpss-generate-pca 612_gpcrs.txt
  chimpss-generate-pca 612_gpcrs.txt --selection backbone --prefix gpcr_pca_bb
  chimpss-generate-pca 612_gpcrs.txt --threshold 0.80 --workers 24
  chimpss-generate-pca 612_gpcrs.txt --bw_positions "1.50,2.50,3.50"
"""

import argparse
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
from MDAnalysis.analysis import align
from sklearn.decomposition import PCA

from chimpss.analysis.gpcr_pca import (
    Structure_Analyzer,
    build_bw_assignments,
    build_resids_copopulated,
    conservation_filter,
    fetch_all_parallel,
    load_bw_exclude,
    load_bw_positions,
    select_conserved_atoms,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pdb_list',
                   help='Text file with PDB IDs (whitespace or newline separated)')
    p.add_argument('--stor_dir', default='./many_structures/',
                   help='Directory for PDB files and per-structure metadata cache '
                        '(default: ./many_structures/)')
    p.add_argument('--selection', default='name CA',
                   help='MDAnalysis selection applied within conserved residues '
                        '(default: "name CA")')
    p.add_argument('--prefix', default='gpcr_pca',
                   help='Prefix for all output files (default: gpcr_pca)')
    p.add_argument('--bw_positions', default=None,
                   help='Explicit BW positions to use, bypassing the conservation filter. '
                        'Accepts a comma/space-separated string or a path to a text file.')
    p.add_argument('--bw_exclude', default=None,
                   help='BW positions to remove from the final list. Same format as '
                        '--bw_positions.')
    p.add_argument('--threshold', type=float, default=0.90,
                   help='BW conservation threshold 0-1 (default: 0.90)')
    p.add_argument('--species', default='Homo sapiens',
                   help='Only include structures from this species '
                        '(default: "Homo sapiens"). Pass "" or "all" to disable.')
    p.add_argument('--outlier_cutoff', type=float, default=100.0,
                   help='Structures with |PC1| > this value are dropped as outliers '
                        '(default: 100.0)')
    p.add_argument('--workers', type=int, default=16,
                   help='Parallel HTTP fetch workers (default: 16)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.stor_dir, exist_ok=True)

    with open(args.pdb_list, 'r') as fh:
        pdb_ids = sorted({tok.strip().upper()
                          for tok in fh.read().split()
                          if tok.strip()})
    print(f"Loaded {len(pdb_ids)} PDB IDs from '{args.pdb_list}'")

    tests = [Structure_Analyzer(pid, stor_dir=args.stor_dir) for pid in pdb_ids]
    print(f"Fetching data ({args.workers} parallel workers)...")
    fetch_results = fetch_all_parallel(tests, max_workers=args.workers)
    tests = [t for t in tests if fetch_results.get(t.pdb_code, (False,))[0]]
    print(f"  {len(tests)} / {len(pdb_ids)} structures fetched successfully")

    species_filter = args.species.strip()
    if species_filter and species_filter.lower() != 'all':
        before = len(tests)
        tests = [t for t in tests
                 if t.meta.get('structure', {}).get('species', '').lower()
                    == species_filter.lower()]
        print(f"  Species filter '{species_filter}': kept {len(tests)}, "
              f"dropped {before - len(tests)}")
    else:
        print("  Species filter: disabled")

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

    print("Building Ballesteros-Weinstein assignments...")
    bw_assignments = build_bw_assignments(tests)
    print(f"  Assignments built for {len(bw_assignments)} structures")

    if args.bw_positions is not None:
        try:
            conserved_bw = load_bw_positions(args.bw_positions)
        except ValueError as exc:
            sys.exit(f"ERROR in --bw_positions: {exc}")
        print(f"  Using {len(conserved_bw)} user-specified BW positions")
    else:
        conserved_bw = conservation_filter(
            bw_assignments, n_structures=len(tests), threshold=args.threshold)
        print(f"  {len(conserved_bw)} BW positions at >={args.threshold:.0%} conservation")

    if args.bw_exclude is not None:
        try:
            exclude_set = load_bw_exclude(args.bw_exclude)
        except ValueError as exc:
            sys.exit(f"ERROR in --bw_exclude: {exc}")
        before = len(conserved_bw)
        conserved_bw = [lbl for lbl in conserved_bw if lbl not in exclude_set]
        print(f"  Excluded {before - len(conserved_bw)} positions via --bw_exclude "
              f"({len(conserved_bw)} remaining)")

    resids_copopulated = build_resids_copopulated(tests, bw_assignments, conserved_bw)
    print(f"  {len(resids_copopulated)} structures have all conserved positions mapped")

    if not resids_copopulated:
        hint = ("Try lowering --threshold."
                if args.bw_positions is None
                else "Check that the supplied BW labels exist in this GPCR family.")
        sys.exit(f"ERROR: no structures survived the BW mapping filter. {hint}")

    test_by_code = {t.pdb_code: t for t in tests}
    selected = []
    for pdb_code, resids in resids_copopulated.items():
        test = test_by_code[pdb_code]
        ag = select_conserved_atoms(test.u_pdb, resids, args.selection)
        selected.append((pdb_code, ag))

    atom_counts = np.array([ag.n_atoms for _, ag in selected])
    values, counts = np.unique(atom_counts, return_counts=True)
    modal_count = int(values[counts.argmax()])
    print(f"  Atom count distribution: {dict(zip(values.tolist(), counts.tolist()))}")
    print(f"  Keeping {modal_count} atoms/structure "
          f"({(atom_counts == modal_count).sum()}/{len(selected)} pass)")

    selected = [(code, ag) for code, ag in selected if ag.n_atoms == modal_count]
    if not selected:
        sys.exit("ERROR: no structures survived the atom-count filter.")

    ref_code, ref_ag = selected[0]
    print(f"Aligning {len(selected)} structures to reference {ref_code}...")
    for _, ag in selected:
        align.alignto(ag, ref_ag, select='all', weights='mass')

    all_positions = np.array([ag.positions for _, ag in selected])
    vectorized = all_positions.reshape(len(selected), -1)

    print("Initial PCA for outlier detection...")
    pca_init = PCA()
    comp_init = pca_init.fit_transform(vectorized)
    not_outlier = np.abs(comp_init[:, 0]) <= args.outlier_cutoff
    n_dropped = int((~not_outlier).sum())
    if n_dropped:
        dropped = [code for (code, _), flag in zip(selected, ~not_outlier) if flag]
        print(f"  Dropping {n_dropped} outliers (|PC1| > {args.outlier_cutoff}): {dropped}")
    else:
        print("  No outliers found")

    selected_clean = [s for s, keep in zip(selected, not_outlier) if keep]
    vectorized_clean = vectorized[not_outlier]
    codes_retained = [code for code, _ in selected_clean]

    print(f"Running final PCA on {len(selected_clean)} structures...")
    pca = PCA()
    train_projections = pca.fit_transform(vectorized_clean)
    evr = pca.explained_variance_ratio_
    print(f"  PC1 {evr[0]:.1%}  PC2 {evr[1]:.1%}  PC3 {evr[2]:.1%}  "
          f"(cumulative top-3: {evr[:3].sum():.1%})")

    pca_path  = f"{args.prefix}_pca.joblib"
    ref_path  = f"{args.prefix}_ref.pdb"
    meta_path = f"{args.prefix}_meta.json"
    proj_path = f"{args.prefix}_train_projections.npy"

    joblib.dump(pca, pca_path)
    print(f"Saved PCA model       → {pca_path}")

    np.save(proj_path, train_projections)
    print(f"Saved train coords    → {proj_path}")

    ref_ag.write(ref_path)
    print(f"Saved reference atoms → {ref_path}")

    structure_meta = {
        code: test_by_code[code].meta.get('structure', {})
        for code in codes_retained
    }
    raw_states = [structure_meta[c].get('state') for c in codes_retained]
    state_counts = Counter(str(s) if s is not None else 'None' for s in raw_states)
    print("  State distribution in retained structures:")
    for state, n in sorted(state_counts.items()):
        print(f"    {state!r:30s}: {n}")

    meta = {
        'selection':      args.selection,
        'conserved_bw':   conserved_bw,
        'n_atoms':        modal_count,
        'ref_pdb_code':   ref_code,
        'ref_pdb_path':   ref_path,
        'codes_retained': codes_retained,
        'threshold':      args.threshold,
        'stor_dir':       args.stor_dir,
        'structure_meta': structure_meta,
    }
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved metadata        → {meta_path}")

    print(f"\nDone. {len(codes_retained)} structures in the PCA space.")
    print(f"To project a new trajectory:\n"
          f"  chimpss-project-pca "
          f"--pdb_code <CODE> --topology <top> --trajectory <traj> "
          f"--prefix {args.prefix}")


if __name__ == '__main__':
    main()
