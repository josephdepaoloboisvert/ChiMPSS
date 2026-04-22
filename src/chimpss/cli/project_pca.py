"""
CLI entry point for projecting an MD trajectory onto a pre-computed GPCR PCA space.

Console script: chimpss-project-pca

Usage examples:
  chimpss-project-pca --pdb_code 6CMO --topology system.pdb --trajectory md.xtc \\
      --chain A --prefix gpcr_pca

  chimpss-project-pca --pdb_code 6CMO --topology structure.pdb --prefix gpcr_pca

  chimpss-project-pca --pdb_code 6CMO --topology system.pdb --trajectory md.xtc \\
      --resid_offset -33 --prefix gpcr_pca
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np
import MDAnalysis as mda
import joblib

from chimpss.analysis.gpcr_pca import (
    Structure_Analyzer,
    auto_detect_resid_offset,
    fetch_bw_map,
    map_conserved_resids,
    build_mobile_ag,
    _imputation_setup,
    project_trajectory,
    _verbose_mapping_report,
    _verbose_loadings_report,
    _verbose_training_comparison,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pdb_code', required=True,
                   help='PDB code of the parent crystal structure for BW numbering lookup')
    p.add_argument('--topology', required=True,
                   help='Topology file for the MD system (PDB, GRO, PSF, …)')
    p.add_argument('--trajectory', default=None,
                   help='Trajectory file (XTC, DCD, …). Omit to project a single frame.')
    p.add_argument('--chain', default=None,
                   help='Chain ID of the GPCR in the trajectory. '
                        'Omit for single-chain systems.')
    p.add_argument('--prefix', default='gpcr_pca',
                   help='Prefix of the PCA files from chimpss-generate-pca (default: gpcr_pca)')
    p.add_argument('--stor_dir', default='./many_structures/',
                   help='Cache directory for GPCRdb metadata (default: ./many_structures/)')
    p.add_argument('--resid_offset', type=int, default=0,
                   help='Integer added to GPCRdb sequence numbers to obtain resids in '
                        'your topology (default: 0)')
    p.add_argument('--out', default=None,
                   help='Output .npy file (default: <pdb_code>_projections.npy)')
    p.add_argument('--no_auto_offset', action='store_true',
                   help='Disable automatic resid-offset detection.')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print detailed diagnostics.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.out is None:
        args.out = f"{args.pdb_code.upper()}_projections.npy"

    pca_path  = f"{args.prefix}_pca.joblib"
    meta_path = f"{args.prefix}_meta.json"
    ref_path  = f"{args.prefix}_ref.pdb"

    for path in (pca_path, meta_path, ref_path):
        if not os.path.exists(path):
            sys.exit(f"ERROR: '{path}' not found. "
                     f"Run chimpss-generate-pca --prefix {args.prefix} first.")

    pca = joblib.load(pca_path)
    with open(meta_path, 'r') as fh:
        meta = json.load(fh)

    selection    = meta['selection']
    conserved_bw = meta['conserved_bw']
    expected_n   = meta['n_atoms']

    pca_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(pca_path))
    print(f"PCA model : {os.path.abspath(pca_path)}  "
          f"(modified {pca_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"Loaded PCA: {pca.n_components_} components, "
          f"trained on {len(meta['codes_retained'])} structures")
    print(f"Selection : '{selection}'  |  Expected atoms: {expected_n}")

    ref_u  = mda.Universe(ref_path)
    ref_ag = ref_u.select_atoms('all')
    if ref_ag.n_atoms != expected_n:
        sys.exit(
            f"ERROR: reference file '{ref_path}' has {ref_ag.n_atoms} atoms "
            f"but metadata says {expected_n}.  Regenerate with --selection '{selection}'.")

    os.makedirs(args.stor_dir, exist_ok=True)
    analyzer, bw_map = fetch_bw_map(args.pdb_code, args.stor_dir)

    if args.trajectory:
        u = mda.Universe(args.topology, args.trajectory)
    else:
        u = mda.Universe(args.topology)
    print(f"Universe  : {len(u.trajectory)} frame(s), {u.atoms.n_atoms} atoms")
    if args.chain:
        print(f"Chain     : {args.chain}")

    if args.no_auto_offset:
        resid_offset = args.resid_offset
        if resid_offset:
            print(f"Resid offset: {resid_offset:+d}  (auto-detection disabled)")
    else:
        resid_offset = auto_detect_resid_offset(
            u, analyzer, chain=args.chain, user_offset=args.resid_offset)
        if resid_offset:
            print(f"Resid offset: {resid_offset:+d}")

    print("Mapping conserved BW positions to trajectory residues...")
    resids, missing_info = map_conserved_resids(
        bw_map, conserved_bw, u,
        chain=args.chain,
        resid_offset=resid_offset)

    imputation = None

    if missing_info:
        n_missing = len(missing_info)
        if n_missing > 2:
            lines = "\n".join(f"  [{lbl}] {reason}"
                              for lbl, _, reason in missing_info)

            chain_sel = f"chainid {args.chain} and " if args.chain else ""
            bb_resids = sorted(
                u.select_atoms(f"{chain_sel}backbone").residues.resids)
            traj_min, traj_max = (bb_resids[0], bb_resids[-1]) if bb_resids else (None, None)

            missing_seqnums = []
            for _, _, reason in missing_info:
                if 'seqnum' in reason:
                    try:
                        missing_seqnums.append(
                            int(reason.split('seqnum')[1].split('→')[0].strip()))
                    except (ValueError, IndexError):
                        pass

            diag = ""
            if bb_resids and missing_seqnums:
                missing_min = min(missing_seqnums) + resid_offset
                missing_max = max(missing_seqnums) + resid_offset
                diag = (f"\nDiagnostic:"
                        f"\n  Trajectory backbone resids : {traj_min}-{traj_max} "
                        f"({len(bb_resids)} residues)")
                if missing_min > traj_max:
                    diag += (
                        f"\n  Missing resids: {missing_min}-{missing_max}"
                        f"\n  All missing resids are above the trajectory maximum."
                        f"\n  The topology may be C-terminally truncated.")
                elif missing_max < traj_min:
                    diag += (
                        f"\n  Missing resids: {missing_min}-{missing_max}"
                        f"\n  All missing resids are below the trajectory minimum."
                        f"\n  The topology may be N-terminally truncated.")
                else:
                    offset_guess = traj_min - (min(missing_seqnums)
                                               if missing_seqnums else traj_min)
                    diag += (
                        f"\n  Missing resids: {missing_min}-{missing_max}"
                        f"\n  Missing resids fall within the trajectory resid range."
                        f"\n  This looks like a numbering mismatch."
                        f"\n  Try --resid_offset {offset_guess:+d}")

            raise Exception(
                f"{n_missing} conserved BW positions are missing — too many for "
                f"mean imputation (limit: 2).\n{lines}{diag}")

        print(f"\n  WARNING: {n_missing} residue(s) missing — "
              f"will be mean-imputed (contribute zero to all PCs):")
        for lbl, _, reason in missing_info:
            print(f"    [{lbl}] {reason}")

        imputation = _imputation_setup(missing_info, conserved_bw, expected_n, pca)
        print(f"  PC1 loading norm from imputed features: "
              f"{imputation['pc1_imputed_frac']:.1%}")
        print(f"  PC2 loading norm from imputed features: "
              f"{imputation['pc2_imputed_frac']:.1%}")
        if (imputation['pc1_imputed_frac'] > 0.10
                or imputation['pc2_imputed_frac'] > 0.10):
            print("  WARNING: imputed features carry >10% of a PC's loading — "
                  "interpret projections with caution.")
        print()
    else:
        print(f"  All {len(conserved_bw)} conserved positions mapped successfully")

    if args.verbose:
        _verbose_mapping_report(bw_map, conserved_bw, resids, u,
                                args.chain, resid_offset)

    mobile_ag = build_mobile_ag(u, resids, selection, chain=args.chain)
    expected_mobile = (expected_n if imputation is None
                       else len(imputation['present_feat_idx']) // 3)
    if mobile_ag.n_atoms != expected_mobile:
        sys.exit(
            f"ERROR: selected {mobile_ag.n_atoms} atoms, expected {expected_mobile}.\n"
            f"  Check --chain (currently: {args.chain})\n"
            f"  Training selection was '{selection}'")
    print(f"Selected  : {mobile_ag.n_atoms} atoms "
          f"({len(conserved_bw) - len(missing_info)} residues present"
          + (f", {len(missing_info)} imputed" if missing_info else "")
          + ")")

    if args.verbose:
        _verbose_loadings_report(pca, conserved_bw, expected_n)

    projections = project_trajectory(u, mobile_ag, ref_ag, pca,
                                     imputation=imputation)

    np.save(args.out, projections)
    print(f"\nSaved projections → {args.out}  shape: {projections.shape}")
    print(f"  PC1  [{projections[:, 0].min():.2f}, {projections[:, 0].max():.2f}]")
    print(f"  PC2  [{projections[:, 1].min():.2f}, {projections[:, 1].max():.2f}]")
    if projections.shape[1] > 2:
        print(f"  PC3  [{projections[:, 2].min():.2f}, {projections[:, 2].max():.2f}]")

    if args.verbose:
        _verbose_training_comparison(projections, meta, args.prefix)


if __name__ == '__main__':
    main()
