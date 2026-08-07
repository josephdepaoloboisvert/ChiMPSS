"""CLI entry point for class-A GPCR activation-motif measurements.

Console script: chimpss-motifs

Measure a simulation ensemble::

  chimpss-motifs trajectory --pdb_code 6ZA8 \\
      --topology NTSR1_ML-301.equil.pdb \\
      --trajectory saved_variables/24/resampled_trj.dcd \\
      --out motifs/NTSR1_ML-301.hmr.rep0

Measure the PCA training set, producing the active/inactive population means
the ensembles are compared against::

  chimpss-motifs reference --pca_meta gpcr_pca_meta.json \\
      --stor_dir ./many_structures/ --out motifs/reference_set
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)

    t = sub.add_parser('trajectory', help='Measure one trajectory or structure')
    t.add_argument('--pdb_code', required=True,
                   help='PDB code of the parent structure, for GPCRdb BW lookup')
    t.add_argument('--topology', required=True)
    t.add_argument('--trajectory', default=None,
                   help='Omit to measure a single frame from --topology')
    t.add_argument('--chain', default=None,
                   help='Chain ID of the receptor. Omit for single-chain systems.')
    t.add_argument('--stor_dir', default='./many_structures/',
                   help='GPCRdb metadata cache (default: ./many_structures/)')
    t.add_argument('--resid_offset', type=int, default=0)
    t.add_argument('--no_auto_offset', action='store_true')
    t.add_argument('--out', required=True,
                   help='Output prefix; writes <out>.npy and <out>.json')

    r = sub.add_parser('reference', help='Measure the PCA training structures')
    r.add_argument('--pca_meta', required=True,
                   help='<prefix>_meta.json written by chimpss-generate-pca')
    r.add_argument('--stor_dir', default='./many_structures/')
    r.add_argument('--out', required=True,
                   help='Output prefix; writes <out>.tsv and <out>_by_state.json')

    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np

    from chimpss.analysis import motifs as M
    from chimpss.analysis.run_stats import write_json, write_tsv

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.mode == 'trajectory':
        arr, meta = M.measure_trajectory(
            pdb_code=args.pdb_code,
            topology=args.topology,
            trajectory=args.trajectory,
            chain=args.chain,
            stor_dir=args.stor_dir,
            resid_offset=args.resid_offset,
            auto_offset=not args.no_auto_offset)

        meta['summary'] = M.summarize(arr)

        np.save(f'{args.out}.npy', arr)
        write_json(meta, f'{args.out}.json')

        print(f'\nSaved {arr.shape[0]} frames x {arr.shape[1]} motifs '
              f'-> {args.out}.npy')
        for col, s in meta['summary'].items():
            if s['n']:
                print(f"  {col:18s} {s['mean']:8.3f} +/- {s['std']:.3f}  (n={s['n']})")
            else:
                print(f"  {col:18s} not measurable in this system")

        if meta['all_nan_columns']:
            print('\nColumns with no measurable value (expected for '
                  'receptors lacking the motif residue):')
            for col in meta['all_nan_columns']:
                print(f'  {col}')
        return

    # reference mode
    if not os.path.exists(args.pca_meta):
        sys.exit(f'ERROR: {args.pca_meta} not found. '
                 f'Run chimpss-generate-pca first.')

    rows, by_state = M.measure_reference_set(args.pca_meta,
                                             stor_dir=args.stor_dir)
    if not rows:
        sys.exit('ERROR: no reference structures could be measured. '
                 'Check that --stor_dir holds the cached PDB files.')

    columns = ['pdb_code', 'state'] + list(M.MOTIF_COLUMNS)
    write_tsv(rows, f'{args.out}.tsv', columns=columns)
    write_json(by_state, f'{args.out}_by_state.json')

    print(f'\nMeasured {len(rows)} reference structures -> {args.out}.tsv')
    for state, stats in by_state.items():
        print(f"\n  {state} (n={stats['n']})")
        for col in M.MOTIF_COLUMNS:
            s = stats[col]
            if s['n']:
                print(f"    {col:18s} {s['mean']:8.3f} +/- {s['std']:.3f}  (n={s['n']})")


if __name__ == '__main__':
    main()
