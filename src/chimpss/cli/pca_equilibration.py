"""
CLI entry point for equilibration detection in an external pan-GPCR PCA basis.

Projects the time-ordered trajectory of one thermodynamic state onto a PCA
fitted on experimental GPCR structures, runs the standard detectors on each
principal component, and tabulates the answer against the observables the
production criterion already uses.

The comparison is controlled: the self-fitted basis is rebuilt from the *same
atoms, frames and superposition* as the external one, so the difference between
their ``t0`` estimates isolates the effect of fitting the axes on the very data
being judged, and nothing else.

Console script: chimpss-pca-equil

Example:
  chimpss-pca-equil runs/NTSR1_ML-301.hmr.rep0 \\
      --pdb_code 6ZA8 --prefix models/pangpcr \\
      --topology runs/NTSR1_ML-301.hmr.rep0/saved_variables/10/resampled_top.pdb
"""

import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help='Production run directory (contains saved_variables/).')
    p.add_argument('--pdb_code', required=True,
                   help='PDB code of the parent crystal structure, for the '
                        'GPCRdb Ballesteros-Weinstein lookup (e.g. 6ZA8).')
    p.add_argument('--topology', required=True,
                   help='Topology PDB matching the atom count of positions.npy.')
    p.add_argument('--prefix', default='gpcr_pca',
                   help='Prefix of the model from chimpss-generate-pca.')
    p.add_argument('--stor_dir', default='./many_structures/',
                   help='Cache directory for GPCRdb metadata.')
    p.add_argument('--state', type=int, default=0,
                   help='Thermodynamic state to follow (0 = lowest rung).')
    p.add_argument('--skip', type=int, default=10,
                   help='Iterations discarded at the start of each block.')
    p.add_argument('--chain_index', type=int, default=0,
                   help='Index of the receptor chain in the topology.')
    p.add_argument('--detectors', nargs='*',
                   default=['max_neff', 'mser', 'mser_sqrt', 'mser_w50'],
                   help='Detectors to run on every observable.')
    p.add_argument('--n_pcs', type=int, default=None,
                   help='Components to combine. Default: enough to reach 90%% '
                        'of the training variance.')
    p.add_argument('--energy_pdb', default=None,
                   help='Topology for the energy and structural series, so the '
                        'incumbent observables appear in the same table.')
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--json_out', default=None)
    return p.parse_args()


def _fmt_pct(v):
    return '-' if v is None else f'{100 * v:>6.1f}%'


def main():
    args = parse_args()

    from chimpss.analysis.gpcr_pca import fetch_naming_info
    from chimpss.analysis.pca_equilibration import basis_comparison, load_model

    try:
        model = load_model(args.prefix)
    except FileNotFoundError as exc:
        sys.exit(f'ERROR: {exc}')

    print('=' * 74)
    print(f'  Pan-GPCR PCA equilibration :: {os.path.basename(args.run_dir.rstrip("/"))}')
    print('=' * 74)
    print(f'  basis            : {args.prefix}  '
          f'({len(model.meta["codes_retained"])} structures, '
          f'{len(model.conserved_bw)} BW positions, '
          f'selection {model.selection!r})')
    evr = model.explained_variance_ratio
    print(f'  training variance: PC1 {evr[0]:.1%}  PC2 {evr[1]:.1%}  '
          f'PC3 {evr[2]:.1%}  (top-3 {evr[:3].sum():.1%})')

    os.makedirs(args.stor_dir, exist_ok=True)
    print(f'\n  GPCRdb numbering for {args.pdb_code.upper()}...')
    try:
        naming_info, _ = fetch_naming_info(args.pdb_code, args.stor_dir)
    except Exception as exc:
        sys.exit(f'ERROR: could not fetch GPCRdb data for {args.pdb_code}: {exc}')

    report = basis_comparison(
        args.run_dir, args.topology, model, naming_info, args.prefix,
        state=args.state, skip=args.skip, chain_index=args.chain_index,
        detectors=args.detectors, n_pcs=args.n_pcs,
        energy_pdb=args.energy_pdb, cache=not args.no_cache)

    m = report.get('mapping', {})
    print('\n  Sequence mapping (GPCRdb -> trajectory)')
    print(f'    aligned residues : {m.get("n_aligned")} / {m.get("n_traj")}  '
          f'identity {m.get("identity", 0):.1%}')
    offsets = m.get('offsets', {})
    print(f'    resid offsets    : {offsets}')
    if len(offsets) > 1:
        print(f'    -> {len(offsets)} offsets: the construct carries an internal '
              f'deletion. A single-offset mapping would mis-assign every')
        print( '       position downstream of it.')
    print(f'    BW positions     : {m.get("n_resolved")} / {m.get("n_positions")} resolved')
    if m.get('missing'):
        for lbl, _, reason in m['missing'][:5]:
            print(f'      [{lbl}] {reason}')
        if len(m['missing']) > 5:
            print(f'      ... and {len(m["missing"]) - 5} more')
    print(f'    frames           : {m.get("n_frames")} '
          f'(state {args.state}, skip {args.skip})')

    obs = report.get('observables', {})
    if not obs:
        sys.exit('ERROR: no frames collected.')

    print('\n  t0 as a fraction of the series, by observable and detector')
    dets = args.detectors
    header = '    ' + f'{"observable":<14}' + ''.join(f'{d:>12}' for d in dets)
    print(header)
    print('    ' + '-' * (len(header) - 4))
    order = [k for k in ('energy', 'structural', 'self_pca', 'pangpcr') if k in obs]
    for name in order:
        cells = []
        for d in dets:
            row = obs[name]['by_detector'].get(d, {})
            cells.append(_fmt_pct(row.get('discard')))
        print(f'    {name:<14}' + ''.join(f'{c:>12}' for c in cells))

    if 'pangpcr' in obs and 'self_pca' in obs:
        print('\n  Circularity control (identical atoms, frames and superposition)')
        for d in dets:
            a = obs['pangpcr']['by_detector'].get(d, {}).get('discard')
            b = obs['self_pca']['by_detector'].get(d, {}).get('discard')
            if a is None or b is None:
                continue
            print(f'    {d:<12} external {100 * a:>6.1f}%   '
                  f'self-fitted {100 * b:>6.1f}%   '
                  f'difference {100 * (b - a):>+6.1f} pts')

    if 'pangpcr' in obs:
        d0 = args.detectors[0]
        rows = obs['pangpcr']['by_detector'].get(d0, {}).get('per_pc', [])
        if rows:
            print(f'\n  Per-component detail, external basis, detector {d0!r}')
            print(f'    {"PC":>3} {"var":>7} {"t0":>7} {"discard":>9} {"traj sd":>9}')
            for r in rows[:10]:
                t0 = '-' if r['t0'] is None else f'{r["t0"]:>7d}'
                print(f'    {r["pc"]:>3} {r["evr"]:>6.1%} {t0} '
                      f'{_fmt_pct(r["discard"]):>9} {r["spread"]:>9.3f}')

    if 'spread_ratio' in report:
        sr = report['spread_ratio']
        print('\n  Trajectory spread as a fraction of the training spread')
        print('    ' + '  '.join(f'PC{k + 1} {100 * v:.1f}%'
                                 for k, v in enumerate(sr) if np.isfinite(v)))
        print('    A small fraction means the run occupies a sliver of pan-GPCR')
        print('    space, so these axes are dominated by differences between')
        print('    receptors rather than by anything this simulation does.')

    tr = report.get('training', {})
    if 'active_inactive_cohens_d' in tr:
        d = tr['active_inactive_cohens_d']
        print("\n  Training-set state separation (Cohen's d)")
        print(f"    PC1 {d[0]:.2f}   PC2 {d[1]:.2f}")

    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f'\n  report written to {args.json_out}')
    print('=' * 74)


if __name__ == '__main__':
    main()
