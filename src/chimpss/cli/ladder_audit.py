"""CLI entry point for the temperature-ladder audit.

Console script: chimpss-ladder-audit

Reads the small arrays every FultonMarket block saves (temperatures, reduced
potentials, state indices) and reports, per neighbour pair, the thermodynamic
length, the Gaussian-predicted exchange acceptance and the acceptance actually
realised by the configurations that sat at the two rungs.  No simulation, no
positions; seconds for a whole run.

  # audit a run, print the table, write <run>/analysis/ladder_audit.json
  chimpss-ladder-audit runs/NTSR1_ML-301.hmr.rep3

  # restrict to blocks, write a TSV too
  chimpss-ladder-audit runs/NTSR1_ML-301.hmr.rep0 --blocks 120:161 --tsv

  # propose a ladder equidistant in thermodynamic length at 40 % acceptance
  # and check each proposed pair by reweighting the saved samples
  chimpss-ladder-audit runs/NTSR1_ML-301.hmr.rep0 --respace --target_acceptance 0.4 --virtual

The respaced ladder is written as a plain float64 ``.npy`` that
``chimpss-fultonmarket --ladder file --temperatures_file`` accepts.
"""
import argparse
import json
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help='Run directory containing saved_variables/')
    p.add_argument('--blocks', nargs='+', default=None,
                   help='Blocks to use: integers and/or half-open ranges such as 120:161. '
                        'Default: every block whose ladder matches the final one.')
    p.add_argument('--skip', type=int, default=1,
                   help='Leading iterations dropped from every block. Iteration 0 holds the '
                        'block\'s starting configurations, identical on every replica, so 1 is '
                        'the floor. Default 1.')
    p.add_argument('--targets', nargs='+', type=float, default=[0.35, 0.40, 0.50],
                   help='Target acceptances for which to report the rung count an '
                        'equidistant ladder needs. Default 0.35 0.40 0.50.')
    p.add_argument('--smooth', choices=['cv_linear', 'none'], default='cv_linear',
                   help='Smooth the per-rung sigma_H by a linear-in-T heat-capacity fit '
                        'before computing lengths (default) or use the raw estimate.')
    p.add_argument('--out', default=None,
                   help='Output prefix. Default <run_dir>/analysis/ladder_audit.')
    p.add_argument('--tsv', action='store_true', help='Also write <out>_pairs.tsv.')
    p.add_argument('--max_rows', type=int, default=None,
                   help='Print at most this many pair rows. Default: all.')

    g = p.add_argument_group('respacing')
    g.add_argument('--respace', action='store_true',
                   help='Write a ladder equidistant in thermodynamic length along the '
                        'measured profile to <out>_respaced.npy / .json.')
    g.add_argument('--target_acceptance', type=float, default=None,
                   help='Neighbour acceptance the respaced ladder aims for (sets the rung count).')
    g.add_argument('--n_rungs', type=int, default=None,
                   help='Fixed rung count for the respaced ladder, instead of a target acceptance.')
    g.add_argument('--T_min', type=float, default=None,
                   help='Lowest rung of the respaced ladder. Default: the run\'s own.')
    g.add_argument('--T_max', type=float, default=None,
                   help='Highest rung of the respaced ladder. Default: the run\'s own.')
    g.add_argument('--virtual', action='store_true',
                   help='Estimate each respaced pair\'s acceptance by reweighting the saved '
                        'samples, as a check before any simulation.')

    args = p.parse_args(argv)
    if args.respace and (args.target_acceptance is None) == (args.n_rungs is None):
        p.error('--respace needs exactly one of --target_acceptance or --n_rungs')
    if args.virtual and not args.respace:
        p.error('--virtual only applies with --respace')
    return args


def main(argv=None):
    args = parse_args(argv)

    import numpy as np

    from chimpss.analysis import ladder as L
    from chimpss.analysis.run_stats import write_json, write_tsv

    out = args.out or os.path.join(args.run_dir, 'analysis', 'ladder_audit')
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)

    try:
        data = L.load_run(args.run_dir, blocks=L.parse_blocks(args.blocks), skip=args.skip)
    except (FileNotFoundError, ValueError) as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)

    report = L.ladder_report(data, targets=args.targets, smooth=args.smooth)
    print(L.format_table(report, max_rows=args.max_rows))
    write_json(report, f'{out}.json')
    print(f'\nWrote {out}.json')
    if args.tsv:
        write_tsv(report['pairs'], f'{out}_pairs.tsv')
        print(f'Wrote {out}_pairs.tsv')

    if not args.respace:
        return

    T = data.temperatures
    sigma = np.array([r['sigma_fit'] for r in report['rungs']])
    try:
        new = L.respace(T, sigma, n_rungs=args.n_rungs, target_acceptance=args.target_acceptance,
                        T_min=args.T_min, T_max=args.T_max)
    except ValueError as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)

    record = {
        'source_run': os.path.abspath(args.run_dir),
        'blocks_used': data.blocks_used,
        'skip': data.skip,
        'smooth': args.smooth,
        'target_acceptance': args.target_acceptance,
        'n_rungs': int(new['n_rungs']),
        'dL_target': new['dL_target'],
        'dL_actual': new['dL_actual'],
        'L_total': new['L_total'],
        'T_min': new['T_min'], 'T_max': new['T_max'],
        'extrapolated': new['extrapolated'],
        'temperatures': new['temperatures'].tolist(),
        'pair_dL': new['pair_dL'].tolist(),
        'predicted_acceptance': new['predicted_acceptance'].tolist(),
        'cv_fit': report['summary']['cv_fit'],
    }
    if args.virtual:
        va = [L.virtual_pair_acceptance(a, b, T, data.Hk_blocks)
              for a, b in zip(new['temperatures'][:-1], new['temperatures'][1:])]
        record['virtual_acceptance'] = [v['acceptance'] for v in va]
        record['virtual_ess_fraction'] = [v['ess_fraction'] for v in va]

    np.save(f'{out}_respaced.npy', new['temperatures'])
    with open(f'{out}_respaced.json', 'w') as fh:
        json.dump(record, fh, indent=2)

    print(f'\nRespaced ladder: {new["n_rungs"]} rungs, {new["T_min"]:.2f}-{new["T_max"]:.2f} K, '
          f'dL {new["dL_actual"]:.3f} per pair (predicted acceptance '
          f'{float(L.predicted_acceptance(new["dL_actual"])):.3f})'
          + ('  [extrapolated beyond the sampled range]' if new['extrapolated'] else ''))
    if args.virtual:
        v = np.array(record['virtual_acceptance'])
        e = np.array(record['virtual_ess_fraction'])
        print(f'  reweighted acceptance of the proposed pairs: min/mean/max '
              f'{v.min():.3f}/{v.mean():.3f}/{v.max():.3f}   min ESS fraction {e.min():.2f}')
    print(f'Wrote {out}_respaced.npy and {out}_respaced.json')


if __name__ == '__main__':
    main()
