"""CLI entry point for thermodynamic trailblazing of a temperature ladder.

Console script: chimpss-trailblaze

Two ways to get a ladder whose neighbour pairs are equidistant in
thermodynamic length:

  # walk it: short simulations rung by rung on one GPU, ~1 h for a campaign
  # system; writes <output_dir>/trailblaze/{trailblaze.json,temperatures.npy,seeds/}
  # which chimpss-fultonmarket --ladder trailblaze then finds and reuses
  chimpss-trailblaze walk equil.pdb system.xml runs/NTSR1_rep4 --input_state state.xml \\
      --T_min 310 --T_max 367 --timestep 3.5 --target_acc 0.4

  # walk it from the top instead (--direction down), or both ways (--direction both): the up
  # walk, then the down walk seeded from its hottest rung, reconciled into one ladder placed by
  # the integral of the merged metric and seeded from the down walk's annealed configurations
  chimpss-trailblaze walk equil.pdb system.xml runs/NTSR1_rep5 --input_state state.xml \\
      --T_min 310 --T_max 367 --timestep 3.5 --direction both

  # reconcile two existing walks (e.g. one run's up walk with another's down walk)
  chimpss-trailblaze reconcile runs/NTSR1_rep5/trailblaze --up_dir runs/NTSR1_rep4/trailblaze

  # or respace it from a finished run of the same system, no simulation
  chimpss-trailblaze from-run runs/NTSR1_ML-301.hmr.rep0 --target_acc 0.4 --out ladders/NTSR1

The walk is resumable: rerunning the same command continues from the last
accepted rung, and a complete record returns immediately.
"""
import argparse
import os
import sys

from chimpss.cli.trailblaze_args import add_trailblaze_arguments, trailblaze_settings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)

    w = sub.add_parser('walk', help='Walk the ladder with short simulations')
    w.add_argument('input_pdb', type=str, help='Path to PDB file with starting coordinates')
    w.add_argument('input_system', type=str, help='Path to XML file with the serialized system')
    w.add_argument('output_dir', type=str,
                   help='Run directory; the ladder is cached under <output_dir>/trailblaze/')
    w.add_argument('--input_state', required=True, type=str,
                   help='Path to the XML state the walk starts from (the burn-in output)')
    w.add_argument('--T_min', default=300, type=float, help='Lowest rung (K). Default 300.')
    w.add_argument('--T_max', default=367, type=float, help='Highest rung (K). Default 367.')
    w.add_argument('--timestep', default=2.0, type=float, help='Integration timestep (fs). Default 2.0.')
    add_trailblaze_arguments(w, direction_flags=('--direction', '--tb_direction'))
    w.add_argument('--protein-name', default=None, type=str)
    w.add_argument('--ligand-name', default=None, type=str)

    r = sub.add_parser('reconcile', help='Merge an existing up walk and down walk into one ladder')
    r.add_argument('trailblaze_dir', type=str,
                   help='Directory that receives trailblaze.json, temperatures.npy, seeds/ and reconcile.json '
                        '(the trailblaze/ directory of the run that will use the ladder)')
    r.add_argument('--up_dir', default=None, type=str, help='The up walk. Default <trailblaze_dir>/up.')
    r.add_argument('--down_dir', default=None, type=str, help='The down walk. Default <trailblaze_dir>/down.')
    r.add_argument('--target_acc', default=None, type=float,
                   help="Neighbour acceptance of the reconciled ladder. Default: the walks' own.")

    f = sub.add_parser('from-run', help='Respace the ladder from a finished run, no simulation')
    f.add_argument('run_dir', type=str, help='Run directory containing saved_variables/')
    f.add_argument('--out', required=True, type=str,
                   help='Output prefix: writes <out>.npy (the ladder) and <out>.json')
    f.add_argument('--target_acc', default=0.40, type=float, help='Default 0.40.')
    f.add_argument('--blocks', nargs='+', default=None,
                   help='Blocks to use: integers and/or ranges such as 120:161. Default: all.')
    f.add_argument('--skip', default=1, type=int,
                   help='Leading iterations dropped from every block. Default 1.')
    f.add_argument('--T_min', default=None, type=float, help="Default: the run's own.")
    f.add_argument('--T_max', default=None, type=float, help="Default: the run's own.")
    f.add_argument('--no_virtual', action='store_true',
                   help='Skip the reweighted check of each proposed pair.')

    return p.parse_args(argv)


#: The same settings dict chimpss-fultonmarket builds; one definition for both entry points.
walk_settings = trailblaze_settings


def _walk(args):
    from chimpss.fultonmarket import FultonMarket as FM

    market = FM(input_pdb=args.input_pdb, input_system=args.input_system, input_state=args.input_state,
                T_min=args.T_min, T_max=args.T_max, protein_name=args.protein_name,
                ligand_name=args.ligand_name, ladder='trailblaze', trailblaze_kwargs=walk_settings(args))
    temps, meta = market.trailblaze(args.output_dir, dt=args.timestep)
    extra = ''
    if meta.get('direction') == 'both':
        extra = f" (reconciled from {meta.get('K_up')} up and {meta.get('K_down')} down rungs)"
    print(f"\nLadder ({meta.get('direction', 'up')}): {meta['n_rungs']} rungs{extra}, {temps[0]:.2f}-{temps[-1]:.2f} K, "
          f"total thermodynamic length {meta['L_total']:.2f}, neighbour acceptance min {meta['acc_min']:.3f} "
          f"median {meta['acc_median']:.3f}")
    print(f"Cached under {market.trailblaze_dir}; chimpss-fultonmarket --ladder trailblaze on the same "
          f"output_dir will reuse it.")


def _reconcile(args):
    from chimpss.fultonmarket.trailblaze import DOWN_DIR, RECORD_NAME, UP_DIR, TrailblazeRecord, reconcile

    up_dir = args.up_dir or os.path.join(args.trailblaze_dir, UP_DIR)
    down_dir = args.down_dir or os.path.join(args.trailblaze_dir, DOWN_DIR)
    records = {}
    for name, d in (('up', up_dir), ('down', down_dir)):
        rec = TrailblazeRecord.load_if_complete(os.path.join(d, RECORD_NAME))
        if rec is None:
            print(f'ERROR: no complete {name} walk in {d} (looked for {RECORD_NAME})')
            sys.exit(1)
        if rec.direction != name:
            print(f'ERROR: the record in {d} is a {rec.direction} walk, not {name}')
            sys.exit(1)
        records[name] = rec
    try:
        rec, diag = reconcile(records['up'], records['down'], args.trailblaze_dir, up_dir=up_dir, down_dir=down_dir,
                              target_acc=args.target_acc)
    except ValueError as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)
    h = diag['hysteresis']
    lad = diag['ladders']
    print(f"\nReconciled ladder: {diag['K_consensus']} rungs (up walk {diag['K_up']}, down walk {diag['K_down']}), "
          f"{rec.temperatures[0]:.2f}-{rec.temperatures[-1]:.2f} K, dL {lad['consensus']['dL_mean']:.3f} per pair")
    print(f"  fitted-metric hysteresis: mean {100 * h['sigma_fit_rel_mean']:+.1f} %, max |{100 * h['sigma_fit_rel_max']:.1f}| % "
          f"(per-rung sigma noise ~{100 * h['expected_sigma_noise']:.0f} %); mean-enthalpy lag max "
          f"{h['mean_H_over_sigma_max']:.2f} sigma; L_total up {diag['L_total_up_fit']:.2f} vs down {diag['L_total_down_fit']:.2f}")
    for name in ('up', 'down', 'consensus'):
        d = lad[name]
        meas = '' if d.get('acc_min') is None else f"; measured acceptance min {d['acc_min']:.3f} median {d['acc_median']:.3f}"
        print(f"  {name:9s} {d['n_rungs']:3d} rungs: dL CV {d['dL_cv']:.3f}, predicted acceptance "
              f"{d['acc_pred_min']:.3f}-{d['acc_pred_max']:.3f} on the merged metric{meas}")
    print(f"Wrote {os.path.join(args.trailblaze_dir, RECORD_NAME)}, temperatures.npy, seeds/ (from the down walk) "
          f"and reconcile.json; chimpss-fultonmarket --ladder trailblaze on that run will use them.")


def _from_run(args):
    import json

    import numpy as np

    from chimpss.analysis import ladder as L

    try:
        data = L.load_run(args.run_dir, blocks=L.parse_blocks(args.blocks), skip=args.skip)
    except (FileNotFoundError, ValueError) as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)
    report = L.ladder_report(data, targets=(args.target_acc,))
    sigma = np.array([r['sigma_fit'] for r in report['rungs']])
    try:
        new = L.respace(data.temperatures, sigma, target_acceptance=args.target_acc,
                        T_min=args.T_min, T_max=args.T_max)
    except ValueError as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)

    record = {
        'ladder': 'file', 'source': 'from-run', 'source_run': os.path.abspath(args.run_dir),
        'blocks_used': data.blocks_used, 'skip': data.skip, 'target_acc': args.target_acc,
        'min_acc': None, 'n_rungs': int(new['n_rungs']), 'dL_target': new['dL_target'],
        'dL_actual': new['dL_actual'], 'L_total': new['L_total'], 'extrapolated': new['extrapolated'],
        'temperatures': new['temperatures'].tolist(), 'pair_dL': new['pair_dL'].tolist(),
        'predicted_acceptance': new['predicted_acceptance'].tolist(),
        'source_summary': {k: report['summary'][k] for k in
                           ('n_rungs', 'samples_per_rung', 'L_total', 'A_meas_mean', 'corr_pred_meas')},
    }
    if not args.no_virtual:
        va = [L.virtual_pair_acceptance(a, b, data.temperatures, data.Hk_blocks)
              for a, b in zip(new['temperatures'][:-1], new['temperatures'][1:])]
        record['virtual_acceptance'] = [v['acceptance'] for v in va]
        record['virtual_ess_fraction'] = [v['ess_fraction'] for v in va]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{args.out}.npy', new['temperatures'])
    with open(f'{args.out}.json', 'w') as fh:
        json.dump(record, fh, indent=2)

    print(f"Ladder: {new['n_rungs']} rungs, {new['T_min']:.2f}-{new['T_max']:.2f} K, dL {new['dL_actual']:.3f} "
          f"per pair (predicted acceptance {float(L.predicted_acceptance(new['dL_actual'])):.3f})"
          + ('  [extrapolated beyond the sampled range]' if new['extrapolated'] else ''))
    if not args.no_virtual:
        v = np.array(record['virtual_acceptance'])
        print(f'  reweighted acceptance of the proposed pairs: min/mean/max {v.min():.3f}/{v.mean():.3f}/{v.max():.3f}')
    print(f'Wrote {args.out}.npy (use with chimpss-fultonmarket --temperatures_file) and {args.out}.json')


def main(argv=None):
    args = parse_args(argv)
    if args.mode == 'walk':
        _walk(args)
    elif args.mode == 'reconcile':
        _reconcile(args)
    else:
        _from_run(args)


if __name__ == '__main__':
    main()
