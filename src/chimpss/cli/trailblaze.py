"""CLI entry point for thermodynamic trailblazing of a temperature ladder.

Console script: chimpss-trailblaze

Two ways to get a ladder whose neighbour pairs are equidistant in
thermodynamic length:

  # walk it: short simulations rung by rung on one GPU, ~1 h for a campaign
  # system; writes <output_dir>/trailblaze/{trailblaze.json,temperatures.npy,seeds/}
  # which chimpss-fultonmarket --ladder trailblaze then finds and reuses
  chimpss-trailblaze walk equil.pdb system.xml runs/NTSR1_rep4 --input_state state.xml \\
      --T_min 310 --T_max 367 --timestep 3.5 --target_acc 0.4

  # or respace it from a finished run of the same system, no simulation
  chimpss-trailblaze from-run runs/NTSR1_ML-301.hmr.rep0 --target_acc 0.4 --out ladders/NTSR1

The walk is resumable: rerunning the same command continues from the last
accepted rung, and a complete record returns immediately.
"""
import argparse
import os
import sys


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
    w.add_argument('--target_acc', default=0.40, type=float,
                   help='Neighbour acceptance aimed for; dL* = 2 erfcinv(A). Default 0.40.')
    w.add_argument('--min_acc', default=0.30, type=float,
                   help='Reject a proposed rung and shrink the step below this. Default 0.30.')
    w.add_argument('--max_acc', default=0.75, type=float,
                   help='Reject a proposed rung and stretch the step above this. Default 0.75.')
    w.add_argument('--tb_first_equil_ps', default=200.0, type=float,
                   help='Equilibration of the starting configuration at T_min (ps). Default 200.')
    w.add_argument('--tb_equil_ps', default=20.0, type=float,
                   help='Equilibration at each new rung (ps). Default 20.')
    w.add_argument('--tb_sample_ps', default=100.0, type=float,
                   help='Enthalpy sampling per rung (ps), extended until --tb_neff_min effective '
                        'samples or --tb_max_sample_ps. Default 100.')
    w.add_argument('--tb_sample_interval_ps', default=1.0, type=float, help='Default 1.')
    w.add_argument('--tb_max_sample_ps', default=400.0, type=float, help='Default 400.')
    w.add_argument('--tb_neff_min', default=50, type=int, help='Default 50.')
    w.add_argument('--tb_max_rungs', default=400, type=int, help='Default 400.')
    w.add_argument('--tb_sigma_window', default=5, type=int,
                   help='Rungs over which sigma_H is pooled for the step (1 = each rung alone). Default 5.')
    w.add_argument('--tb_sigma_source', default='pooled', choices=['pooled', 'slope'],
                   help="What the step uses: 'pooled' (default) the pooled enthalpy fluctuation; "
                        "'slope' the fluctuation-dissipation value from the enthalpy slope d<H>/dT "
                        "over the trailing rungs, which tracks the production metric more closely "
                        "when single-trajectory windows miss slow fluctuations.")
    w.add_argument('--tb_force', action='store_true',
                   help='Resume a record written with different settings.')
    w.add_argument('--protein-name', default=None, type=str)
    w.add_argument('--ligand-name', default=None, type=str)

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


def walk_settings(args):
    return dict(target_acc=args.target_acc, min_acc=args.min_acc, max_acc=args.max_acc,
                first_equil_ps=args.tb_first_equil_ps, equil_ps=args.tb_equil_ps,
                sample_ps=args.tb_sample_ps, sample_interval_ps=args.tb_sample_interval_ps,
                max_sample_ps=args.tb_max_sample_ps, neff_min=args.tb_neff_min,
                max_rungs=args.tb_max_rungs, sigma_window=args.tb_sigma_window, sigma_source=args.tb_sigma_source,
                force=args.tb_force)


def _walk(args):
    from chimpss.fultonmarket import FultonMarket as FM

    market = FM(input_pdb=args.input_pdb, input_system=args.input_system, input_state=args.input_state,
                T_min=args.T_min, T_max=args.T_max, protein_name=args.protein_name,
                ligand_name=args.ligand_name, ladder='trailblaze', trailblaze_kwargs=walk_settings(args))
    temps, meta = market.trailblaze(args.output_dir, dt=args.timestep)
    print(f"\nLadder: {meta['n_rungs']} rungs, {temps[0]:.2f}-{temps[-1]:.2f} K, total thermodynamic "
          f"length {meta['L_total']:.2f}, neighbour acceptance min {meta['acc_min']:.3f} "
          f"median {meta['acc_median']:.3f}")
    print(f"Cached under {market.trailblaze_dir}; chimpss-fultonmarket --ladder trailblaze on the same "
          f"output_dir will reuse it.")


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
    else:
        _from_run(args)


if __name__ == '__main__':
    main()
