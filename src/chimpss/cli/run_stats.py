"""CLI entry point for manuscript composition / production-statistics tables.

Console script: chimpss-run-stats

Three sub-commands, one per table:

  # Table 1 -- system composition (box averaged over one run's production)
  chimpss-run-stats composition --topology systems/NTSR1_ML-301.topology.pdb \\
      --run_dir runs/NTSR1_ML-301/nm/rep0 --out tables/table1_NTSR1

  # Table 2 -- one row per production run
  chimpss-run-stats production --run_dir runs/NTSR1_ML-301/hmr/rep0 \\
      --iter_length 0.0175 --dt 3.5 --mass_scheme hmr \\
      --label "NTSR1/ML-301 hmr rep.0" --out tables/table2_NTSR1_hmr_rep0

  # Table S4 -- what HMR actually did to the masses
  chimpss-run-stats hmr --base systems/NTSR1_ML-301.FG_system.xml \\
      --hmr systems/NTSR1_ML-301.FG_HMR_system.xml \\
      --topology systems/NTSR1_ML-301.topology.pdb --out tables/tableS4_NTSR1
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)

    c = sub.add_parser('composition', help='Table 1 row for one system')
    c.add_argument('--topology', required=True, help='Bridgeport topology PDB')
    c.add_argument('--run_dir', default=None,
                   help='Production run directory, used to average the box. '
                        'Omit to fall back to the PDB CRYST1 record.')
    c.add_argument('--skip_blocks', type=int, default=0,
                   help='Leading blocks to exclude from the box average.')
    c.add_argument('--out', required=True, help='Output prefix (.json)')

    r = sub.add_parser('production', help='Table 2 row for one run')
    r.add_argument('--run_dir', required=True)
    r.add_argument('--iter_length', type=float, required=True,
                   help='Swap interval in ns (0.010 for the 2 fs arm)')
    r.add_argument('--dt', type=float, required=True, help='Timestep in fs')
    r.add_argument('--mass_scheme', default=None, choices=['nm', 'hmr', None])
    r.add_argument('--label', default=None)
    r.add_argument('--out', required=True, help='Output prefix (.json)')

    h = sub.add_parser('hmr', help='Table S4 -- HMR mass-transfer specification')
    h.add_argument('--base', required=True, help='FG_system.xml')
    h.add_argument('--hmr', required=True, help='FG_HMR_system.xml')
    h.add_argument('--topology', required=True, help='topology.pdb')
    h.add_argument('--out', required=True, help='Output prefix (.json)')

    return p.parse_args()


def main():
    args = parse_args()

    from chimpss.analysis import run_stats as S

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.mode == 'composition':
        box = None
        if args.run_dir:
            mean, sd = S.mean_box(args.run_dir, skip_blocks=args.skip_blocks)
            if mean is not None:
                box = mean
                print(f'Production box (nm): '
                      f'{mean[0]:.3f} x {mean[1]:.3f} x {mean[2]:.3f}  '
                      f'(SD {sd[0]:.4f} / {sd[1]:.4f} / {sd[2]:.4f})')
            else:
                print(f'No box_vectors.npy under {args.run_dir}; '
                      f'falling back to the CRYST1 record.')

        row = S.system_composition(args.topology, box_nm=box)
        S.write_json(row, f'{args.out}.json')
        print(f'\nWrote {args.out}.json')
        for k in ('atoms', 'protein_residues', 'waters', 'popc', 'na', 'cl',
                  'box_nm', 'volume_nm3'):
            print(f'  {k:18s} {row.get(k)}')
        return

    if args.mode == 'production':
        row = S.run_statistics(args.run_dir,
                               iter_length_ns=args.iter_length,
                               dt_fs=args.dt,
                               mass_scheme=args.mass_scheme,
                               label=args.label)
        S.write_json(row, f'{args.out}.json')

        if 'error' in row:
            print(f"WARNING: {row['error']}")
            sys.exit(1)

        print(f'\nWrote {args.out}.json')
        for k in ('run', 'mass_scheme', 'dt_fs', 'blocks', 'final_replicas',
                  'states_inserted', 'aggregate_ns', 'ns_per_replica',
                  'acceptance_mean', 'acceptance_min_pair'):
            print(f'  {k:18s} {row.get(k)}')
        if row.get('acceptance_mean') is None:
            print(f"  NOTE: acceptance unavailable -- {row.get('acceptance_source')}")
        return

    # hmr mode
    report = S.hmr_report(args.base, args.hmr, args.topology)
    S.write_json(report, f'{args.out}.json')

    print(f'\nWrote {args.out}.json')
    print(f"  particles                {report['n_particles']}")
    print(f"  atoms changed            {report['n_atoms_changed']}")
    print(f"  solute hydrogens         {report['n_solute_hydrogens']}")
    print(f"  water atoms changed      {report['n_water_atoms_changed']} "
          f"(must be 0)")
    print(f"  total mass delta (Da)    {report['total_mass_delta_da']:.3e} "
          f"(must be ~0)")
    print('  hydrogen mass populations:')
    for pop in report['hydrogen_mass_populations']:
        print(f"    {pop['final_mass_da']:.6f} Da "
              f"({pop['scale_vs_standard']:.2f}x)  n={pop['count']}")

    if report['n_water_atoms_changed']:
        print('\nERROR: water was repartitioned -- apply_hmr resname filter missed '
              'this water model.')
        sys.exit(1)
    if abs(report['total_mass_delta_da']) > 1e-6:
        print('\nERROR: total mass not conserved.')
        sys.exit(1)


if __name__ == '__main__':
    main()
