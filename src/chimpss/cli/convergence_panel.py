"""Evaluate the convergence panel on a PTRE run.

Console script: chimpss-convergence-panel

  chimpss-convergence-panel runs/NTSR1_ML-301.hmr.rep0 \\
      --pdb runs/NTSR1_ML-301.hmr.rep0/saved_variables/100/resampled_top.pdb \\
      --pangpcr_prefix models/pangpcr1513 --pdb_code 6OS9 --prefix_curve

Reads the lowest ``--n_states`` rungs of every block (cached under
``<run_dir>/analysis/``), superposes the conserved atoms on the pan-GPCR
reference, and evaluates each gate on the two contiguous halves of the
equilibrated run against tolerances taken from the experimental structures.
"""
import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--pdb', required=True, help='Topology with receptor residue numbering '
                   '(a block\'s resampled_top.pdb).')
    p.add_argument('--pangpcr_prefix', required=True)
    p.add_argument('--pdb_code', required=True, help='Parent structure; sets GPCRdb numbering.')
    p.add_argument('--receptor', default=None,
                   help='GPCRdb protein slug (ntr1_human) for receptor-specific tolerances. '
                        'Default: looked up from the parent structure in the training set.')
    p.add_argument('--stor_dir', default='./many_structures/')
    p.add_argument('--n_states', type=int, default=5, help='Lowest rungs to pool (default 5).')
    p.add_argument('--skip', type=int, default=0, help='Iterations dropped per block (default 0).')
    p.add_argument('--b0', type=int, default=None,
                   help='First equilibrated block. Default: discard the first quarter.')
    p.add_argument('--width_fraction', type=float, default=0.25,
                   help='Width tolerance as a fraction of the same-state RMSD (default 0.25).')
    p.add_argument('--n_pc', type=int, default=10)
    p.add_argument('--prefix_curve', action='store_true',
                   help='Also evaluate the panel at every 8 blocks from 32 onward.')
    p.add_argument('--out', default=None, help='JSON output path.')
    p.add_argument('--prepare', action='store_true',
                   help='Pre-flight for a live run: resolve the conserved atoms against '
                        '--pdb, fetch and cache the GPCRdb numbering, compute the '
                        'tolerances, and write <run_dir>/analysis/live_panel_config.json. '
                        'Run this from a node with network access before submitting a '
                        'job with chimpss-fultonmarket --convergence panel.')
    p.add_argument('--consecutive', type=int, default=3,
                   help='With --prepare: blocks the verdict must hold (default 3).')
    return p.parse_args()


def main():
    from chimpss.analysis.gpcr_pca import fetch_naming_info
    from chimpss.analysis.pca_equilibration import load_model
    from chimpss.analysis import convergence_panel as cp

    a = parse_args()
    if a.prepare:
        from chimpss.analysis.convergence_panel import LivePanel
        lp = LivePanel(run_dir=a.run_dir, topology=a.pdb, model_prefix=a.pangpcr_prefix,
                       pdb_code=a.pdb_code, receptor=a.receptor, stor_dir=a.stor_dir,
                       n_states=a.n_states, skip=a.skip, width_fraction=a.width_fraction,
                       consecutive=a.consecutive, n_pc=a.n_pc)
        cfg = lp.prepare()
        print(f"mapping: identity {cfg['mapping'].get('identity', float('nan')):.3f}, "
              f"{cfg['mapping'].get('n_resolved')}/{cfg['mapping'].get('n_positions')} positions")
        print(f"tolerances: same-state RMSD {cfg['tolerances']['same_state_rmsd']:.2f} A "
              f"(receptor {cfg['receptor']}, n={cfg['tolerances']['receptor_n_structures']}); "
              f"family {cfg['tolerances']['family_same_state_rmsd']:.2f}")
        print('ready: submit with chimpss-fultonmarket --convergence panel '
              f'--panel_pangpcr_prefix {a.pangpcr_prefix} --panel_pdb_code {a.pdb_code}')
        return
    model = load_model(a.pangpcr_prefix)
    naming, _ = fetch_naming_info(a.pdb_code, a.stor_dir)

    receptor = a.receptor
    if receptor is None:
        with open(f'{a.pangpcr_prefix}_meta.json') as fh:
            sm = json.load(fh)['structure_meta']
        receptor = sm.get(a.pdb_code.upper(), {}).get('protein')
        if receptor is None:
            print(f'  parent {a.pdb_code} is not in the training set; using family tolerances')
    tol = cp.experimental_tolerances(a.pangpcr_prefix, receptor=receptor, n_pc=a.n_pc)

    print(f'reading the {a.n_states} lowest rungs of every block ...', flush=True)
    coords, blocks, states, info = cp.collect_low_temperature_frames(
        a.run_dir, a.pdb, model, naming, n_states=a.n_states, skip=a.skip)
    if coords.shape[0] == 0:
        sys.exit('no frames read')
    if info['n_resolved'] < info['n_positions']:
        sys.exit(f"only {info['n_resolved']} of {info['n_positions']} conserved positions "
                 f"resolved in the topology; the panel needs all of them")
    X = cp.superpose_frames(coords, model)
    P = cp.project_superposed(X, model, n_pc=a.n_pc)
    train_sd = np.load(f'{a.pangpcr_prefix}_train_projections.npy')[:, :a.n_pc].std(0)

    r = cp.panel(X, P, blocks, tol, b0=a.b0, width_fraction=a.width_fraction, train_sd=train_sd)
    r['run_dir'] = a.run_dir
    r['mapping'] = {k: info[k] for k in ('identity', 'n_aligned', 'n_positions', 'n_resolved')}
    print(cp.format_panel(r))

    bounds = cp.pool_bounds(a.run_dir)
    if len(bounds) > 2:
        bl = sorted(bounds)
        ov = [cp.pool_overlap(bounds, x, y) for x, y in zip(bl[:-1], bl[1:])]
        r['production_pool_overlap'] = dict(adjacent_median=float(np.median(ov)),
                                            adjacent_p10=float(np.quantile(ov, .1)))
        print(f"production resamples: adjacent source-pool overlap median {np.median(ov):.2f} "
              f"(p10 {np.quantile(ov, .1):.2f}); the trailing-window JSD criterion is "
              f"{'largely by construction' if np.median(ov) > 0.5 else 'comparing mostly independent data'}")

    if a.prefix_curve:
        nb = int(np.unique(blocks).size)
        pref = list(range(32, nb, 8)) + [nb]
        curve = cp.prefix_curve(X, P, blocks, tol, pref, width_fraction=a.width_fraction)
        r['prefix_curve'] = curve
        print(f"\n{'N':>4} {'verdict':>14} {'b0':>3} {'mean RMSD':>9} {'q95':>6} {'state':>12} {'PC1 shift':>9} {'W1':>6} {'basins':>6}")
        for c in curve:
            print(f"{c['N']:>4} {c['verdict']:>14} {c['b0']:>3} "
                  + (f"{c['mean_rmsd']:>9.3f} {c['mean_rmsd_q95']:>6.3f} {c['state']:>12} {c['pc1_shift']:>+9.2f} {c['w1']:>6.3f} {c['n_basins']:>6}"
                     if c['mean_rmsd'] is not None else ''))
        first = next((c['N'] for c in curve if c['verdict'] == 'CONVERGED'), None)
        print(f"first prefix at which the panel reads CONVERGED: {first}")

    out = a.out or os.path.join(a.run_dir, 'analysis', 'convergence_panel.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(r, fh, indent=1, default=float)
    print(f'\nwritten -> {out}')


if __name__ == '__main__':
    main()
