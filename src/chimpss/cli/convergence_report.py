"""
CLI entry point for retrospective convergence and mixing analysis.

Answers, from cached block data and without running any simulation:

* what the convergence verdict would have been under a different comparison
  window, equilibration detector, or block exclusion;
* where the Frobenius and JSD thresholds sit relative to the noise floor of the
  statistics themselves;
* how well the replica ladder actually mixed, and how long a configuration
  takes to traverse it.

The expensive step is the all-pairs distance cache, which is built once and
written to ``<run_dir>/analysis/pairwise.npz``; every question after that is
milliseconds.

Console script: chimpss-convergence-report
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Retrospective convergence and mixing report for a PTRE run.')
    p.add_argument('run_dir', help='Production run directory (contains saved_variables/).')
    p.add_argument('--pdb', default=None,
                   help='Topology PDB, required for equilibration detection on the '
                        'energy series (the burn-in equil.pdb).')
    p.add_argument('--window_blocks', type=int, nargs='*', default=[1, 3, 5, 10],
                   help='Trailing window widths to replay. Default: 1 3 5 10.')
    p.add_argument('--total_n_sims', type=int, default=80,
                   help='Requested block count, for the minimum-fraction check.')
    p.add_argument('--minimum_fraction', type=float, default=0.35)
    p.add_argument('--frobenius_thresh', type=float, default=0.05)
    p.add_argument('--jsd_thresh', type=float, default=0.10)
    p.add_argument('--null_calibrate', action='store_true',
                   help='Bootstrap the null distribution of each metric within a '
                        'single block and report where the threshold sits relative '
                        'to it.')
    p.add_argument('--null_block', type=int, default=None,
                   help='Block to calibrate against. Default: the median block.')
    p.add_argument('--n_boot', type=int, default=200)
    p.add_argument('--no_mixing', action='store_true', help='Skip mixing diagnostics.')
    p.add_argument('--overwrite_cache', action='store_true')
    p.add_argument('--json_out', default=None, help='Write the full report as JSON.')
    return p.parse_args()


def _fmt(v, nd=4):
    if v is None:
        return '-'
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return 'inf' if np.isinf(v) else 'NaN'
    return f'{v:.{nd}f}'


def main():
    args = parse_args()
    from chimpss.analysis.convergence import (PairwiseCache, replay,
                                              null_distribution, calibrated_threshold,
                                              METRICS)
    from chimpss.analysis import mixing as mx

    run = args.run_dir.rstrip('/')
    report = {'run_dir': run}

    print('=' * 72)
    print(f'  Convergence report :: {os.path.basename(run)}')
    print('=' * 72)

    cache = PairwiseCache.build(run, overwrite=args.overwrite_cache)
    print(f'\n  blocks with a complete matrix set : {len(cache.blocks)}')
    report['n_blocks_with_matrices'] = len(cache.blocks)

    # --- equilibration series, if a topology was supplied --------------------
    discard = None
    if args.pdb:
        from chimpss.analysis.equilibration import energy_series, DETECTORS
        series = energy_series(run, args.pdb)
        n = series.size
        print(f'  energy series                     : {n} frames')
        print('\n  Equilibration detectors on the full series')
        print(f'    {"detector":<12} {"t0":>6} {"discard":>9}')
        det = {}
        for name, fn in DETECTORS.items():
            try:
                t0 = int(fn(series))
                det[name] = {'t0': t0, 'discard': t0 / n}
                print(f'    {name:<12} {t0:>6} {100 * t0 / n:>8.1f}%')
            except Exception as exc:
                det[name] = {'error': str(exc)}
        report['detectors'] = det

        # per-block discard from the incumbent, for the published window rule
        per = max(1, n // max(len(cache.blocks), 1))
        discard = []
        for b in range(1, max(cache.blocks) + 2):
            x = series[:min(b * per, n)]
            try:
                discard.append(int(DETECTORS['max_neff'](x)) / x.size)
            except Exception:
                discard.append(None)

    # --- window replay -------------------------------------------------------
    print('\n  Convergence verdict by comparison window')
    print(f'    {"window":<22} {"passing":>8} {"first":>7}')
    windows = {}
    if discard is not None:
        r = replay(cache, discard=discard, total_n_sims=args.total_n_sims,
                   minimum_fraction=args.minimum_fraction,
                   frobenius_thresh=args.frobenius_thresh, jsd_thresh=args.jsd_thresh)
        windows['published (from t0)'] = {'passing': r.passing, 'first': r.first}
        print(f'    {"published (from t0)":<22} {len(r.passing):>8} {str(r.first):>7}')
    for k in args.window_blocks:
        r = replay(cache, window_blocks=k, discard=discard,
                   require_gate=discard is not None,
                   total_n_sims=args.total_n_sims, minimum_fraction=args.minimum_fraction,
                   frobenius_thresh=args.frobenius_thresh, jsd_thresh=args.jsd_thresh)
        windows[f'trailing {k}'] = {'passing': r.passing, 'first': r.first}
        print(f'    {"trailing " + str(k) + " block(s)":<22} {len(r.passing):>8} {str(r.first):>7}')
    report['windows'] = windows

    # --- null calibration ----------------------------------------------------
    if args.null_calibrate:
        blk = args.null_block if args.null_block is not None else \
            cache.blocks[len(cache.blocks) // 2]
        print(f'\n  Threshold vs the noise floor  (bootstrap within block {blk}, '
              f'n={args.n_boot})')
        print(f'    {"metric":<14} {"median":>8} {"p99":>8} {"thresh/p99":>11}')
        nulls = {}
        for m in METRICS:
            try:
                d = null_distribution(run, blk, m, n_boot=args.n_boot)
                p99 = calibrated_threshold(d, 0.99)
                nulls[m] = {'median': float(np.median(d)), 'p99': p99,
                            'ratio': args.jsd_thresh / p99 if p99 else None}
                print(f'    {m:<14} {np.median(d):>8.4f} {p99:>8.4f} '
                      f'{args.jsd_thresh / p99:>10.1f}x')
            except Exception as exc:
                nulls[m] = {'error': str(exc)}
                print(f'    {m:<14} failed: {exc}')
        report['null'] = nulls

    # --- mixing --------------------------------------------------------------
    if not args.no_mixing:
        mr = mx.mixing_report(run)
        rt = mr.get('round_trips', {})
        print('\n  Replica ladder mixing')
        print(f'    median tau_2                    : {_fmt(mr["median_tau_2"], 1)} iterations')
        print(f'    block length                    : {mr["block_length_iterations"]} iterations')
        if rt:
            print(f'    full round trips (whole run)    : {rt.get("total_round_trips")}')
            print(f'    half legs                       : {rt.get("half_legs")}')
            print(f'    mean iterations per half leg    : {_fmt(rt.get("mean_iterations_per_leg"), 1)}')
        report['mixing'] = {'median_tau_2': mr['median_tau_2'],
                            'block_length_iterations': mr['block_length_iterations'],
                            'round_trips': rt}

    # --- resample health -----------------------------------------------------
    print('\n  Resample health (pool size predicts MBAR collapse)')
    pools = []
    for b in cache.blocks:
        ip = os.path.join(run, 'saved_variables', str(b), 'resampled_indcs.npy')
        if not os.path.exists(ip):
            continue
        i = np.load(ip)
        t0 = int(i[:, 1].min())
        pools.append((b, int(i[:, 1].max()) - t0 + 1, len(np.unique(i, axis=0))))
    if pools:
        small = [p for p in pools if p[1] < 35]
        print(f'    blocks with pool < 35 frames    : {len(small)} of {len(pools)}')
        print(f'    blocks drawing <= 10 unique     : '
              f'{[p[0] for p in pools if p[2] <= 10]}')
        report['resample'] = {'n_blocks': len(pools),
                              'small_pool_blocks': [p[0] for p in small],
                              'collapsed_blocks': [p[0] for p in pools if p[2] <= 10]}

    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f'\n  report written to {args.json_out}')
    print('=' * 72)


if __name__ == '__main__':
    main()
