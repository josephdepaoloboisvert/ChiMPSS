"""
CLI entry point for the convergence method-assessment data bundle.

Collects, in one pass over a completed parallel-tempering run, every number the
assessment report is built from, and writes each figure as a standalone SVG.
The prose is deliberately *not* generated: what a given run demonstrates about
a method is a judgement, and the point of the report is the judgement.  This
produces the evidence and the charts; an analyst (or an agent following the
``convergence-report`` skill) writes the argument around them.

Everything runs from cached block data on CPU.  The expensive steps -- the
all-pairs distance cache and the alpha-carbon coordinate cache -- are persisted
under ``<run_dir>/analysis/`` and reused.

Console script: chimpss-convergence-bundle
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help='Production run directory (contains saved_variables/).')
    p.add_argument('--pdb', default=None,
                   help='Topology PDB, for the energy and structural series.')
    p.add_argument('--out_dir', default=None,
                   help='Where to write bundle.json and figures/ '
                        '(default: <run_dir>/analysis/report).')
    p.add_argument('--skip', type=int, default=10,
                   help='Iterations discarded at the start of each block.')
    p.add_argument('--window_blocks', type=int, nargs='*', default=[1, 3, 5, 10, 20])
    p.add_argument('--jsd_thresh', type=float, default=0.10)
    p.add_argument('--gate_detector', default='max_neff',
                   help='Detector supplying the per-block equilibration fraction for '
                        'the gate. This materially changes the passing counts: on '
                        'NTSR1 rep0 a one-block window passes 5 blocks under max_neff '
                        'and 23 under the MSER family. Default: max_neff (incumbent).')
    p.add_argument('--n_clusters', type=int, default=20)
    p.add_argument('--null_reps', type=int, default=25,
                   help='Half-splits per block for the within-block null. Fewer than '
                        'about 20 gives an unstable 99th percentile.')
    p.add_argument('--no_realign', action='store_true',
                   help='Skip realign_cache. Only correct if ca_coords.npz was built '
                        'with a shared reference; otherwise every structural metric '
                        'reads global rotation instead of conformation.')
    p.add_argument('--pangpcr_prefix', default=None,
                   help='Pan-GPCR PCA model prefix, to add the external-basis figures.')
    p.add_argument('--pdb_code', default=None,
                   help='Parent crystal structure code, required with --pangpcr_prefix.')
    p.add_argument('--stor_dir', default='./many_structures/')
    p.add_argument('--pc_timeseries', type=int, default=3,
                   help='Leading components of the external basis to track per block. '
                        'PC1 is usually the activation coordinate, but motion confined '
                        'to PC1 and motion spread over several components are different '
                        'physical statements. Three by default because on '
                        'NTSR1_ML-301.hmr.rep0 the residual drift late in the run sits '
                        'on PC3, not PC1: tracking only the leading component would have '
                        'reported the ensemble as settled while it was still moving.')
    p.add_argument('--collapse_at', type=int, default=10,
                   help='Distinct-frame count at or below which a resample counts as collapsed.')
    return p.parse_args()


def _jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def main():
    args = parse_args()
    run = args.run_dir.rstrip('/')
    out_dir = args.out_dir or os.path.join(run, 'analysis', 'report')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    from chimpss.analysis import mixing as mx
    from chimpss.analysis import report_figures as rf
    from chimpss.analysis.convergence import (METRICS, PairwiseCache, ca_cache,
                                              cluster_populations,
                                              covariance_overlap, realign_cache,
                                              replay)
    from chimpss.analysis.equilibration import DETECTORS

    B = {'run_dir': run, 'skip': args.skip}
    figs = {}

    def emit(name, svg):
        figs[name] = svg
        with open(os.path.join(fig_dir, f'{name}.svg'), 'w') as fh:
            fh.write(svg)

    print('=' * 70)
    print(f'  Convergence bundle :: {os.path.basename(run)}')
    print('=' * 70)

    # --- pairwise distances --------------------------------------------------
    cache = PairwiseCache.build(run)
    have = list(cache.blocks)
    B['n_blocks'] = len(have)
    print(f'  blocks with a complete matrix set : {len(have)}')

    import itertools
    pw = {}
    for m in METRICS:
        M = cache.jsd[m] if hasattr(cache, 'jsd') else None
        if M is None:
            d = np.load(os.path.join(run, 'analysis', 'pairwise.npz'))
            M = d[f'jsd_{m}']
        v = np.array([M[a, b] for a, b in itertools.combinations(have, 2)])
        v = v[np.isfinite(v)]
        pw[m] = {'median': float(np.median(v)), 'n': int(v.size),
                 'pass_frac': float((v < args.jsd_thresh).mean()),
                 'values': v.tolist()}
        print(f'    {m:<14} median {np.median(v):.4f}  '
              f'clears {args.jsd_thresh}: {100*(v<args.jsd_thresh).mean():.1f}%')
    B['pairwise'] = {k: {x: y for x, y in v.items() if x != 'values'}
                     for k, v in pw.items()}
    emit('pairwise_distributions', rf.pairwise_hist_svg(pw, thresh=args.jsd_thresh))

    # --- equilibration detectors and window replay ---------------------------
    if args.pdb:
        from chimpss.analysis.equilibration import energy_series, structural_series
        try:
            series = energy_series(run, args.pdb, skip=args.skip)
            B['detectors'] = {n: int(f(series)) for n, f in DETECTORS.items()}
            B['n_frames'] = int(series.size)
            print('  equilibration detectors on the energy series:')
            for n, t in B['detectors'].items():
                print(f'    {n:<12} t0 {t:>5}  ({100*t/series.size:.1f}%)')
        except Exception as exc:
            print(f'  energy series unavailable: {exc}')
        try:
            B['structural_t0'] = {n: int(f(structural_series(
                run, args.pdb, skip=args.skip))) for n, f in DETECTORS.items()}
        except Exception as exc:
            print(f'  structural series unavailable: {exc}')

    # Per-block equilibration fraction, as the live code would have seen it.
    # replay() requires this for the gate; without it require_gate=True can
    # never be satisfied and every window reports zero passing blocks.
    discard = None
    if 'detectors' in B and B.get('n_frames'):
        n = B['n_frames']
        per = max(1, n // max(len(have), 1))
        discard = []
        for b in range(1, max(have) + 2):
            x = series[:min(b * per, n)]
            try:
                discard.append(int(DETECTORS[args.gate_detector](x)) / x.size)
            except Exception:
                discard.append(None)
        B['discard_series'] = discard
        B['gate_detector'] = args.gate_detector

    wins = {}
    for k in args.window_blocks:
        r = replay(cache, window_blocks=k, discard=discard,
                   require_gate=discard is not None,
                   jsd_thresh=args.jsd_thresh)
        wins[k] = {'passing': list(r.passing), 'first': r.first}
        print(f'    trailing {k:>2} block(s): {len(r.passing)} passing'
              + (f'  (gate: {args.gate_detector})' if discard is not None
                 else '  (no gate: --pdb not given)'))
    B['windows'] = wins

    # --- structural ensembles ------------------------------------------------
    ca = ca_cache(run, _printf=lambda *a, **k: None)
    if not args.no_realign:
        ca = realign_cache(ca)
    blocks = sorted(ca)
    print(f'  alpha-carbon cache: {len(blocks)} blocks'
          + ('' if args.no_realign else ' (realigned to a shared reference)'))

    cp = cluster_populations(ca, n_clusters=args.n_clusters)
    H, centers = cp['histograms'], cp['centers']
    adj = [(blocks[i], _jsd(H[blocks[i]], H[blocks[i + 1]]))
           for i in range(len(blocks) - 1) if blocks[i + 1] - blocks[i] == 1]

    rng = np.random.default_rng(0)
    null = []
    for b in blocks:
        X = ca[b].reshape(ca[b].shape[0], -1)
        for _ in range(args.null_reps):
            idx = rng.permutation(X.shape[0]); h = X.shape[0] // 2
            def hist(Y):
                lab = ((Y[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
                c = np.bincount(lab, minlength=centers.shape[0]).astype(float)
                return c / c.sum()
            null.append(_jsd(hist(X[idx[:h]]), hist(X[idx[h:]])))
    p99 = float(np.percentile(null, 99))
    vals = np.array([v for _, v in adj])
    B['cluster'] = {'adjacent': [[int(a), float(v)] for a, v in adj],
                    'null_p99': p99, 'null_n': len(null),
                    'null_median': float(np.median(null)),
                    'inside_null': float(np.mean(vals < p99)),
                    'inside_null_last20': float(np.mean(vals[-20:] < p99)),
                    'occupancy': {str(k): int(v) for k, v in cp['occupancy'].items()}}
    print(f'  cluster populations: adjacent median {np.median(vals):.4f}, '
          f'null p99 {p99:.4f} (n={len(null)}), inside {100*np.mean(vals<p99):.1f}%')
    emit('cluster_populations', rf.series_svg(
        [a for a, _ in adj], list(vals), 'population JSD, block to block',
        band=p99, band_label=f'null p99 {p99:.3f}', H=290,
        aria='Cluster-population Jensen-Shannon distance between adjacent blocks.',
        point_color=lambda a, b: 'var(--pass)' if b < p99 else 'var(--fail)'))

    ov = [(blocks[i], covariance_overlap(ca[blocks[i]], ca[blocks[i + 1]]))
          for i in range(len(blocks) - 1) if blocks[i + 1] - blocks[i] == 1]
    ovv = np.array([v for _, v in ov])
    quarters = [float(x.mean()) for x in np.array_split(ovv, 4)]
    B['covariance'] = {'adjacent': [[int(a), float(v)] for a, v in ov],
                       'quarters': quarters}
    print(f'  covariance overlap quarters: {[round(q,3) for q in quarters]}')
    emit('covariance_overlap', rf.series_svg(
        [a for a, _ in ov], list(ovv), 'covariance overlap, block to block',
        quarters=quarters, H=290,
        aria='Romo-Grossfield covariance overlap between adjacent blocks.'))

    # --- mixing --------------------------------------------------------------
    mr = mx.mixing_report(run)
    pb, rt = mr['per_block'], mr['round_trips']
    bl = sorted(pb)
    tau = [pb[b]['tau_2'] if np.isfinite(pb[b]['tau_2']) else None for b in bl]
    B['mixing'] = {
        'blocks': [int(b) for b in bl], 'tau_2': tau,
        'acceptance': [float(pb[b]['acceptance']) for b in bl],
        'walkers_at_state0': [int(pb[b]['walkers_at_state0']) for b in bl],
        'median_tau_2': float(mr['median_tau_2']),
        'block_length_iterations': int(mr['block_length_iterations']),
        'round_trips': {k: v for k, v in rt.items() if k != 'per_walker'},
        'per_walker': [int(x) for x in rt.get('per_walker', [])],
    }
    finite = [t for t in tau if t]
    print(f"  mixing: median tau_2 {mr['median_tau_2']:.1f} vs block "
          f"{mr['block_length_iterations']} iterations, "
          f"{rt.get('total_round_trips')} round trips")
    emit('mixing_tau2', rf.series_svg(
        [int(b) for b in bl], [t if t else 0.0 for t in tau],
        'mixing time &#964;&#8322; (iterations)', H=290,
        hline=mr['block_length_iterations'],
        hline_label=f"block = {mr['block_length_iterations']} it.",
        color='var(--accent)',
        aria='Replica-ladder mixing time tau-2 per block.'))
    if rt.get('per_walker'):
        emit('round_trips', rf.roundtrip_hist_svg(rt['per_walker']))

    # --- resample pool -------------------------------------------------------
    pool = []
    for b in sorted(int(x) for x in os.listdir(os.path.join(run, 'saved_variables'))
                    if x.isdigit()):
        ip = os.path.join(run, 'saved_variables', str(b), 'resampled_indcs.npy')
        if not os.path.exists(ip):
            continue
        i = np.load(ip)
        t0 = int(i[:, 1].min())
        pool.append([b, int(i[:, 1].max()) - t0 + 1, int(len(np.unique(i, axis=0))), t0])
    B['pool'] = pool
    if pool:
        P = np.array(pool, float)
        collapsed = [int(p[0]) for p in pool if p[2] <= args.collapse_at]
        B['collapsed_blocks'] = collapsed
        print(f'  resample pool: unique median {np.median(P[:,2]):.0f}, '
              f'collapsed {collapsed}')
        emit('resample_pool', rf.pool_svg(pool, collapse_at=args.collapse_at))
        emit('walkers_at_state0', rf.walkers_svg(
            B['mixing']['blocks'], B['mixing']['walkers_at_state0'], collapsed))

    # --- MBAR source-pool overlap --------------------------------------------
    # The JSD between two resamples is a monotone function of how much source
    # data they share (NTSR1 rep0: median 0.02-0.05 at >90 % overlap, 0.17-0.24
    # at none).  Report the overlap so a trailing-window pass can be read for
    # what it is.
    try:
        from chimpss.analysis.convergence_panel import pool_bounds, pool_overlap
        bounds = pool_bounds(run)
        bl = sorted(bounds)
        if len(bl) > 2:
            ov = [pool_overlap(bounds, x, y) for x, y in zip(bl[:-1], bl[1:])]
            B['pool_overlap'] = {'adjacent_median': float(np.median(ov)),
                                 'adjacent_p10': float(np.quantile(ov, .1)),
                                 'per_pair': [[int(y), float(o)] for y, o in zip(bl[1:], ov)]}
            print(f"  MBAR source pools: adjacent overlap median {np.median(ov):.2f} "
                  f"(p10 {np.quantile(ov, .1):.2f})"
                  + ("  -- trailing-window JSD agreement is largely by construction"
                     if np.median(ov) > 0.5 else ""))
    except Exception as exc:                                    # pragma: no cover
        print(f"  pool overlap unavailable: {exc}")

    # --- external basis ------------------------------------------------------
    if args.pangpcr_prefix and args.pdb_code and args.pdb:
        try:
            from chimpss.analysis.gpcr_pca import (fetch_naming_info,
                                                   load_training_projections)
            from chimpss.analysis.pca_equilibration import (_imputation_for,
                                                            collect_frames,
                                                            load_model,
                                                            project_coordinates)
            model = load_model(args.pangpcr_prefix)
            naming, _ = fetch_naming_info(args.pdb_code, args.stor_dir)
            coords, info = collect_frames(run, args.pdb, model, naming,
                                          skip=args.skip,
                                          _printf=lambda *a, **k: None)
            ext = project_coordinates(coords, model,
                                      imputation=_imputation_for(info, model))
            train, states, evr = load_training_projections(args.pangpcr_prefix)
            blk = np.array(info['blocks']); ub = np.unique(blk)
            means = np.stack([ext[blk == k].mean(axis=0) for k in ub])
            n_pcs = max(1, min(args.pc_timeseries, means.shape[1]))
            from scipy import stats as _st

            # Centroids are per-component: the activation separation lives on
            # PC1, but a later component may separate something else, and a
            # trend there is only interpretable against its own reference
            # levels.
            cent = {c: {s: float(train[[x == s for x in states], c].mean())
                        for s in set(states)} for c in range(n_pcs)}
            comps = {}
            for c in range(n_pcs):
                sl, ic, r, p, _ = _st.linregress(ub, means[:, c])
                seg_svg, seg = rf.segmented_drift_svg(
                    ub, means[:, c], cent[c], ylab=f'mean PC{c+1}')
                comps[f'pc{c+1}'] = {
                    'by_block': means[:, c].tolist(),
                    'explained_variance': float(evr[c]),
                    'trajectory_sd': float(means[:, c].std()),
                    'training_sd': float(train[:, c].std()),
                    'fit': {'slope': float(sl), 'intercept': float(ic),
                            'r': float(r), 'p': float(p)},
                    'segmented': seg,
                }
                emit(f'pc{c+1}_drift', seg_svg)
                print(f'  pan-GPCR: PC{c+1} drift {sl:+.4f}/block, r={r:+.3f}, '
                      f'p={p:.2e}   (segmented: {seg["before"]["slope"]:+.4f} -> '
                      f'{seg["after"]["slope"]:+.4f} at block {seg["changepoint"]:.0f})')

            B['pangpcr'] = {'mapping': {k: v for k, v in info.items()
                                        if k in ('identity', 'n_aligned', 'n_traj',
                                                 'offsets', 'n_resolved', 'n_positions')},
                            'centroids': cent[0], 'centroids_by_pc': cent,
                            'n_train': int(train.shape[0]),
                            'blocks': ub.tolist(), 'components': comps,
                            # kept for callers written against the single-component form
                            'pc1_by_block': means[:, 0].tolist(),
                            'fit': comps['pc1']['fit']}

            emit('pc_landscape', rf.pc_landscape_svg(train, states, means, ub, ext, evr))
            emit('pc_components', rf.multi_series_svg(
                ub, {f'PC{c+1}  ({evr[c]:.1%} var)': means[:, c].tolist()
                     for c in range(n_pcs)},
                'mean projection per block',
                aria='Mean projection of each block onto the leading components of the '
                     'external basis. A component with no trend is one along which the '
                     'ensemble is not moving.'))
        except Exception as exc:
            print(f'  pan-GPCR projection skipped: {type(exc).__name__}: {exc}')

    with open(os.path.join(out_dir, 'bundle.json'), 'w') as fh:
        json.dump(B, fh, indent=1, default=str)
    print(f'\n  bundle  -> {os.path.join(out_dir, "bundle.json")}')
    print(f'  figures -> {fig_dir}  ({len(figs)} SVG)')
    print('=' * 70)


if __name__ == '__main__':
    main()
