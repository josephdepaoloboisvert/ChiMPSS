#!/usr/bin/env python
"""Mixing, state walks, resampled-pool structure/energy series and resampling health for a FultonMarket run.

    python mix_and_health.py RUN_DIR [--ref PDB] [--ligand 'resname UNK'] [--skip 10] [--logs GLOB]
                             [--out JSON] [--no-offline] [--no-coords] [--blocks 0:10] [--n-check 6] [--seed 0]

Prints a markdown summary and writes <run>/analysis/mix_and_health.json, which also carries the
rows for the two report charts (chart_data.walks, chart_data.series, chart_data.pool).

Conventions it relies on (verified on NTSR1_ML-301.hmr.rep4, October 2026):
- states.npy is (iteration, replica); energies.npy is (iteration, replica, state) reduced potentials;
  positions.npy is a headerless float32 memmap (iteration, replica, atom, 3).
- At every resume the engine gives replica k the configuration that ended the previous block at rung k,
  so walkers are followed by re-pairing columns on their final rung (checked on coordinates here).
- resampled_indcs.npy rows are (state, frame) in FultonMarketAnalysis' concatenation of all blocks saved
  at evaluation time, with its default skip of 10 frames per block; the state index refers to the
  ladder of the block being evaluated.
- The engine skips the convergence check on the block that reaches the aggregate-time cap, so that block
  has no resampled set; this script regenerates it with the same analyzer unless --no-offline.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

KB = 0.008314462618
ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def log(msg):
    print(f'[mix_and_health] {msg}', file=sys.stderr, flush=True)


def parse_blocks(spec, available):
    if not spec:
        return list(available)
    out = []
    for part in spec.split(','):
        if ':' in part:
            a, b = part.split(':')
            out.extend(x for x in available if int(a) <= x < int(b))
        else:
            out.append(int(part))
    return sorted(set(out) & set(available))


def detect_settings(run_dir, logs_glob):
    logs = sorted(glob.glob(logs_glob)) if logs_glob else []
    ref = ligand = None
    iter_ps = None
    for lf in logs:
        with open(lf, errors='replace') as fh:
            for line in fh:
                m = re.search(r'Found input_pdb: (\S+)', line)
                if m:
                    ref = m.group(1)
                m = re.search(r"matching --ligand '([^']+)'", line)
                if m:
                    ligand = m.group(1)
                m = re.search(r'iter_length\s*:\s*([\d.]+) ns', line)
                if m:
                    iter_ps = float(m.group(1)) * 1000.0
    cfg = os.path.join(run_dir, 'analysis', 'live_panel_config.json')
    if ref is None and os.path.exists(cfg):
        try:
            ref = json.load(open(cfg)).get('topology')
        except Exception:
            pass
    return logs, ref, ligand, iter_ps


def parse_log_ess(logs):
    """ESS of the 310 K weight column for every block whose resample the engine wrote."""
    out = {}
    pat_ess = re.compile(r'effective sample size ([\d.]+) of (\d+) \(([\d.]+)%\)')
    pat_wr = re.compile(r'frames written to .*saved_variables/(\d+)/resampled_top')
    pat_t0 = re.compile(r'Currently would discard ([\d.]+)% to energy equilibration')
    for lf in logs:
        last = None
        last_t0 = None
        with open(lf, errors='replace') as fh:
            for line in fh:
                m = pat_ess.search(line)
                if m:
                    last = (float(m.group(1)), int(m.group(2)), float(m.group(3)))
                m = pat_t0.search(line)
                if m:
                    last_t0 = float(m.group(1))
                m = pat_wr.search(line)
                if m and last is not None:
                    out[int(m.group(1))] = {'ess': last[0], 'pool': last[1], 'pct': last[2], 'discard_pct': last_t0}
    return out


def counts_tau2(walk, mixing_time):
    """Relaxation time of the rung transition matrix on a walker-continuous matrix with NaN for unborn walkers."""
    K = walk.shape[1]
    a, b = walk[:-1].ravel(), walk[1:].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    c = np.zeros((K, K))
    np.add.at(c, (a[ok].astype(int), b[ok].astype(int)), 1.0)
    visited = c.sum(axis=1) > 0                 # a rung nobody has reached yet would add a spurious unit eigenvalue
    c = c[np.ix_(visited, visited)]
    return float(mixing_time(c / c.sum(axis=1)[:, None]))


def half_legs(walk, K):
    hl = 0
    for w in range(walk.shape[1]):
        p = walk[:, w]
        p = p[~np.isnan(p)]
        side = None
        for v in p:
            if v == 0:
                hl += side == 'top'
                side = 'bot'
            elif v == K - 1:
                hl += side == 'bot'
                side = 'top'
    return int(hl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--ref', help='reference PDB for RMSD (default: the run input PDB found in the logs or live_panel_config.json)')
    ap.add_argument('--ligand', help="ligand selection (default: the --ligand of the run, else 'resname UNK')")
    ap.add_argument('--skip', type=int, default=10, help='frames per block the analyzer never pools (its default, 10)')
    ap.add_argument('--logs', help='glob of the run logs (default <run>/../../logs/*<run_name>*.out)')
    ap.add_argument('--out', help='JSON output (default <run>/analysis/mix_and_health.json)')
    ap.add_argument('--blocks', help='blocks to include, e.g. 0:10 or 0,2,5')
    ap.add_argument('--no-offline', action='store_true', help='do not regenerate missing resampled sets')
    ap.add_argument('--no-coords', action='store_true', help='skip everything that reads positions.npy')
    ap.add_argument('--n-check', type=int, default=6, help='DCD frames per block to cross-check the index map')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iter-ps', type=float, help='picoseconds per exchange iteration (default: from the logs)')
    ap.add_argument('--assume-continuity', choices=('auto', 'yes', 'no'), default='auto',
                    help='auto: verify walker continuity across resumes on coordinates; no: treat every block as '
                         'independent segments (runs saved before the per-replica seeding fix); yes: trust the pairing')
    args = ap.parse_args()

    import mdtraj as md
    from chimpss.analysis import mixing as M
    try:
        from chimpss.analysis import ladder as L
    except Exception:
        L = None

    t_start = time.time()
    R = os.path.abspath(args.run_dir.rstrip('/'))
    run_name = os.path.basename(R)
    logs_glob = args.logs or os.path.join(os.path.dirname(os.path.dirname(R)), 'logs', f'*{run_name}*.out')
    logs, ref_auto, lig_auto, iter_auto = detect_settings(R, logs_glob)
    ref_path = args.ref or ref_auto
    ligand = args.ligand or lig_auto or 'resname UNK'
    iter_ps = args.iter_ps or iter_auto
    if ref_path is None or not os.path.exists(ref_path):
        sys.exit('reference PDB not found; pass --ref')
    out_path = args.out or os.path.join(R, 'analysis', 'mix_and_health.json')
    rng = np.random.default_rng(args.seed)
    log(f'run {R}; ref {ref_path}; ligand {ligand!r}; {len(logs)} log file(s); iter_ps {iter_ps}')

    sv = lambda b, f: os.path.join(R, 'saved_variables', str(b), f)
    need = ('states.npy', 'energies.npy', 'temperatures.npy')
    available = sorted(int(d) for d in os.listdir(os.path.join(R, 'saved_variables'))
                       if d.isdigit() and all(os.path.exists(sv(int(d), f)) for f in need))
    blocks = parse_blocks(args.blocks, available)
    temps = {b: np.load(sv(b, 'temperatures.npy')) for b in available}
    states = {b: np.load(sv(b, 'states.npy')) for b in available}
    energies = {b: np.load(sv(b, 'energies.npy'), mmap_mode='r') for b in available}
    betas = {b: 1.0 / (KB * temps[b]) for b in available}
    T_final = temps[blocks[-1]]
    K = len(T_final)
    fidx = {b: np.array([int(np.argmin(np.abs(T_final - t))) for t in temps[b]]) for b in available}

    ref = md.load(ref_path)
    N_AT = ref.n_atoms
    ca = ref.topology.select('name CA')
    lig = ref.topology.select(f'({ligand}) and not element H')
    if len(lig) == 0:
        log(f'no heavy atoms match {ligand!r}; ligand RMSD will be absent')
    sel = np.concatenate([ca, lig]) if len(lig) else ca
    ca_s = np.arange(len(ca))
    lig_s = np.arange(len(ca), len(sel))
    ref_s = ref.atom_slice(sel)
    top_s = ref_s.topology

    pos = {}
    if not args.no_coords:
        for b in available:
            p = sv(b, 'positions.npy')
            shape = (states[b].shape[0], states[b].shape[1], N_AT, 3)
            if os.path.exists(p) and os.path.getsize(p) == int(np.prod(shape)) * 4:
                pos[b] = np.memmap(p, dtype=np.float32, mode='r', shape=shape)
            else:
                log(f'block {b}: positions.npy missing or not {shape}; coordinate metrics skipped for it')
    coords_ok = all(b in pos for b in blocks)

    def enthalpy(b, it, rep):
        u = np.asarray(energies[b][it, rep], dtype=np.float64)
        return float(np.mean(u / betas[b]))

    # ------------------------------------------------------------ walker-continuous state walks
    n_rows = sum(states[b].shape[0] for b in blocks)
    row_start = {}
    r0 = 0
    for b in blocks:
        row_start[b] = r0
        r0 += states[b].shape[0]
    block_of_row = np.concatenate([np.full(states[b].shape[0], b) for b in blocks])

    def build_walk(pairing):
        """Rows = saved frames of all blocks, columns = walkers (pairing=True) or replica slots (False)."""
        walk = np.full((n_rows, K), np.nan)
        col = {}
        first = blocks[0]
        col[first] = np.full(K, -1, dtype=int)
        col[first][:states[first].shape[1]] = np.arange(states[first].shape[1])
        for i, b in enumerate(blocks):
            s = states[b]
            c = col[b]
            for w in range(K):
                if c[w] >= 0:
                    walk[row_start[b]:row_start[b] + s.shape[0], w] = fidx[b][s[:, c[w]]]
            if i + 1 < len(blocks):
                nb = blocks[i + 1]
                nxt = np.full(K, -1, dtype=int)
                if pairing:
                    taken = np.zeros(states[nb].shape[1], dtype=bool)
                    for w in range(K):
                        if c[w] >= 0:
                            nxt[w] = fidx[b][s[-1, c[w]]]       # replica index at resume = rung index on the new ladder
                            taken[nxt[w]] = True
                    free = list(np.where(~taken)[0])
                    for w in np.where(nxt < 0)[0]:
                        if free:
                            nxt[w] = free.pop(0)
                else:
                    nxt[:states[nb].shape[1]] = np.arange(states[nb].shape[1])
                col[nb] = nxt
        return walk, col

    walk, col = build_walk(pairing=args.assume_continuity != 'no')
    boundary_checks = []
    continuity = None
    if args.assume_continuity == 'no':
        continuity = False
    elif args.assume_continuity == 'yes':
        continuity = True
    elif coords_ok and len(blocks) > 1:
        for i, b in enumerate(blocks[:-1]):
            nb = blocks[i + 1]
            d_pair, d_same = [], []
            for w in range(K):
                if col[b][w] < 0 or col[nb][w] < 0:
                    continue
                a = np.asarray(pos[b][-1, col[b][w]][ca], dtype=np.float64)
                p = np.asarray(pos[nb][0, col[nb][w]][ca], dtype=np.float64)
                d_pair.append(np.sqrt(np.mean(np.sum((a - p) ** 2, axis=1))))
                if col[b][w] < states[nb].shape[1]:
                    q = np.asarray(pos[nb][0, col[b][w]][ca], dtype=np.float64)
                    d_same.append(np.sqrt(np.mean(np.sum((a - q) ** 2, axis=1))))
            boundary_checks.append({'from': b, 'to': nb, 'pairing_rmsd_nm_max': float(np.max(d_pair)),
                                    'same_column_rmsd_nm_mean': float(np.mean(d_same)) if d_same else None})
        continuity = bool(all(x['pairing_rmsd_nm_max'] < 0.01 for x in boundary_checks))
        log(f'walker continuity across resumes: {continuity}')
    if continuity is False:
        # no physical walker crosses a resume: columns are replica slots, every block its own segment,
        # drawn and analysed within blocks only (the line lifts at each block start)
        walk, col = build_walk(pairing=False)
        for b in blocks[1:]:
            walk[row_start[b], :] = np.nan
    # Replicas that start a block from one identical configuration accept any swap on the first exchange
    # round (every pair has zero energy difference), which scatters them over the whole ladder: rep3 moves
    # 40 rungs on average at that step and 1.8 afterwards. Such a step is a restart artefact, not mixing;
    # it is detected per block and excluded from every walker statistic.
    scramble_blocks = []
    for b in blocks:
        d = np.abs(np.diff(states[b].astype(float), axis=0)).mean(axis=1)
        if len(d) > 2 and d[0] > 2.0 and d[0] > 3.0 * np.median(d[1:]):
            scramble_blocks.append(b)
            walk[row_start[b], :] = np.nan
    if scramble_blocks:
        log(f'restart scramble on the first exchange step of blocks {scramble_blocks}; those steps are excluded')
    same_block = (block_of_row[1:] == block_of_row[:-1])

    walkers = []
    segments = []                                     # per (slot, block) segment; the unit of analysis without continuity
    for w in range(K):
        x = walk[:, w]
        ok = ~np.isnan(x)
        xs = x[ok]
        d1 = np.abs(x[1:] - x[:-1])                   # NaN across any lifted pen, so cross-block steps never count
        steps = d1[~np.isnan(d1)]
        rec = {'walker': w, 'start_rung': int(xs[0]), 'end_rung': int(xs[-1]),
               'min_rung': int(xs.min()), 'max_rung': int(xs.max()), 'n_distinct': int(len(np.unique(xs))),
               'iterations': int(ok.sum()), 'mean_abs_step': float(steps.mean()) if len(steps) else None,
               'max_step': int(steps.max()) if len(steps) else None,
               'reached_bottom': bool(xs.min() == 0), 'reached_top': bool(xs.max() == K - 1)}
        if continuity is False:
            # a slot is re-placed at random by every restart scramble, so end-touch flags and whole-run
            # ranges say nothing about mixing; keep the within-block ranges instead
            rngs = []
            for b in blocks:
                seg = x[row_start[b]:row_start[b] + states[b].shape[0]]
                seg = seg[~np.isnan(seg)]
                if len(seg) > 1:
                    rngs.append(int(seg.max() - seg.min()))
                    segments.append({'walker': w, 'block': b, 'range': int(seg.max() - seg.min()), 'n': int(len(seg))})
            rec.update({'reached_bottom': False, 'reached_top': False, 'block_range_median': float(np.median(rngs)) if rngs else None})
        walkers.append(rec)

    lags = [l for l in (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 500) if l < n_rows]
    msd = {}
    for lag in lags:
        d = walk[lag:] - walk[:-lag]
        if continuity is False:                        # only pairs inside one block
            d = d[block_of_row[lag:] == block_of_row[:-lag]]
        msd[lag] = float(np.nanmean(d ** 2)) if np.any(~np.isnan(d)) else None
    short = [msd[l] / (2 * l) for l in (1, 2, 5) if l in msd and msd[l] is not None]
    longer = [msd[l] / (2 * l) for l in (20, 50, 100) if l in msd and msd[l] is not None]
    D_short = float(np.mean(short)) if short else None
    D_long = float(np.mean(longer)) if longer else D_short
    dT = float((T_final[-1] - T_final[0]) / (K - 1))
    rt = {'D_short': D_short, 'D_long': D_long, 'rung_spacing_K': dT, 'D_long_K2': D_long * dT ** 2 if D_long else None,
          'half_leg_iterations': float((K - 1) ** 2 / (2 * D_long)) if D_long else None,
          'round_trip_iterations': float((K - 1) ** 2 / D_long) if D_long else None}

    # ------------------------------------------------------------ per-block mixing
    mixing_rows = []
    for i, b in enumerate(blocks):
        s = states[b]
        apath = os.path.join(R, 'analysis', 'ladder', f'{b:04d}.json')
        acc = None
        if os.path.exists(apath):
            acc = json.load(open(apath))
        elif L is not None:
            try:
                acc = L.block_audit(R, b, threshold=0.30)
            except Exception as exc:
                log(f'block {b}: ladder audit failed ({exc})')
        A = np.array(acc['A_pool']) if acc else None
        upto = walk[:row_start[b] + s.shape[0]]
        blk = walk[row_start[b]:row_start[b] + s.shape[0]]
        with np.errstate(all='ignore'):
            cum_bot = int(np.nansum(np.nanmin(upto, axis=0) == 0)); cum_top = int(np.nansum(np.nanmax(upto, axis=0) == K - 1))
            blk_bot = int(np.nansum(np.nanmin(blk, axis=0) == 0)); blk_top = int(np.nansum(np.nanmax(blk, axis=0) == K - 1))
        mixing_rows.append({
            'block': b, 'rungs': int(s.shape[1]), 'exchange_iterations': int(s.shape[0] - 1),
            'A_mean': float(A.mean()) if A is not None else None, 'A_min': float(A.min()) if A is not None else None,
            'A_max': float(A.max()) if A is not None else None,
            'pairs_below_030': int((A < 0.30).sum()) if A is not None else None,
            'rung_change_fraction': float(M.acceptance_from_states(s)),
            'tau2_block': float(M.mixing_time(M.transition_matrix(s))),
            'tau2_cumulative': counts_tau2(upto, M.mixing_time),
            'walkers_bottom_block': blk_bot if continuity is not False else None,
            'walkers_bottom_cum': cum_bot if continuity is not False else None,
            'walkers_top_block': blk_top if continuity is not False else None,
            'walkers_top_cum': cum_top if continuity is not False else None,
            'half_legs_cum': half_legs(upto, K) if continuity is not False else None,
        })

    # ------------------------------------------------------------ resampled sets
    log_ess = parse_log_ess(logs)
    frame_src = []
    for b in available:
        for it in range(args.skip, states[b].shape[0]):
            frame_src.append((b, it))
    frame_src = np.array(frame_src)

    def source_tuples(inds, b_eval):
        out = np.empty((len(inds), 3), dtype=int)
        for i, (s, f) in enumerate(inds):
            bb, it = frame_src[f]
            T_s = temps[b_eval][s]                      # the state index refers to the ladder of the evaluated block
            k = int(np.argmin(np.abs(temps[bb] - T_s)))
            out[i] = (bb, it, int(np.where(states[bb][it] == k)[0][0]))
        return out

    def offline_resample(b):
        from chimpss.fultonmarket.analysis import FultonMarketAnalysis
        upper = sum(max(0, states[x].shape[0] - args.skip) for x in available if x <= b) - 1
        an = FultonMarketAnalysis(input_dir=R, pdb=ref_path, upper_limit=upper, verbosity='none')
        an.importance_resampling(n_samples=1000)
        w0 = np.asarray(an.weights[:, 0], dtype=np.float64)
        w0 = w0 / w0.sum()
        return {'inds': np.asarray(an.resampled_inds), 'ess': float(1.0 / np.sum(w0 ** 2)), 'pool': int(len(w0)),
                't0': int(an.t0), 'n_frames_total': int(an.energies.shape[0])}

    sets = {}
    per_frame = {}
    prev_unique = None
    for b in blocks:
        n_total = sum(max(0, states[x].shape[0] - args.skip) for x in available if x <= b)
        if os.path.exists(sv(b, 'resampled_indcs.npy')):
            inds = np.load(sv(b, 'resampled_indcs.npy'))
            meta = dict(log_ess.get(b, {}))
            meta['origin'] = 'engine'
            meta['t0'] = int(round(n_total * meta['discard_pct'] / 100.0)) if meta.get('discard_pct') is not None else None
        elif not args.no_offline:
            log(f'block {b}: no engine resample; regenerating offline')
            o = offline_resample(b)
            inds = o['inds']
            meta = {'ess': o['ess'], 'pool': o['pool'], 'pct': 100 * o['ess'] / o['pool'], 't0': o['t0'],
                    'origin': 'offline', 'discard_pct': 100 * o['t0'] / o['n_frames_total']}
        else:
            sets[b] = {'block': b, 'origin': 'none'}
            continue
        meta['n_frames_total'] = n_total
        # ESS == pool means pymbar returned uniform weights (the solve did not converge, typical of
        # 7-9 frames per state on 110-120-rung ladders): the set is an unweighted draw from every rung
        meta['uniform_weights'] = bool(meta.get('ess') is not None and meta.get('pool') and meta['ess'] >= 0.999 * meta['pool'])
        src = source_tuples(inds, b)
        uniq, inv, counts = np.unique(src, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        uniq_set = set(map(tuple, uniq))
        overlap = None if prev_unique is None else len(uniq_set & prev_unique) / len(uniq_set)
        H_u = np.array([enthalpy(bb, it, rep) for bb, it, rep in uniq])
        rec = {'block': b, **meta, 'n_draws': int(len(inds)), 'unique': int(len(uniq)),
               'max_multiplicity': int(counts.max()), 'share_top10_frames': float(np.sort(counts)[-10:].sum() / counts.sum()),
               'share_from_newest_block': float((src[:, 0] == b).mean()),
               'by_source_block': {int(x): float((src[:, 0] == x).mean()) for x in np.unique(src[:, 0])},
               'by_state_T': {f'{temps[b][int(x)]:.1f}': float((inds[:, 0] == x).mean()) for x in np.unique(inds[:, 0])},
               'overlap_with_previous_set': overlap,
               'H_mean': float(H_u[inv].mean()), 'H_std': float(H_u[inv].std())}
        pf = {'H': H_u[inv]}
        if coords_ok:
            xyz_u = np.empty((len(uniq), len(sel), 3), dtype=np.float32)
            for i, (bb, it, rep) in enumerate(uniq):
                xyz_u[i] = pos[bb][it, rep][sel]
            tu = md.Trajectory(xyz_u, top_s)
            tu.superpose(ref_s, atom_indices=ca_s)
            ca_u = np.sqrt(np.mean(np.sum((tu.xyz[:, ca_s] - ref_s.xyz[0, ca_s]) ** 2, axis=2), axis=1)) * 10.0
            pw = np.array([md.rmsd(tu, tu, i, atom_indices=ca_s) for i in range(len(uniq))]) * 10.0
            wmat = np.outer(counts, counts).astype(float)
            np.fill_diagonal(wmat, 0.0)
            rec['pairwise_ca_width'] = float((pw * wmat).sum() / wmat.sum()) if wmat.sum() > 0 else None
            rec['ca_rmsd_mean'] = float(ca_u[inv].mean()); rec['ca_rmsd_std'] = float(ca_u[inv].std())
            rec['ca_rmsd_p95'] = float(np.percentile(ca_u[inv], 95))
            pf['ca'] = ca_u[inv]
            if len(lig):
                lig_u = np.sqrt(np.mean(np.sum((tu.xyz[:, lig_s] - ref_s.xyz[0, lig_s]) ** 2, axis=2), axis=1)) * 10.0
                rec['lig_rmsd_mean'] = float(lig_u[inv].mean()); rec['lig_rmsd_std'] = float(lig_u[inv].std())
                rec['lig_rmsd_p95'] = float(np.percentile(lig_u[inv], 95))
                pf['lig'] = lig_u[inv]
            if os.path.exists(sv(b, 'resampled_trj.dcd')) and args.n_check > 0:
                pick = rng.choice(len(inds), min(args.n_check, len(inds)), replace=False)
                frames = [md.load_frame(sv(b, 'resampled_trj.dcd'), int(i), top=sv(b, 'resampled_top.pdb'), atom_indices=sel) for i in pick]
                t_d = frames[0].join(frames[1:]) if len(frames) > 1 else frames[0]
                mine = md.Trajectory(np.array([xyz_u[inv[i]] for i in pick], dtype=np.float32), top_s)
                d = np.array([md.rmsd(mine, t_d, i, atom_indices=ca_s)[i] for i in range(len(pick))]) * 10.0
                rec['dcd_check_max_rmsd_A'] = float(d.max())
        sets[b] = rec
        per_frame[b] = pf
        prev_unique = uniq_set
        log(f"block {b}: {rec['origin']} ESS {rec.get('ess')} / {rec.get('pool')}; unique {rec['unique']}; "
            f"CA {rec.get('ca_rmsd_mean')}; lig {rec.get('lig_rmsd_mean')}")

    pool = {}
    for b in sets:
        if sets[b].get('uniform_weights'):
            sets[b]['origin'] = 'uniform'          # an unweighted draw from every rung, not a 310 K set
    for q in ('ca', 'lig', 'H'):
        vals = [per_frame[b][q] for b in per_frame if q in per_frame[b] and not sets[b].get('uniform_weights')]
        if vals:
            allv = np.concatenate(vals)
            pool[q + '_mean'] = float(allv.mean()); pool[q + '_std'] = float(allv.std()); pool['n'] = int(len(allv))
    for b, s in sets.items():
        for q, key in (('ca', 'ca_rmsd_mean'), ('lig', 'lig_rmsd_mean'), ('H', 'H_mean')):
            if key in s and q + '_std' in pool and pool[q + '_std'] > 0 and not s.get('uniform_weights'):
                s['z_' + q] = (s[key] - pool[q + '_mean']) / pool[q + '_std']

    # ------------------------------------------------------------ chart data
    cw = 1 if K <= len(ALPHABET) else 2

    def enc(v):
        if np.isnan(v):
            return '-' * cw
        v = int(v)
        return ALPHABET[v] if cw == 1 else ALPHABET[v // len(ALPHABET)] + ALPHABET[v % len(ALPHABET)]

    walk_rows = []
    for w in walkers:
        t0 = round(float(T_final[w['start_rung']]), 1); t1 = round(float(T_final[w['end_rung']]), 1)
        lab = f"{w['walker']}: {round(t0)}\u2192{round(t1)} K" if continuity is not False else f"slot {w['walker']}: {round(t0)} K"
        walk_rows.append({'w': w['walker'], 't0': t0, 't1': t1, 'lab': lab,
                          'lo': w['min_rung'], 'hi': w['max_rung'], 'b': int(w['reached_bottom']), 'u': int(w['reached_top']),
                          'cw': cw, 'p': ''.join(enc(v) for v in walk[:, w['walker']])})
    series_rows = []
    for b in blocks:
        s = sets[b]
        if s.get('origin') in (None, 'none'):
            continue
        row = {'block': b, 'origin': s['origin'], 'draws': s['n_draws'], 'dH': round(s['H_mean'] - pool.get('H_mean', 0.0)), 'hSd': round(s['H_std']),
               'ess': round(s['ess'], 1) if s.get('ess') is not None else None, 'unique': s['unique']}
        if 'ca_rmsd_mean' in s:
            row.update({'ca': round(s['ca_rmsd_mean'], 2), 'caSd': round(s['ca_rmsd_std'], 2)})
        if 'lig_rmsd_mean' in s:
            row.update({'lig': round(s['lig_rmsd_mean'], 2), 'ligSd': round(s['lig_rmsd_std'], 2)})
        series_rows.append(row)
    pool_row = {'caMean': round(pool.get('ca_mean', 0), 2), 'caSd': round(pool.get('ca_std', 0), 2),
                'ligMean': round(pool.get('lig_mean', 0), 2), 'ligSd': round(pool.get('lig_std', 0), 2),
                'hMean': 0, 'hSd': round(pool.get('H_std', 0))}

    result = {'run': R, 'run_name': run_name, 'ref': ref_path, 'ligand': ligand, 'iter_ps': iter_ps, 'skip': args.skip,
              'blocks': blocks, 'K': K, 'T_final': T_final.tolist(), 'row_start': row_start, 'n_rows': n_rows,
              'walker_continuity': continuity, 'boundary_checks': boundary_checks, 'scramble_blocks': scramble_blocks,
              'walkers': walkers, 'segments': segments, 'msd': msd,
              'round_trip': rt, 'mixing': mixing_rows, 'sets': sets, 'pool': pool,
              'chart_data': {'walks': walk_rows, 'series': series_rows, 'pool': pool_row,
                             'block_starts': [row_start[b] for b in blocks]},
              'elapsed_s': time.time() - t_start}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log(f'wrote {out_path} in {time.time() - t_start:.0f} s')

    # ------------------------------------------------------------ markdown summary
    n_iter = sum(r['exchange_iterations'] for r in mixing_rows)
    ns_rep = n_iter * iter_ps / 1000.0 if iter_ps else None
    agg = sum(r['exchange_iterations'] * r['rungs'] for r in mixing_rows) * iter_ps / 1000.0 if iter_ps else None
    f = lambda v, d=3: '' if v is None else f'{v:.{d}f}'
    blank = lambda v: '' if v is None else v
    print(f'# {run_name}: {len(blocks)} blocks, {K} rungs {T_final[0]:.1f}-{T_final[-1]:.1f} K, {n_iter} exchange iterations'
          + (f', {ns_rep:.2f} ns per replica, {agg:.0f} ns aggregate' if ns_rep else ''))
    if continuity is False:
        print('Walker continuity across resumes: FALSE. Columns are replica slots and every block is an independent segment; '
              'walker counts, ranges, MSD and half-legs below are within-block quantities and cumulative counts are blank.')
    if scramble_blocks:
        print(f'Restart scramble: the first exchange step of blocks {scramble_blocks} moved replicas across the whole ladder '
              f'(identical start configurations accept every swap); those steps are excluded from all walker statistics.')
    print(f'walker continuity across resumes verified on coordinates: {continuity}'
          + (f" (max pairing RMSD {max(x['pairing_rmsd_nm_max'] for x in boundary_checks):.4f} nm; same-column mean "
             f"{np.mean([x['same_column_rmsd_nm_mean'] for x in boundary_checks if x['same_column_rmsd_nm_mean'] is not None]):.2f} nm)" if boundary_checks else ''))
    print('\n## Mixing\n')
    print('| Block | Rungs | Exchange iterations | Acceptance mean (min to max) | Rung changes per replica-iteration | τ₂ block (iterations) | τ₂ cumulative (iterations) | Walkers at bottom (block / cumulative) | Walkers at top (block / cumulative) | Half-legs (cumulative) |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in mixing_rows:
        acc = f"{f(r['A_mean'])} ({f(r['A_min'])} to {f(r['A_max'])})" if r['A_mean'] is not None else 'n/a'
        print(f"| {r['block']} | {r['rungs']} | {r['exchange_iterations']} | {acc} | {r['rung_change_fraction']:.3f} | {r['tau2_block']:.0f} | "
              f"{r['tau2_cumulative']:.0f} | {blank(r['walkers_bottom_block'])} / {blank(r['walkers_bottom_cum'])} | "
              f"{blank(r['walkers_top_block'])} / {blank(r['walkers_top_cum'])} | {blank(r['half_legs_cum'])} |")
    print(f"\nMSD by lag (rung²): " + ', '.join(f'{l}: {v:.2f}' for l, v in msd.items() if v is not None))
    if rt['D_long']:
        print(f"Diffusion D = {rt['D_long']:.3f} rung² per iteration = {rt['D_long_K2']:.3f} K² per iteration at {dT:.3f} K per rung "
              f"(short-lag {rt['D_short']:.3f} rung²); half-leg ≈ {rt['half_leg_iterations']:.0f} iterations, "
              f"round trip ≈ {rt['round_trip_iterations']:.0f} iterations"
              + (f" ≈ {rt['round_trip_iterations'] * iter_ps / 1000:.0f} ns per replica" if iter_ps else ''))
    rngs = [w['max_rung'] - w['min_rung'] for w in walkers]
    if continuity is False:
        sr = [g['range'] for g in segments]
        print(f"Segments: {len(segments)} slot-block segments of {int(np.median([g['n'] for g in segments]))} saved frames; within-block rung range "
              f"median {np.median(sr):.0f} (5th to 95th percentile {np.percentile(sr, 5):.0f} to {np.percentile(sr, 95):.0f}); "
              f"mean |step| {np.mean([w['mean_abs_step'] for w in walkers if w['mean_abs_step'] is not None]):.2f}, "
              f"max step {max(w['max_step'] for w in walkers if w['max_step'] is not None)}. End-touch counts are not reported: "
              f"the restart scramble places slots at random.")
    else:
      print(f"Walkers: {sum(w['reached_bottom'] for w in walkers)} of {K} touched rung 0, {sum(w['reached_top'] for w in walkers)} touched rung {K - 1}, "
          f"{sum(w['reached_bottom'] and w['reached_top'] for w in walkers)} both; rung range median {np.median(rngs):.0f} (min {min(rngs)}, max {max(rngs)}); "
          f"distinct rungs median {np.median([w['n_distinct'] for w in walkers]):.0f}; mean |step| {np.mean([w['mean_abs_step'] for w in walkers if w['mean_abs_step'] is not None]):.2f}, "
          f"max step {max(w['max_step'] for w in walkers if w['max_step'] is not None)}")
    print('\n## Resampled-pool series (mean ± std over the draws; z against the pooled reference)\n')
    print('| Block | Resample | Cα RMSD (Å) | Ligand RMSD (Å) | Enthalpy − pooled mean (kJ/mol) | Pairwise Cα width (Å) | z Cα | z ligand | z H |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for b in blocks:
        s = sets[b]
        if s.get('origin') in (None, 'none'):
            print(f'| {b} | none | | | | | | | |'); continue
        print(f"| {b} | {s['origin']} | {f(s.get('ca_rmsd_mean'), 2)} ± {f(s.get('ca_rmsd_std'), 2)} | {f(s.get('lig_rmsd_mean'), 2)} ± {f(s.get('lig_rmsd_std'), 2)} | "
              f"{s['H_mean'] - pool.get('H_mean', 0):+.0f} ± {s['H_std']:.0f} | {f(s.get('pairwise_ca_width'), 2)} | {f(s.get('z_ca'), 2)} | {f(s.get('z_lig'), 2)} | {f(s.get('z_H'), 2)} |")
    print(f"\nPooled reference (uniform sets excluded) over {pool.get('n')} draws: Cα {f(pool.get('ca_mean'), 2)} ± {f(pool.get('ca_std'), 2)} Å, ligand {f(pool.get('lig_mean'), 2)} ± {f(pool.get('lig_std'), 2)} Å, "
          f"H {f(pool.get('H_mean'), 0)} ± {f(pool.get('H_std'), 0)} kJ/mol")
    print('\n## Resampling health\n')
    print('| Block | Resample | Pool frames | Discarded to equilibration | ESS (frames) | ESS (% of pool) | Distinct frames in draws | Most-drawn frame | Draws from newest block | Distinct frames shared with previous set | Draws by rung (K: share) | DCD check (Å) |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for b in blocks:
        s = sets[b]
        if s.get('origin') in (None, 'none'):
            print(f'| {b} | none | | | | | | | | | | |'); continue
        ov = '' if s['overlap_with_previous_set'] is None else f"{100 * s['overlap_with_previous_set']:.0f} %"
        byT = ', '.join(f'{k}: {100 * v:.0f} %' for k, v in list(s['by_state_T'].items())[:4])
        flag = ' (uniform: MBAR did not converge)' if s.get('uniform_weights') else ''
        print(f"| {b} | {s['origin']} | {s.get('pool', '')} | {f(s.get('discard_pct'), 1)} % | {f(s.get('ess'), 1)}{flag} | {f(s.get('pct'), 2)} | {s['unique']} | {s['max_multiplicity']} | "
              f"{100 * s['share_from_newest_block']:.0f} % | {ov} | {byT} | {f(s.get('dcd_check_max_rmsd_A'), 3)} |")


if __name__ == '__main__':
    main()
