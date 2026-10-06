#!/usr/bin/env python
"""Render the two report charts from mix_and_health.json into one-line claude.Visualize modules.

    python render_charts.py <run>/analysis/mix_and_health.json [--out-dir <run>/analysis/charts]
                            [--title-walks TEXT] [--subtitle-walks TEXT] [--title-series TEXT]

Writes state_walks.js and pool_series.js; each file is the complete `code` string of one widget
(one line, single quotes inside), ready to pass to the Claude Docs widget `create`.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(os.path.dirname(HERE), 'templates')


def js_literal(obj):
    """A compact JS literal with single-quoted strings (keys unquoted)."""
    if isinstance(obj, dict):
        return '{' + ','.join(f'{k}:{js_literal(v)}' for k, v in obj.items()) + '}'
    if isinstance(obj, list):
        return '[' + ','.join(js_literal(v) for v in obj) + ']'
    if isinstance(obj, bool):
        return 'true' if obj else 'false'
    if obj is None:
        return 'null'
    if isinstance(obj, (int, float)):
        return repr(obj)
    s = str(obj).replace('\\', '\\\\').replace("'", "\\'")
    return f"'{s}'"


def fill(template, mapping):
    out = template
    for k, v in mapping.items():
        out = out.replace(k, v)
    left = [t for t in ('__COMPONENT__', '__ROWS__', '__SERIES__', '__POOL__', '__PANELS__', '__TITLE__', '__SUBTITLE__',
                        '__LEGEND__', '__N_ROWS__', '__K__', '__T_MIN__', '__T_MAX__', '__BLOCK_STARTS__', '__MODE__') if t in out]
    assert not left, f'unfilled placeholders: {left}'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json')
    ap.add_argument('--out-dir')
    ap.add_argument('--title-walks', help='typed title of the walks chart (default depends on walker continuity)')
    ap.add_argument('--subtitle-walks', help='typed caption of the walks chart (default depends on walker continuity)')
    ap.add_argument('--title-series', default='Resampled target-temperature ensemble per block: structure and energy')
    args = ap.parse_args()

    d = json.load(open(args.json))
    cd = d['chart_data']
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.json)), 'charts')
    os.makedirs(out_dir, exist_ok=True)
    slug = d['run_name'].replace('.', '-').replace('_', '-').lower()
    T = d['T_final']
    continuous = d.get('walker_continuity') is not False
    title_walks = args.title_walks or ('Walkers diffuse along the ladder; those that touched the bottom rung are blue' if continuous
                                       else 'Replica slots per block: every block restarts from the slot rung (no walker continuity across resumes)')
    subtitle_walks = args.subtitle_walks or ('Panels ordered by starting replica; bottom rung at the bottom edge, top rung at the top; saved iterations left to right; ticks mark block starts' if continuous
                                             else 'One panel per replica slot; each block is a separate segment; bottom rung at the bottom edge, top rung at the top; ticks mark block starts')

    walk_rows = [dict(w) for w in cd['walks']]
    by_w = {w['walker']: w for w in d.get('walkers', [])}
    seg_by_w = {}
    for g in d.get('segments', []):
        seg_by_w.setdefault(g['walker'], []).append(g['range'])
    for w in walk_rows:
        info = by_w.get(w['w'], {})
        if continuous:
            w['lab'] = f"{w['w']}: {round(w['t0'])}\u2192{round(w['t1'])} K"
            w['hov'] = f"walker {w['w']}: {w['t0']} K to {w['t1']} K, rungs {w['lo']} to {w['hi']}"
        else:
            rng = sorted(seg_by_w.get(w['w'], []))
            med = rng[len(rng) // 2] if rng else 0
            w['lab'] = f"slot {w['w']}"
            w['hov'] = f"slot {w['w']}: {len(rng)} block segments, within-block rung range median {med}"
            w['b'] = 0; w['u'] = 0
    walks = open(os.path.join(TPL, 'state_walks_widget.js')).read().strip()
    walks = fill(walks, {
        '__COMPONENT__': f'{slug}-state-walks', '__ROWS__': js_literal(walk_rows),
        '__N_ROWS__': str(d['n_rows']), '__K__': str(d['K']), '__T_MIN__': f'{T[0]:.1f}', '__T_MAX__': f'{T[-1]:.1f}',
        '__BLOCK_STARTS__': ','.join(str(x) for x in cd['block_starts']), '__MODE__': 'walkers' if continuous else 'slots',
        '__TITLE__': title_walks, '__SUBTITLE__': subtitle_walks,
    })
    open(os.path.join(out_dir, 'state_walks.js'), 'w').write(walks + '\n')

    rows = [dict(r) for r in cd['series'] if r.get('origin') != 'uniform']
    n_uniform = sum(1 for r in cd['series'] if r.get('origin') == 'uniform')
    if not rows:
        print(f'wrote {out_dir}/state_walks.js ({len(walks)} chars); no weighted resampled set in the JSON, so no series chart')
        return
    for r in rows:
        r.setdefault('draws', 1000)
        r.setdefault('ca', 0.0); r.setdefault('caSd', 0.0)
    has_lig = all('lig' in r for r in rows)
    panels = [{'key': 'ca', 'label': 'Alpha-carbon RMSD to the equilibrated start (Å)', 'unit': 'Å'}]
    if has_lig:
        panels.append({'key': 'lig', 'label': 'Ligand heavy-atom RMSD after Cα superposition (Å)', 'unit': 'Å'})
    panels.append({'key': 'dh', 'label': 'Frame enthalpy minus the pooled mean (kJ/mol)', 'unit': 'kJ/mol'})
    pool = dict(cd['pool'])
    if not has_lig:
        pool['ligMean'] = 0; pool['ligSd'] = 0
    series = open(os.path.join(TPL, 'pool_series_widget.js')).read().strip()
    series = fill(series, {
        '__COMPONENT__': f'{slug}-resampled-pool-series', '__SERIES__': js_literal(rows), '__POOL__': js_literal(pool),
        '__PANELS__': js_literal(panels), '__TITLE__': args.title_series,
        '__LEGEND__': 'Band, dashed line: pooled mean \u00b1 std of weighted sets. Filled: engine resample; hollow: computed offline'
                      + (f'; {n_uniform} uniform-draw blocks omitted' if n_uniform else ''),
    })
    open(os.path.join(out_dir, 'pool_series.js'), 'w').write(series + '\n')
    print(f'wrote {out_dir}/state_walks.js ({len(walks)} chars) and {out_dir}/pool_series.js ({len(series)} chars)')


if __name__ == '__main__':
    main()
