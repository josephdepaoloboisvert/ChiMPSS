r"""Inline-SVG figures for the convergence method-assessment report.

Every function returns an SVG fragment as a string, sized to a 720-unit wide
viewBox and drawing its colours from CSS custom properties -- ``var(--ink)``,
``var(--accent)``, ``var(--burnt)``, ``var(--fail)``, ``var(--rule-soft)`` and
so on -- so a single page stylesheet controls light and dark rendering for all
of them.  Nothing here loads a plotting library or writes a raster image: the
figures are text, they diff, and they inherit the page's theme.

Why SVG rather than matplotlib
------------------------------
The report is published as an HTML artifact whose content-security policy
blocks external images, and a raster chart cannot follow the viewer's theme.
Hand-built SVG solves both and costs little, because these are simple charts.

The one recurring bug
---------------------
A mark placed outside the ``viewBox`` is silently clipped -- it does not raise,
it just vanishes, and the chart looks plausible without it.  An earlier version
of this report drew thirty JSD points above the top of the plot that way.
:func:`chimpss.analysis.report_html.check_svg_bounds` re-parses every emitted
figure and fails on any coordinate outside the box; run it before publishing.
"""
import numpy as np

__all__ = [
    'MONO', 'SANS', 'VIRIDIS', 'viridis',
    'pairwise_hist_svg', 'series_svg', 'pool_svg', 'roundtrip_hist_svg',
    'walkers_svg', 'pc_landscape_svg', 'drift_svg',
    'segmented_drift_svg', 'window_threshold_svg',
]

MONO = 'IBM Plex Mono, monospace'
SANS = 'IBM Plex Sans, sans-serif'

#: Sampled viridis, enough resolution for a time ramp.
VIRIDIS = ['#440154', '#471365', '#482475', '#463480', '#414487', '#3b528b',
           '#355f8d', '#2f6c8e', '#2a788e', '#25848e', '#21918c', '#1e9c89',
           '#22a884', '#2fb47c', '#44bf70', '#5ec962', '#7ad151', '#9bd93c',
           '#bddf26', '#dfe318', '#fde725']


def viridis(f):
    """Colour for a fraction in [0, 1] along the viridis ramp."""
    f = min(max(float(f), 0.0), 1.0)
    return VIRIDIS[int(round(f * (len(VIRIDIS) - 1)))]


def _axes(out, L, R, T, B, W, H, xt, yt, sx, sy, xlab, ylab, xfmt='{:.0f}', yfmt='{:.2f}'):
    out.append('<g stroke="var(--rule-soft)" stroke-width="1">')
    for v in yt:
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}"></line>')
    out.append('</g>')
    out.append(f'<g fill="var(--muted)" font-family="{MONO}" font-size="11">')
    for v in xt:
        out.append(f'<text x="{sx(v):.1f}" y="{H-B+16}" text-anchor="middle">{xfmt.format(v)}</text>')
    for v in yt:
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{yfmt.format(v)}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-5}" fill="var(--ink-2)" '
               f'font-family="{SANS}" font-size="12" text-anchor="middle">{xlab}</text>')
    out.append(f'<text x="13" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" font-family="{SANS}" '
               f'font-size="12" text-anchor="middle" '
               f'transform="rotate(-90 13 {(T+H-B)/2:.0f})">{ylab}</text>')


def pairwise_hist_svg(dist, thresh=0.10):
    W, H, L, R, T, B = 720, 330, 62, 150, 20, 44
    series = [('torsion', 'var(--accent)'), ('alpha_carbon', 'var(--burnt)'),
              ('contact', '#d8a13a')]
    allv = np.concatenate([np.array(dist[k]['values']) for k, _ in series])
    hi = float(np.percentile(allv, 99.5))
    edges = np.linspace(0, hi, 46)
    dens = {}
    for k, _ in series:
        v = np.array(dist[k]['values'])
        h, _ = np.histogram(np.clip(v, 0, hi), bins=edges, density=True)
        dens[k] = h
    ymax = max(h.max() for h in dens.values()) * 1.08
    sx = lambda v: L + v / hi * (W - L - R)
    sy = lambda v: H - B - v / ymax * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Distributions of '
           f'Jensen-Shannon distance over all block pairs for the torsion, '
           f'alpha-carbon and contact metrics. All three distributions sit well '
           f'to the right of the 0.10 threshold, so the median pair fails every '
           f'metric.">']
    # threshold band
    out.append(f'<rect x="{L}" y="{T}" width="{sx(thresh)-L:.1f}" height="{H-B-T}" '
               f'fill="var(--accent-soft)" fill-opacity="0.55"></rect>')
    _axes(out, L, R, T, B, W, H, np.linspace(0, hi, 6), np.linspace(0, ymax, 5),
          sx, sy, 'Jensen&#8211;Shannon distance between block pairs', 'density',
          '{:.2f}', '{:.0f}')
    out.append(f'<line x1="{sx(thresh):.1f}" y1="{T}" x2="{sx(thresh):.1f}" y2="{H-B}" '
               f'stroke="var(--ink)" stroke-width="1.2" stroke-dasharray="4 3"></line>')
    out.append(f'<text x="{sx(thresh)+5:.1f}" y="{T+12}" fill="var(--ink)" '
               f'font-family="{MONO}" font-size="10.5">threshold 0.10</text>')
    for k, col in series:
        h = dens[k]
        pts = []
        for i, val in enumerate(h):
            pts.append(f'{sx(edges[i]):.1f},{sy(val):.1f}')
            pts.append(f'{sx(edges[i+1]):.1f},{sy(val):.1f}')
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" '
                   f'stroke-width="1.8"></polyline>')
        med = dist[k]['median']
        out.append(f'<line x1="{sx(med):.1f}" y1="{sy(0):.1f}" x2="{sx(med):.1f}" '
                   f'y2="{sy(ymax*0.12):.1f}" stroke="{col}" stroke-width="2.2"></line>')
    lx, ly = W - R + 12, T + 14
    names = {'torsion': 'torsion', 'alpha_carbon': 'alpha-carbon', 'contact': 'contact'}
    for i, (k, col) in enumerate(series):
        y = ly + i * 34
        out.append(f'<line x1="{lx}" y1="{y-4}" x2="{lx+14}" y2="{y-4}" stroke="{col}" stroke-width="2.4"></line>')
        out.append(f'<text x="{lx+19}" y="{y}" fill="var(--ink-2)" font-family="{SANS}" '
                   f'font-size="11">{names[k]}</text>')
        out.append(f'<text x="{lx}" y="{y+13}" fill="var(--muted)" font-family="{MONO}" '
                   f'font-size="10">med {dist[k]["median"]:.3f}</text>')
        out.append(f'<text x="{lx}" y="{y+24}" fill="var(--muted)" font-family="{MONO}" '
                   f'font-size="10">pass {100*dist[k]["pass_frac"]:.1f}%</text>')
    out.append('</svg>')
    return '\n      '.join(out)


def series_svg(x, y, ylab, xlab='block', band=None, band_label=None,
               quarters=None, H=300, ylim=None, color='var(--ink-2)',
               aria='', point_color=None, hline=None, hline_label=None):
    W, L, R, T, B = 720, 62, 118, 20, 44
    y = np.asarray(y, float); x = np.asarray(x, float)
    lo, hi = (ylim if ylim else (min(y.min(), 0 if band is None else 0), y.max()))
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    sx = lambda v: L + (v - x.min()) / (x.max() - x.min()) * (W - L - R)
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}">']
    if band is not None:
        out.append(f'<rect x="{L}" y="{sy(band):.1f}" width="{W-L-R}" '
                   f'height="{H-B-sy(band):.1f}" fill="var(--accent-soft)" fill-opacity="0.7"></rect>')
    _axes(out, L, R, T, B, W, H, np.arange(0, x.max()+1, 20), np.linspace(lo, hi, 5),
          sx, sy, xlab, ylab, '{:.0f}', '{:.2f}')
    if band is not None:
        out.append(f'<line x1="{L}" y1="{sy(band):.1f}" x2="{W-R}" y2="{sy(band):.1f}" '
                   f'stroke="var(--accent)" stroke-width="1.1" stroke-dasharray="4 3"></line>')
        out.append(f'<text x="{W-R+6}" y="{sy(band)+4:.1f}" fill="var(--accent)" '
                   f'font-family="{SANS}" font-size="10.5">{band_label}</text>')
    if hline is not None:
        out.append(f'<line x1="{L}" y1="{sy(hline):.1f}" x2="{W-R}" y2="{sy(hline):.1f}" '
                   f'stroke="var(--fail)" stroke-width="1.1" stroke-dasharray="5 3"></line>')
        out.append(f'<text x="{W-R+6}" y="{sy(hline)+4:.1f}" fill="var(--fail)" '
                   f'font-family="{SANS}" font-size="10.5">{hline_label}</text>')
    if quarters:
        n = len(x); seg = np.array_split(np.arange(n), len(quarters))
        for q, idx in zip(quarters, seg):
            out.append(f'<line x1="{sx(x[idx[0]]):.1f}" y1="{sy(q):.1f}" '
                       f'x2="{sx(x[idx[-1]]):.1f}" y2="{sy(q):.1f}" '
                       f'stroke="var(--burnt)" stroke-width="2.4" stroke-opacity="0.85"></line>')
            out.append(f'<text x="{sx(x[idx[len(idx)//2]]):.1f}" y="{sy(q)-7:.1f}" '
                       f'fill="var(--burnt)" font-family="{MONO}" font-size="10" '
                       f'text-anchor="middle">{q:.3f}</text>')
    out.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy(b):.1f}" for a,b in zip(x,y))}" '
               f'fill="none" stroke="{color}" stroke-width="1.4" stroke-opacity="0.9"></polyline>')
    for a, b in zip(x, y):
        c = point_color(a, b) if point_color else color
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2.2" fill="{c}"></circle>')
    out.append('</svg>')
    return '\n      '.join(out)


def pool_svg(pool, collapse_at=10, small_pool=35):
    """Unique resampled frames per block (log scale) against the pool it drew from."""
    W, H, L, R, T, B = 720, 340, 62, 128, 20, 44
    b = np.array([p[0] for p in pool], float)
    size = np.array([p[1] for p in pool], float)
    uniq = np.array([p[2] for p in pool], float)
    lg = lambda v: np.log10(np.maximum(v, 1.0))
    lo, hi = 0.0, lg(max(uniq.max(), size.max())) * 1.04
    sx = lambda v: L + (v - b.min()) / (b.max() - b.min()) * (W - L - R)
    sy = lambda v: H - B - (lg(v) - lo) / (hi - lo) * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Distinct '
           f'configurations drawn by the MBAR resample in each block, on a log '
           f'scale, against the size of the post-equilibration pool each block '
           f'drew from. Five blocks collapse to ten or fewer distinct frames, '
           f'and every one of them drew from a pool smaller than 35 frames.">']
    out.append('<g stroke="var(--rule-soft)" stroke-width="1">')
    # decades that actually fall inside the axis: a fixed [1, 10, 100, 1000]
    # silently draws gridlines and labels above the viewBox whenever the data
    # do not reach the top decade, and they are clipped without error.
    ticks = [t for t in (1, 10, 100, 1000, 10000, 100000) if lg(t) <= hi]
    for v in ticks:
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}"></line>')
    out.append('</g>')
    out.append(f'<rect x="{L}" y="{T}" width="{W-L-R}" height="{sy(collapse_at)-T:.1f}" '
               f'fill="none"></rect>')
    out.append(f'<rect x="{L}" y="{sy(collapse_at):.1f}" width="{W-L-R}" '
               f'height="{H-B-sy(collapse_at):.1f}" fill="var(--burnt-soft)" fill-opacity="0.75"></rect>')
    out.append(f'<g fill="var(--muted)" font-family="{MONO}" font-size="11">')
    for v in ticks:
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v}</text>')
    for v in range(0, int(b.max())+1, 20):
        out.append(f'<text x="{sx(v):.1f}" y="{H-B+16}" text-anchor="middle">{v}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-5}" fill="var(--ink-2)" '
               f'font-family="{SANS}" font-size="12" text-anchor="middle">block</text>')
    out.append(f'<text x="13" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" font-family="{SANS}" '
               f'font-size="12" text-anchor="middle" '
               f'transform="rotate(-90 13 {(T+H-B)/2:.0f})">frames (log)</text>')

    out.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy(v):.1f}" for a,v in zip(b,size))}" '
               f'fill="none" stroke="var(--accent)" stroke-width="1.3" stroke-opacity="0.8"></polyline>')
    out.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy(v):.1f}" for a,v in zip(b,uniq))}" '
               f'fill="none" stroke="var(--ink-2)" stroke-width="1.4"></polyline>')
    for a, v in zip(b, uniq):
        col = 'var(--fail)' if v <= collapse_at else 'var(--ink-2)'
        r = 3.6 if v <= collapse_at else 2.1
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(v):.1f}" r="{r}" fill="{col}"></circle>')
        if v <= collapse_at:
            out.append(f'<text x="{sx(a):.1f}" y="{sy(v)+16:.1f}" fill="var(--fail)" '
                       f'font-family="{MONO}" font-size="9.5" text-anchor="middle">{int(a)}</text>')
    lx, ly = W - R + 10, T + 16
    for i, (lab, col) in enumerate((('pool drawn from', 'var(--accent)'),
                                    ('distinct frames', 'var(--ink-2)'),
                                    ('collapsed (&#8804;10)', 'var(--fail)'))):
        out.append(f'<line x1="{lx}" y1="{ly+i*20-4}" x2="{lx+14}" y2="{ly+i*20-4}" '
                   f'stroke="{col}" stroke-width="2.4"></line>')
        out.append(f'<text x="{lx+19}" y="{ly+i*20}" fill="var(--ink-2)" '
                   f'font-family="{SANS}" font-size="10.5">{lab}</text>')
    out.append('</svg>')
    return '\n      '.join(out)


def roundtrip_hist_svg(per_walker):
    """Distribution of full ladder traversals per walker."""
    W, H, L, R, T, B = 720, 300, 62, 130, 22, 46
    pw = np.asarray(per_walker, int)
    mx_ = int(pw.max())
    counts = np.bincount(pw, minlength=mx_ + 1)
    ymax = counts.max() * 1.12
    bw = (W - L - R) / (mx_ + 1)
    sy = lambda v: H - B - v / ymax * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Number of full '
           f'ladder round trips completed by each of the {pw.size} walkers. '
           f'{int(counts[0])} walkers never completed one; the median walker '
           f'completed {int(np.median(pw))}.">']
    out.append('<g stroke="var(--rule-soft)" stroke-width="1">')
    yt = np.linspace(0, ymax, 5)
    for v in yt:
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}"></line>')
    out.append('</g>')
    for k, c in enumerate(counts):
        x = L + k * bw + bw * 0.16
        col = 'var(--fail)' if k == 0 else 'var(--accent)'
        out.append(f'<rect x="{x:.1f}" y="{sy(c):.1f}" width="{bw*0.68:.1f}" '
                   f'height="{H-B-sy(c):.1f}" fill="{col}" rx="1"></rect>')
        if c:
            out.append(f'<text x="{x+bw*0.34:.1f}" y="{sy(c)-5:.1f}" fill="var(--muted)" '
                       f'font-family="{MONO}" font-size="10" text-anchor="middle">{int(c)}</text>')
    out.append(f'<g fill="var(--muted)" font-family="{MONO}" font-size="11">')
    for k in range(mx_ + 1):
        out.append(f'<text x="{L+k*bw+bw*0.5:.1f}" y="{H-B+16}" text-anchor="middle">{k}</text>')
    for v in yt:
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-5}" fill="var(--ink-2)" '
               f'font-family="{SANS}" font-size="12" text-anchor="middle">'
               f'full round trips completed</text>')
    out.append(f'<text x="13" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" font-family="{SANS}" '
               f'font-size="12" text-anchor="middle" '
               f'transform="rotate(-90 13 {(T+H-B)/2:.0f})">walkers</text>')
    out.append(f'<text x="{W-R+10}" y="{T+16}" fill="var(--fail)" font-family="{SANS}" '
               f'font-size="10.5">{int(counts[0])} of {pw.size} walkers</text>')
    out.append(f'<text x="{W-R+10}" y="{T+30}" fill="var(--fail)" font-family="{SANS}" '
               f'font-size="10.5">never traversed</text>')
    out.append('</svg>')
    return '\n      '.join(out)


def walkers_svg(blocks, walkers, collapsed):
    """Walkers reaching the lowest rung per block, with collapsed blocks marked."""
    W, H, L, R, T, B = 720, 300, 62, 150, 22, 46
    b = np.asarray(blocks, float); w = np.asarray(walkers, float)
    lo, hi = w.min() - 1, w.max() + 1
    sx = lambda v: L + (v - b.min()) / (b.max() - b.min()) * (W - L - R)
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - T - B)
    med = float(np.median(w))
    order = list(b)
    cm = np.array([w[order.index(c)] for c in collapsed if c in order])

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Distinct walkers '
           f'reaching the 310 kelvin rung in each block. The five blocks whose '
           f'resample collapsed are marked and sit at or above the median, which '
           f'refutes the hypothesis that poor ladder mixing caused the collapses.">']
    _axes(out, L, R, T, B, W, H, np.arange(0, b.max()+1, 20),
          np.arange(int(lo)+1, int(hi)+1), sx, sy, 'block',
          'distinct walkers reaching 310 K', '{:.0f}', '{:.0f}')
    out.append(f'<line x1="{L}" y1="{sy(med):.1f}" x2="{W-R}" y2="{sy(med):.1f}" '
               f'stroke="var(--ink)" stroke-width="1" stroke-dasharray="4 3" stroke-opacity="0.6"></line>')
    out.append(f'<text x="{W-R+6}" y="{sy(med)+4:.1f}" fill="var(--muted)" '
               f'font-family="{SANS}" font-size="10.5">median {med:.0f}</text>')
    out.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy(v):.1f}" for a,v in zip(b,w))}" '
               f'fill="none" stroke="var(--accent)" stroke-width="1.3" stroke-opacity="0.85"></polyline>')
    for a, v in zip(b, w):
        if int(a) in collapsed:
            out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(v):.1f}" r="4.6" fill="var(--fail)"></circle>')
            out.append(f'<text x="{sx(a):.1f}" y="{sy(v)-10:.1f}" fill="var(--fail)" '
                       f'font-family="{MONO}" font-size="9.5" text-anchor="middle">{int(a)}</text>')
        else:
            out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(v):.1f}" r="2" fill="var(--accent)"></circle>')
    if cm.size:
        out.append(f'<text x="{W-R+10}" y="{T+40}" fill="var(--fail)" font-family="{SANS}" '
                   f'font-size="10.5">collapsed blocks</text>')
        out.append(f'<text x="{W-R+10}" y="{T+54}" fill="var(--muted)" font-family="{MONO}" '
                   f'font-size="10">mean {cm.mean():.1f} vs {med:.0f}</text>')
    out.append('</svg>')
    return '\n      '.join(out)


def pc_landscape_svg(train, states, traj_block_means, block_ids, frames, evr):
    W, H = 720, 430
    L, R, T, B = 62, 96, 18, 44
    xs = np.concatenate([train[:, 0], frames[:, 0]])
    ys = np.concatenate([train[:, 1], frames[:, 1]])
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    px, py = (x1 - x0) * .05, (y1 - y0) * .05
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    sy = lambda v: H - B - (v - y0) / (y1 - y0) * (H - T - B)

    COL = {'Inactive': 'var(--accent)', 'Active': 'var(--burnt)',
           'Intermediate': '#d8a13a'}
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="'
           f'Principal component 1 against principal component 2 for '
           f'{len(train)} experimental GPCR structures coloured by activation '
           f'state, with the NTSR1 rep0 trajectory drawn on top coloured by '
           f'block. The trajectory begins in the active-leaning region and '
           f'migrates toward the intermediate centroid.">']

    out.append('<g stroke="var(--rule-soft)" stroke-width="1">')
    out.append(f'<line x1="{sx(0):.1f}" y1="{T}" x2="{sx(0):.1f}" y2="{H-B}"></line>')
    out.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{W-R}" y2="{sy(0):.1f}"></line>')
    out.append('</g>')

    # training structures
    for state in ('Inactive', 'Intermediate', 'Active'):
        m = np.array([s == state for s in states])
        if not m.any():
            continue
        out.append(f'<g fill="{COL[state]}" fill-opacity="0.34">')
        for a, b in train[m][:, :2]:
            out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="3"></circle>')
        out.append('</g>')

    # every frame, faint
    out.append('<g fill="var(--muted)" fill-opacity="0.20">')
    for a, b in frames[::4][:, :2]:
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="1.5"></circle>')
    out.append('</g>')

    # per-block mean path
    pts = ' '.join(f'{sx(a):.1f},{sy(b):.1f}' for a, b in traj_block_means[:, :2])
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--ink)" '
               f'stroke-width="1.1" stroke-opacity="0.45"></polyline>')
    n = len(traj_block_means)
    for i, (a, b) in enumerate(traj_block_means[:, :2]):
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="3.4" '
                   f'fill="{viridis(i/(n-1))}" stroke="var(--card)" stroke-width="0.7"></circle>')
    a0, b0 = traj_block_means[0, :2]; a1, b1 = traj_block_means[-1, :2]
    out.append(f'<circle cx="{sx(a0):.1f}" cy="{sy(b0):.1f}" r="7.5" fill="none" '
               f'stroke="var(--ink)" stroke-width="1.6"></circle>')
    out.append(f'<text x="{sx(a0):.1f}" y="{sy(b0)-12:.1f}" fill="var(--ink)" '
               f'font-family="IBM Plex Mono, monospace" font-size="10.5" '
               f'text-anchor="middle">block 0</text>')
    out.append(f'<text x="{sx(a1):.1f}" y="{sy(b1)+20:.1f}" fill="var(--ink)" '
               f'font-family="IBM Plex Mono, monospace" font-size="10.5" '
               f'text-anchor="middle">block {block_ids[-1]}</text>')

    # axes
    out.append('<g fill="var(--muted)" font-family="IBM Plex Mono, monospace" font-size="11">')
    for v in np.linspace(x0, x1, 5):
        out.append(f'<text x="{sx(v):.1f}" y="{H-B+16}" text-anchor="middle">{v:.0f}</text>')
    for v in np.linspace(y0, y1, 5):
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" fill="var(--ink-2)" '
               f'font-family="IBM Plex Sans, sans-serif" font-size="12" '
               f'text-anchor="middle">PC1 &#183; {evr[0]:.1%} of training variance</text>')
    out.append(f'<text x="14" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" '
               f'font-family="IBM Plex Sans, sans-serif" font-size="12" '
               f'text-anchor="middle" transform="rotate(-90 14 {(T+H-B)/2:.0f})">'
               f'PC2 &#183; {evr[1]:.1%}</text>')

    # legend
    lx, ly = W - R + 10, T + 10
    for k, (lab, col) in enumerate((('Inactive', COL['Inactive']),
                                    ('Intermed.', COL['Intermediate']),
                                    ('Active', COL['Active']))):
        out.append(f'<circle cx="{lx+4}" cy="{ly+k*17}" r="3.6" fill="{col}" fill-opacity="0.6"></circle>')
        out.append(f'<text x="{lx+13}" y="{ly+k*17+4}" fill="var(--muted)" '
                   f'font-family="IBM Plex Sans, sans-serif" font-size="10.5">{lab}</text>')
    # colour ramp for blocks
    ry = ly + 3 * 17 + 14
    out.append(f'<text x="{lx}" y="{ry}" fill="var(--muted)" '
               f'font-family="IBM Plex Sans, sans-serif" font-size="10.5">block</text>')
    for k in range(40):
        out.append(f'<rect x="{lx+4}" y="{ry+8+k*3.2:.1f}" width="9" height="3.3" '
                   f'fill="{viridis(k/39)}"></rect>')
    out.append(f'<text x="{lx+17}" y="{ry+14}" fill="var(--faint)" '
               f'font-family="IBM Plex Mono, monospace" font-size="9.5">0</text>')
    out.append(f'<text x="{lx+17}" y="{ry+8+40*3.2:.0f}" fill="var(--faint)" '
               f'font-family="IBM Plex Mono, monospace" font-size="9.5">{block_ids[-1]}</text>')
    out.append('</svg>')
    return '\n      '.join(out)


def drift_svg(block_ids, pc1, centroids, fit):
    W, H = 720, 320
    L, R, T, B = 62, 108, 20, 42
    y0 = min(pc1.min(), centroids['Inactive']) - 1.5
    y1 = max(pc1.max(), centroids['Active']) + 1.5
    sx = lambda b: L + (b - block_ids[0]) / (block_ids[-1] - block_ids[0]) * (W - L - R)
    sy = lambda v: H - B - (v - y0) / (y1 - y0) * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="'
           f'Mean PC1 of the 310 kelvin ensemble against block index. It falls '
           f'from about 6.5 at block 0 to about 0 at block {block_ids[-1]}, a '
           f'monotone drift from the active-leaning region toward the '
           f'intermediate centroid that has not plateaued by the end of the run.">']
    for lab, col in (('Active', 'var(--burnt)'), ('Intermediate', '#d8a13a'),
                     ('Inactive', 'var(--accent)')):
        v = centroids[lab]
        if not (y0 <= v <= y1):
            continue
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}" '
                   f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3" stroke-opacity="0.75"></line>')
        out.append(f'<text x="{W-R+6}" y="{sy(v)+4:.1f}" fill="{col}" '
                   f'font-family="IBM Plex Sans, sans-serif" font-size="10.5">'
                   f'{lab} ({v:+.1f})</text>')

    pts = ' '.join(f'{sx(b):.1f},{sy(v):.1f}' for b, v in zip(block_ids, pc1))
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--ink-2)" stroke-width="1.5"></polyline>')
    for b, v in zip(block_ids, pc1):
        out.append(f'<circle cx="{sx(b):.1f}" cy="{sy(v):.1f}" r="2.2" '
                   f'fill="{viridis((b-block_ids[0])/(block_ids[-1]-block_ids[0]))}"></circle>')
    a, c = fit['slope'], fit['intercept']
    out.append(f'<line x1="{sx(block_ids[0]):.1f}" y1="{sy(a*block_ids[0]+c):.1f}" '
               f'x2="{sx(block_ids[-1]):.1f}" y2="{sy(a*block_ids[-1]+c):.1f}" '
               f'stroke="var(--fail)" stroke-width="1.4" stroke-dasharray="6 4"></line>')

    out.append('<g fill="var(--muted)" font-family="IBM Plex Mono, monospace" font-size="11">')
    for b in range(0, block_ids[-1] + 1, 20):
        out.append(f'<text x="{sx(b):.1f}" y="{H-B+16}" text-anchor="middle">{b}</text>')
    for v in np.linspace(y0, y1, 5):
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" fill="var(--ink-2)" '
               f'font-family="IBM Plex Sans, sans-serif" font-size="12" '
               f'text-anchor="middle">block (25 ns each)</text>')
    out.append(f'<text x="14" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" '
               f'font-family="IBM Plex Sans, sans-serif" font-size="12" '
               f'text-anchor="middle" transform="rotate(-90 14 {(T+H-B)/2:.0f})">mean PC1</text>')
    out.append('</svg>')
    return '\n      '.join(out)


def segmented_drift_svg(blocks, values, centroids=None, ylab='mean PC1',
                        min_segment=15, H=330):
    """A time series with a fitted changepoint and a linear fit either side.

    A single regression over a non-monotone series reports a slope that
    describes neither half of it.  On ``NTSR1_ML-301.hmr.rep0`` the whole-run
    fit gives -0.028 per block, while the series actually falls at -0.089
    through block 39 and rises at +0.023 thereafter -- a reversal the single
    fit hides entirely.  The changepoint is chosen to minimise the summed
    squared error of the two fits.

    ``centroids`` optionally draws labelled reference levels (activation-state
    means, say) as dashed lines.
    """
    from scipy import stats

    W, L, R, T, B = 720, 62, 132, 22, 46
    x = np.asarray(blocks, float); y = np.asarray(values, float)
    lo, hi = y.min(), y.max()
    if centroids:
        inside = [v for v in centroids.values() if lo - 3 <= v <= hi + 3]
        if inside:
            lo, hi = min(lo, min(inside)), max(hi, max(inside))
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad
    sx = lambda v: L + (v - x.min()) / (x.max() - x.min()) * (W - L - R)
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - T - B)

    best = None
    for k in range(min_segment, len(x) - min_segment):
        err = 0.0
        for sl in (slice(0, k), slice(k, None)):
            a, b, *_ = stats.linregress(x[sl], y[sl])
            err += float(np.sum((y[sl] - (a * x[sl] + b)) ** 2))
        if best is None or err < best[1]:
            best = (k, err)
    k = best[0] if best else len(x) // 2
    fits = [stats.linregress(x[:k], y[:k]), stats.linregress(x[k:], y[k:])]

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean principal '
           f'component 1 per block with a fitted changepoint at block '
           f'{x[k]:.0f}. The series falls at {fits[0].slope:+.3f} per block '
           f'before it and rises at {fits[1].slope:+.3f} after, so the drift '
           f'reverses rather than settling.">']
    out.append('<g stroke="var(--rule-soft)" stroke-width="1">')
    yt = np.linspace(lo, hi, 5)
    for v in yt:
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}"></line>')
    out.append('</g>')

    for lab, col in (('Active', 'var(--burnt)'), ('Intermediate', '#d8a13a'),
                     ('Inactive', 'var(--accent)')):
        v = (centroids or {}).get(lab)
        if v is None or not (lo <= v <= hi):
            continue
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}" '
                   f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3" '
                   f'stroke-opacity="0.75"></line>')
        out.append(f'<text x="{W-R+6}" y="{sy(v)+4:.1f}" fill="{col}" '
                   f'font-family="{SANS}" font-size="10.5">{lab} ({v:+.1f})</text>')

    out.append(f'<line x1="{sx(x[k]):.1f}" y1="{T}" x2="{sx(x[k]):.1f}" y2="{H-B}" '
               f'stroke="var(--ink)" stroke-width="1" stroke-dasharray="2 4" '
               f'stroke-opacity="0.55"></line>')
    out.append(f'<polyline points="{" ".join(f"{sx(a):.1f},{sy(b):.1f}" for a,b in zip(x,y))}" '
               f'fill="none" stroke="var(--ink-2)" stroke-width="1.4"></polyline>')
    n = len(x)
    for i, (a, b) in enumerate(zip(x, y)):
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2.2" '
                   f'fill="{viridis(i/(n-1))}"></circle>')
    for sl, fit, col in ((slice(0, k), fits[0], 'var(--fail)'),
                         (slice(k, None), fits[1], 'var(--pass)')):
        xa, xb = x[sl][0], x[sl][-1]
        out.append(f'<line x1="{sx(xa):.1f}" y1="{sy(fit.slope*xa+fit.intercept):.1f}" '
                   f'x2="{sx(xb):.1f}" y2="{sy(fit.slope*xb+fit.intercept):.1f}" '
                   f'stroke="{col}" stroke-width="1.8" stroke-dasharray="6 4"></line>')

    out.append(f'<g fill="var(--muted)" font-family="{MONO}" font-size="11">')
    for v in range(0, int(x.max()) + 1, 20):
        out.append(f'<text x="{sx(v):.1f}" y="{H-B+16}" text-anchor="middle">{v}</text>')
    for v in yt:
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-5}" fill="var(--ink-2)" '
               f'font-family="{SANS}" font-size="12" text-anchor="middle">block</text>')
    out.append(f'<text x="13" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" font-family="{SANS}" '
               f'font-size="12" text-anchor="middle" '
               f'transform="rotate(-90 13 {(T+H-B)/2:.0f})">{ylab}</text>')
    out.append(f'<text x="{sx(x[k])+5:.1f}" y="{T+12}" fill="var(--ink)" '
               f'font-family="{MONO}" font-size="10">changepoint block {x[k]:.0f}</text>')
    out.append('</svg>')
    return '\n      '.join(out), {'changepoint': float(x[k]),
                                  'before': {'slope': float(fits[0].slope),
                                             'r': float(fits[0].rvalue),
                                             'p': float(fits[0].pvalue)},
                                  'after': {'slope': float(fits[1].slope),
                                            'r': float(fits[1].rvalue),
                                            'p': float(fits[1].pvalue)}}


def window_threshold_svg(widths, needed, current=0.10, noise_floor=None):
    """Minimum JSD threshold each comparison window would have required."""
    W, H, L, R, T, B = 720, 320, 66, 40, 22, 46
    y = np.asarray(needed, float)
    hi = max(y.max(), current) * 1.18
    bw = (W - L - R) / len(widths)
    sy = lambda v: H - B - v / hi * (H - T - B)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Minimum '
           f'Jensen-Shannon threshold each comparison window would have needed '
           f'to converge, against the {current} threshold in use.">']
    if noise_floor:
        out.append(f'<rect x="{L}" y="{sy(noise_floor):.1f}" width="{W-L-R}" '
                   f'height="{H-B-sy(noise_floor):.1f}" fill="var(--accent-soft)" '
                   f'fill-opacity="0.6"></rect>')
    out.append('<g stroke="var(--rule-soft)" stroke-width="1">')
    yt = np.linspace(0, hi, 5)
    for v in yt:
        out.append(f'<line x1="{L}" y1="{sy(v):.1f}" x2="{W-R}" y2="{sy(v):.1f}"></line>')
    out.append('</g>')
    for i, (w, v) in enumerate(zip(widths, y)):
        x = L + i * bw + bw * 0.22
        col = 'var(--pass)' if v <= current else 'var(--burnt)'
        out.append(f'<rect x="{x:.1f}" y="{sy(v):.1f}" width="{bw*0.56:.1f}" '
                   f'height="{H-B-sy(v):.1f}" fill="{col}" rx="1"></rect>')
        out.append(f'<text x="{x+bw*0.28:.1f}" y="{sy(v)-6:.1f}" fill="var(--ink-2)" '
                   f'font-family="{MONO}" font-size="10.5" text-anchor="middle">{v:.3f}</text>')
        out.append(f'<text x="{x+bw*0.28:.1f}" y="{H-B+16}" fill="var(--muted)" '
                   f'font-family="{MONO}" font-size="11" text-anchor="middle">{w}</text>')
    out.append(f'<line x1="{L}" y1="{sy(current):.1f}" x2="{W-R}" y2="{sy(current):.1f}" '
               f'stroke="var(--ink)" stroke-width="1.3" stroke-dasharray="5 3"></line>')
    out.append(f'<text x="{L+6}" y="{sy(current)-6:.1f}" fill="var(--ink)" '
               f'font-family="{MONO}" font-size="10.5">threshold in use {current:.2f}</text>')
    out.append(f'<g fill="var(--muted)" font-family="{MONO}" font-size="11">')
    for v in yt:
        out.append(f'<text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    out.append('</g>')
    out.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-5}" fill="var(--ink-2)" '
               f'font-family="{SANS}" font-size="12" text-anchor="middle">'
               f'comparison window (blocks)</text>')
    out.append(f'<text x="14" y="{(T+H-B)/2:.0f}" fill="var(--ink-2)" font-family="{SANS}" '
               f'font-size="12" text-anchor="middle" '
               f'transform="rotate(-90 14 {(T+H-B)/2:.0f})">JSD threshold required</text>')
    out.append('</svg>')
    return '\n      '.join(out)
