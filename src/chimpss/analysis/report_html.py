r"""Assembly and pre-publication validation for the convergence report.

The report is a single self-contained HTML page: one stylesheet defining a
light and a dark palette as CSS custom properties, then sections of prose with
inline SVG figures from :mod:`chimpss.analysis.report_figures`.

:func:`check_document` is the part worth running every time.  An SVG mark
placed outside its ``viewBox`` is silently clipped -- nothing raises, and the
chart looks plausible without the missing points.  One published version of
this report drew thirty JSD points above the top of the plot that way, and it
was caught by eye rather than by anything automatic.  Validate before
publishing, not after.
"""
import re
import xml.etree.ElementTree as ET

__all__ = ['HEAD', 'page', 'figure', 'check_svg_bounds', 'check_document']

HEAD = """<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>
:root{
  --ink:#111a1f;
  --ink-2:#2c3a42;
  --muted:#5c6b73;
  --faint:#8d9aa1;
  --paper:#f4f7f8;
  --card:#ffffff;
  --rule:#d9e2e5;
  --rule-soft:#e8eef0;
  --accent:#0e666b;
  --accent-soft:#e0efef;
  --burnt:#a8521e;
  --burnt-soft:#f6e8de;
  --pass:#2d6a4d;
  --fail:#a8521e;
  --shadow:0 1px 2px rgba(17,26,31,.06),0 8px 24px -16px rgba(17,26,31,.28);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ink:#e8eff1; --ink-2:#c2d0d5; --muted:#93a3ab; --faint:#6d7d85;
    --paper:#0e1519; --card:#151e23; --rule:#26343a; --rule-soft:#1d282d;
    --accent:#5cc8c6; --accent-soft:#12302f;
    --burnt:#e08a52; --burnt-soft:#2d1e14;
    --pass:#6dbf94; --fail:#e08a52;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --ink:#e8eff1; --ink-2:#c2d0d5; --muted:#93a3ab; --faint:#6d7d85;
  --paper:#0e1519; --card:#151e23; --rule:#26343a; --rule-soft:#1d282d;
  --accent:#5cc8c6; --accent-soft:#12302f;
  --burnt:#e08a52; --burnt-soft:#2d1e14;
  --pass:#6dbf94; --fail:#e08a52;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:860px; margin:0 auto; padding-inline:20px; padding-block:56px 80px}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; font-variant-numeric:tabular-nums}
.eyebrow{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted);
}

/* ---- masthead ---- */
header.mast{border-bottom:1px solid var(--rule); padding-bottom:26px; margin-bottom:34px}
h1{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600;
  font-size:clamp(30px,5.2vw,44px); line-height:1.12; letter-spacing:-.015em;
  margin:12px 0 10px; text-wrap:balance;
}
.dek{font-size:17px; color:var(--ink-2); max-width:62ch; margin:0}
.runline{
  display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:18px;
  font-size:12.5px; color:var(--muted);
}
.runline span{white-space:nowrap}
.runline b{color:var(--ink-2); font-weight:500}

/* ---- verdict ---- */
.verdict{
  background:var(--card); border:1px solid var(--rule);
  border-left:4px solid var(--burnt); border-radius:3px;
  padding:24px 26px; margin-bottom:40px; box-shadow:var(--shadow);
}
.verdict h2{
  font-family:"IBM Plex Serif",Georgia,serif; font-size:23px; font-weight:600;
  margin:8px 0 12px; letter-spacing:-.01em;
}
.verdict p{margin:0 0 12px; color:var(--ink-2); max-width:64ch}
.verdict p:last-child{margin-bottom:0}
.qualifier{
  margin-top:18px; padding-top:16px; border-top:1px solid var(--rule-soft);
  font-size:14.5px; color:var(--muted);
}

/* ---- sections ---- */
section{margin-bottom:46px}
h3{
  font-family:"IBM Plex Serif",Georgia,serif; font-size:21px; font-weight:600;
  margin:0 0 6px; letter-spacing:-.01em; text-wrap:balance;
}
.sub{color:var(--muted); font-size:14.5px; margin:0 0 20px; max-width:64ch}
p{max-width:64ch}

/* ---- tables ---- */
.tablewrap{overflow-x:auto; border:1px solid var(--rule); border-radius:3px; background:var(--card)}
table{border-collapse:collapse; width:100%; min-width:460px; font-size:14px}
th,td{padding:10px 14px; text-align:right; border-bottom:1px solid var(--rule-soft)}
th:first-child,td:first-child{text-align:left}
thead th{
  font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted); font-weight:500;
  border-bottom:1px solid var(--rule); white-space:nowrap;
}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
tr.hl td{background:var(--accent-soft)}
.tag{
  display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:11px;
  letter-spacing:.06em; text-transform:uppercase; padding:2px 7px; border-radius:2px;
}
.tag.pass{color:var(--pass); background:color-mix(in srgb,var(--pass) 12%,transparent)}
.tag.fail{color:var(--fail); background:var(--burnt-soft)}

/* ---- chart ---- */
figure{margin:0}
.chart{
  background:var(--card); border:1px solid var(--rule); border-radius:3px;
  padding:18px 14px 10px; box-shadow:var(--shadow);
}
.chart svg{display:block; width:100%; height:auto; max-width:100%}
figcaption{font-size:13.5px; color:var(--muted); margin-top:12px; max-width:64ch}

/* ---- evidence list ---- */
.lines{display:grid; gap:16px; margin-top:6px}
.line figure{margin-top:14px}
.line svg{display:block; width:100%; height:auto; max-width:100%}
.line{
  background:var(--card); border:1px solid var(--rule); border-radius:3px;
  padding:18px 20px;
}
.line h4{
  font-size:15.5px; font-weight:600; margin:0 0 4px;
  display:flex; align-items:baseline; gap:10px; flex-wrap:wrap;
}
.line h4 .src{font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--faint); font-weight:400}
.line p{margin:0; font-size:14.5px; color:var(--ink-2)}
.line .figure{
  font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  color:var(--ink); font-weight:500;
}

/* ---- callout ---- */
.note{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:16px 20px; border-radius:0 3px 3px 0; margin:24px 0;
}
.note p{margin:0; font-size:14.5px; color:var(--ink-2)}
.note p+p{margin-top:10px}

footer{
  margin-top:56px; padding-top:22px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--faint);
}
footer code{font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--muted)}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:520px){
  .wrap{padding-block:38px 56px}
  .verdict{padding:20px}
}
</style>"""


def page(title, body):
    """Wrap a body fragment in the stylesheet and page shell."""
    return HEAD.replace('__TITLE__', title) + '\n<div class="wrap">\n' + body + '\n</div>\n'


def figure(svg, caption, chart=True):
    """One figure: SVG in a card, caption beneath it.

    ``chart=False`` omits the card, for figures placed inside a ``.line``
    summary card that already provides one.
    """
    inner = '<div class="chart">\n      ' + svg + '\n    </div>' if chart else svg
    return ('<figure>\n    ' + inner
            + '\n    <figcaption>' + caption + '</figcaption>\n  </figure>')


def check_svg_bounds(svg, tol=1.0):
    """Coordinates drawn outside an SVG's ``viewBox``.

    Returns a list of human-readable problems; empty means every mark is
    inside the box.  Text carrying a ``transform`` is skipped, since rotated
    axis labels are positioned before rotation.
    """
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        return [f'malformed SVG: {exc}']
    vb = root.get('viewBox')
    if not vb:
        return ['no viewBox']
    _, _, W, H = (float(v) for v in vb.split())

    xs, ys = [], []
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        try:
            if tag == 'circle':
                xs.append(float(el.get('cx'))); ys.append(float(el.get('cy')))
            elif tag == 'rect':
                x, y = float(el.get('x')), float(el.get('y'))
                xs += [x, x + float(el.get('width'))]
                ys += [y, y + float(el.get('height'))]
            elif tag == 'line':
                xs += [float(el.get('x1')), float(el.get('x2'))]
                ys += [float(el.get('y1')), float(el.get('y2'))]
            elif tag == 'text' and not el.get('transform'):
                xs.append(float(el.get('x'))); ys.append(float(el.get('y')))
            elif tag == 'polyline':
                for pt in el.get('points').split():
                    a, b = pt.split(',')
                    xs.append(float(a)); ys.append(float(b))
        except (TypeError, ValueError):
            continue

    out = []
    for name, vals, lim in (('x', xs, W), ('y', ys, H)):
        bad = [v for v in vals if v < -tol or v > lim + tol]
        if bad:
            out.append(f'{len(bad)} {name} coordinate(s) outside viewBox '
                       f'(0..{lim:.0f}); range {min(vals):.1f}..{max(vals):.1f}')
    return out


def check_document(html, tags=('section', 'figure', 'figcaption', 'div', 'svg',
                               'table', 'p', 'h3', 'h4')):
    """Validate a finished page: tag balance, SVG well-formedness, mark bounds.

    Returns a list of problems; empty means the page is safe to publish.
    """
    problems = []
    for tag in tags:
        opened = len(re.findall(rf'<{tag}[\s>]', html))
        closed = len(re.findall(rf'</{tag}>', html))
        if opened != closed:
            problems.append(f'<{tag}>: {opened} opened, {closed} closed')

    for i, svg in enumerate(re.findall(r'<svg.*?</svg>', html, re.S)):
        for msg in check_svg_bounds(svg):
            problems.append(f'svg {i}: {msg}')

    leftovers = re.findall(r'\{[a-z_][a-z0-9_]*\}', html)
    if leftovers:
        problems.append(f'unsubstituted placeholders: {sorted(set(leftovers))[:6]}')
    return problems
