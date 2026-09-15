"""
Unit tests for the report figure generators and page assembly.

The generators emit SVG as text, so the properties worth guarding are the ones
that fail silently in a browser: a mark placed outside the viewBox is clipped
without error, and a malformed fragment renders as nothing. Both are caught
here rather than by eye.
"""
import numpy as np
import pytest

from chimpss.analysis.report_figures import (VIRIDIS, pairwise_hist_svg,
                                             pc_landscape_svg, pool_svg,
                                             roundtrip_hist_svg, series_svg,
                                             viridis, walkers_svg, drift_svg)
from chimpss.analysis.report_html import (HEAD, check_document, check_svg_bounds,
                                          figure, page)


# ── viridis ───────────────────────────────────────────────────────────────────

def test_viridis_spans_the_ramp():
    assert viridis(0.0) == VIRIDIS[0]
    assert viridis(1.0) == VIRIDIS[-1]


def test_viridis_clamps_out_of_range():
    assert viridis(-5.0) == VIRIDIS[0]
    assert viridis(9.9) == VIRIDIS[-1]


# ── generators ────────────────────────────────────────────────────────────────

@pytest.fixture
def pairwise():
    rng = np.random.default_rng(0)
    return {k: {'values': list(rng.beta(2, 5, 400) * 0.6),
                'median': 0.21, 'pass_frac': 0.33}
            for k in ('torsion', 'alpha_carbon', 'contact')}


def _all_figures(pairwise):
    rng = np.random.default_rng(1)
    train = rng.normal(scale=6.0, size=(60, 4))
    states = ['Active'] * 25 + ['Inactive'] * 30 + ['Intermediate'] * 5
    means = rng.normal(size=(12, 4))
    frames = rng.normal(size=(90, 4))
    cent = {'Active': 6.0, 'Intermediate': 0.0, 'Inactive': -6.0}
    return {
        'pairwise': pairwise_hist_svg(pairwise),
        'series': series_svg(list(range(40)), list(np.linspace(1.0, 0.2, 40)),
                             'metric', band=0.35, band_label='null'),
        'series_quarters': series_svg(list(range(40)), list(np.linspace(0.2, 0.9, 40)),
                                      'overlap', quarters=[0.3, 0.5, 0.7, 0.85]),
        'pool': pool_svg([[i, 60 - i, max(1, 40 - i)] for i in range(40)]),
        'roundtrips': roundtrip_hist_svg([0, 0, 0, 1, 1, 2, 3, 7]),
        'walkers': walkers_svg(list(range(40)), list(rng.integers(4, 9, 40)), [3, 17]),
        'landscape': pc_landscape_svg(train, states, means, list(range(12)), frames,
                                      np.array([0.25, 0.15, 0.10, 0.05])),
        'drift': drift_svg(np.arange(30), np.linspace(5, -5, 30), cent,
                           {'slope': -0.34, 'intercept': 5.0}),
    }


def test_every_generator_emits_wellformed_svg(pairwise):
    for name, svg in _all_figures(pairwise).items():
        assert svg.startswith('<svg'), name
        assert svg.rstrip().endswith('</svg>'), name
        assert check_svg_bounds(svg) == [], f'{name}: {check_svg_bounds(svg)}'


def test_every_generator_carries_an_aria_label(pairwise):
    for name, svg in _all_figures(pairwise).items():
        assert 'role="img"' in svg, name
        assert 'aria-label="' in svg, name


def test_generators_use_theme_tokens_not_literal_colours(pairwise):
    """Chart furniture must follow the page theme, not a hard-coded ink colour."""
    for name, svg in _all_figures(pairwise).items():
        assert 'var(--' in svg, name


def test_pool_figure_marks_collapsed_blocks(pairwise):
    svg = pool_svg([[0, 100, 300], [1, 8, 3], [2, 90, 280]], collapse_at=10)
    assert 'var(--fail)' in svg


def test_roundtrip_histogram_counts_every_walker():
    per_walker = [0, 0, 1, 1, 1, 2]
    svg = roundtrip_hist_svg(per_walker)
    assert '2 of 6 walkers' in svg or '>2<' in svg
    assert check_svg_bounds(svg) == []


def test_series_svg_handles_a_flat_series():
    svg = series_svg(list(range(20)), [0.5] * 20, 'flat')
    assert check_svg_bounds(svg) == []


def test_walkers_svg_handles_no_collapsed_blocks():
    svg = walkers_svg(list(range(10)), [5] * 10, [])
    assert check_svg_bounds(svg) == []


# ── page assembly ─────────────────────────────────────────────────────────────

def test_head_defines_both_themes():
    assert 'prefers-color-scheme:dark' in HEAD.replace(' ', '')
    assert '[data-theme="dark"]' in HEAD
    assert '__TITLE__' in HEAD


def test_page_substitutes_the_title():
    html = page('Convergence Detectors Compared', '<section></section>')
    assert '<title>Convergence Detectors Compared</title>' in html
    assert '__TITLE__' not in html
    assert '<div class="wrap">' in html


def test_figure_wraps_svg_and_caption():
    out = figure('<svg viewBox="0 0 10 10"></svg>', 'a caption')
    assert '<div class="chart">' in out
    assert '<figcaption>a caption</figcaption>' in out


def test_figure_can_omit_the_card():
    out = figure('<svg viewBox="0 0 10 10"></svg>', 'c', chart=False)
    assert '<div class="chart">' not in out


# ── validation ────────────────────────────────────────────────────────────────

def test_check_svg_bounds_accepts_marks_inside():
    assert check_svg_bounds('<svg viewBox="0 0 100 50">'
                            '<circle cx="50" cy="25" r="2"></circle></svg>') == []


def test_check_svg_bounds_flags_marks_outside():
    problems = check_svg_bounds('<svg viewBox="0 0 100 50">'
                                '<circle cx="50" cy="-30" r="2"></circle></svg>')
    assert len(problems) == 1 and 'y coordinate' in problems[0]


def test_check_svg_bounds_flags_polyline_points():
    problems = check_svg_bounds('<svg viewBox="0 0 100 50">'
                                '<polyline points="10,10 160,20"></polyline></svg>')
    assert problems and 'x coordinate' in problems[0]


def test_check_svg_bounds_ignores_rotated_axis_labels():
    """Rotated labels are positioned before rotation; do not flag them."""
    svg = ('<svg viewBox="0 0 100 50"><text x="-40" y="25" '
           'transform="rotate(-90 13 25)">label</text></svg>')
    assert check_svg_bounds(svg) == []


def test_check_svg_bounds_reports_malformed_fragments():
    assert 'malformed SVG' in check_svg_bounds('<svg viewBox="0 0 1 1"><circle')[0]


def test_check_svg_bounds_requires_a_viewbox():
    assert check_svg_bounds('<svg><circle cx="1" cy="1" r="1"></circle></svg>') == ['no viewBox']


def test_check_document_passes_a_clean_page(pairwise):
    body = '\n'.join(figure(s, 'c') for s in _all_figures(pairwise).values())
    assert check_document(page('T', f'<section>{body}</section>')) == []


def test_check_document_flags_unbalanced_tags():
    problems = check_document('<section><figure></figure>')
    assert any('<section>' in p for p in problems)


def test_check_document_flags_unsubstituted_placeholders():
    problems = check_document('<section>value is {slope}</section>')
    assert any('placeholder' in p for p in problems)


def test_check_document_flags_a_clipped_figure():
    html = ('<section>' + figure('<svg viewBox="0 0 100 50">'
            '<circle cx="500" cy="25" r="2"></circle></svg>', 'c') + '</section>')
    assert any('outside viewBox' in p for p in check_document(html))
