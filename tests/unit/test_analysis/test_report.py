"""
Unit tests for the report figure generators and page assembly.

The generators emit SVG as text, so the properties worth guarding are the ones
that fail silently in a browser: a mark placed outside the viewBox is clipped
without error, and a malformed fragment renders as nothing. Both are caught
here rather than by eye.
"""
import re

import numpy as np
import pytest

from chimpss.analysis.report_figures import (VIRIDIS, drift_svg,
                                             pairwise_hist_svg,
                                             pc_landscape_svg, pool_svg,
                                             roundtrip_hist_svg,
                                             multi_series_svg,
                                             segmented_drift_svg, series_svg,
                                             viridis, walkers_svg,
                                             window_threshold_svg)
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
        'segmented': segmented_drift_svg(
            list(range(60)),
            np.concatenate([np.linspace(0, -8, 25), np.linspace(-8, -4, 35)]),
            cent)[0],
        'window': window_threshold_svg([1, 3, 5, 10], [0.03, 0.08, 0.14, 0.25],
                                       current=0.10, noise_floor=0.06),
        'multi': multi_series_svg(
            list(range(40)),
            {'last 1': list(np.linspace(0.30, 0.05, 40)),
             'last 5': [None] * 5 + list(np.linspace(0.50, 0.12, 35))},
            'worst JSD', threshold=0.10, threshold_label='0.10',
            band=0.06, band_label='noise floor'),
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


# ── segmented drift ───────────────────────────────────────────────────────────

def _vee(n=80, turn=30, down=-0.3, up=0.1, noise=0.0, seed=0):
    x = np.arange(n, dtype=float)
    y = np.where(x < turn, down * x, down * turn + up * (x - turn))
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, n)
    return list(x), y


def test_segmented_drift_recovers_a_planted_changepoint():
    x, y = _vee(turn=30)
    _, st = segmented_drift_svg(x, y)
    assert abs(st['changepoint'] - 30) <= 3


def test_segmented_drift_recovers_both_slopes():
    x, y = _vee(turn=30, down=-0.3, up=0.1)
    _, st = segmented_drift_svg(x, y)
    assert st['before']['slope'] == pytest.approx(-0.3, abs=0.05)
    assert st['after']['slope'] == pytest.approx(0.1, abs=0.05)
    assert st['before']['slope'] < 0 < st['after']['slope']


def test_segmented_drift_survives_noise():
    x, y = _vee(turn=40, noise=0.25, seed=3)
    _, st = segmented_drift_svg(x, y)
    assert abs(st['changepoint'] - 40) <= 8
    assert st['before']['slope'] < 0 < st['after']['slope']


def test_segmented_drift_on_a_monotone_series_reports_one_direction():
    """No real changepoint: both segments must still slope the same way."""
    x = list(range(60))
    y = np.linspace(0.0, -6.0, 60)
    _, st = segmented_drift_svg(x, y)
    assert st['before']['slope'] < 0
    assert st['after']['slope'] < 0


def test_segmented_drift_respects_min_segment():
    x, y = _vee(n=80, turn=30)
    _, st = segmented_drift_svg(x, y, min_segment=25)
    assert 25 <= st['changepoint'] <= 55


def test_segmented_drift_draws_centroid_levels_in_range():
    x, y = _vee(turn=30)
    svg, _ = segmented_drift_svg(x, y, {'Active': 500.0, 'Intermediate': -5.0})
    assert 'Intermediate' in svg          # inside the data range
    assert 'Active' not in svg            # far outside it, so not drawn


def test_segmented_drift_reports_significance():
    x, y = _vee(turn=30, noise=0.2, seed=1)
    _, st = segmented_drift_svg(x, y)
    for half in ('before', 'after'):
        assert set(st[half]) == {'slope', 'r', 'p'}
        assert 0.0 <= st[half]['p'] <= 1.0


# ── window thresholds ─────────────────────────────────────────────────────────

def test_window_threshold_colours_bars_by_the_threshold_in_use():
    svg = window_threshold_svg([1, 3, 5], [0.03, 0.08, 0.25], current=0.10)
    assert svg.count('var(--pass)') == 2       # 0.03 and 0.08 clear it
    assert svg.count('var(--burnt)') == 1      # 0.25 does not


def test_window_threshold_labels_every_width():
    svg = window_threshold_svg([1, 3, 5, 10, 20], [0.03, 0.05, 0.06, 0.14, 0.24])
    for w in (1, 3, 5, 10, 20):
        assert f'>{w}</text>' in svg


def test_window_threshold_noise_floor_is_optional():
    with_floor = window_threshold_svg([1, 3], [0.03, 0.08], noise_floor=0.05)
    without = window_threshold_svg([1, 3], [0.03, 0.08])
    assert 'accent-soft' in with_floor
    assert 'accent-soft' not in without


def test_window_threshold_handles_all_bars_above_the_threshold():
    svg = window_threshold_svg([5, 10, 20], [0.15, 0.20, 0.30], current=0.10)
    assert 'var(--pass)' not in svg
    assert check_svg_bounds(svg) == []


# ── multi-series timeseries ───────────────────────────────────────────────────

def _polylines(svg):
    return re.findall(r'<polyline points="([^"]+)"', svg)


def test_multi_series_draws_one_line_per_series():
    svg = multi_series_svg(list(range(10)),
                           {'a': [0.1] * 10, 'b': [0.2] * 10, 'c': [0.3] * 10},
                           'y')
    assert len(_polylines(svg)) == 3


def test_multi_series_breaks_the_line_at_gaps():
    """A policy with no answer for a block must leave a gap, not interpolate."""
    svg = multi_series_svg(list(range(6)),
                           {'a': [0.1, 0.2, None, 0.3, 0.4, 0.5]}, 'y')
    lines = _polylines(svg)
    assert len(lines) == 2
    assert len(lines[0].split()) == 2 and len(lines[1].split()) == 3


def test_multi_series_omits_a_run_of_one_point():
    svg = multi_series_svg(list(range(4)), {'a': [0.1, None, 0.2, None]}, 'y')
    assert _polylines(svg) == []


def test_multi_series_leading_gap_starts_the_line_late():
    """A wide window has no answer until enough blocks precede it."""
    svg = multi_series_svg(list(range(8)),
                           {'wide': [None] * 4 + [0.3, 0.2, 0.15, 0.1]}, 'y')
    assert len(_polylines(svg)[0].split()) == 4


def test_multi_series_draws_the_threshold_and_band():
    svg = multi_series_svg(list(range(10)), {'a': [0.2] * 10}, 'y',
                           threshold=0.10, threshold_label='thresh',
                           band=0.05, band_label='floor')
    assert 'stroke-dasharray="5 3"' in svg
    assert 'thresh' in svg and 'floor' in svg
    assert 'accent-soft' in svg


def test_multi_series_threshold_is_optional():
    svg = multi_series_svg(list(range(10)), {'a': [0.2] * 10}, 'y')
    assert 'accent-soft' not in svg
    assert check_svg_bounds(svg) == []


def test_multi_series_labels_every_series_in_the_legend():
    labels = {'last 1': [0.1] * 10, 'last 10': [0.3] * 10, 'last 20': [0.5] * 10}
    svg = multi_series_svg(list(range(10)), labels, 'y')
    for lab in labels:
        assert f'>{lab}</text>' in svg


def test_multi_series_gives_each_series_its_own_colour():
    svg = multi_series_svg(list(range(10)),
                           {'a': [0.1] * 10, 'b': [0.2] * 10, 'c': [0.3] * 10}, 'y')
    used = {re.search(r'stroke="([^"]+)"', p).group(1)
            for p in re.findall(r'<polyline [^>]+>', svg)}
    assert len(used) == 3


def test_multi_series_log_axis_keeps_marks_in_bounds():
    """Values spanning two decades must not overflow the viewBox."""
    svg = multi_series_svg(list(range(30)),
                           {'a': list(np.geomspace(0.02, 1.5, 30))}, 'y',
                           threshold=0.10, logy=True)
    assert check_svg_bounds(svg) == []


def test_multi_series_log_axis_labels_are_values_not_exponents():
    svg = multi_series_svg(list(range(20)),
                           {'a': list(np.geomspace(0.05, 1.0, 20))}, 'y', logy=True)
    assert '-1.00' not in svg          # would be log10 leaking into the label


def test_multi_series_handles_a_flat_series():
    svg = multi_series_svg(list(range(12)), {'a': [0.25] * 12}, 'y', threshold=0.10)
    assert check_svg_bounds(svg) == []


def test_multi_series_rejects_entirely_empty_input():
    with pytest.raises(ValueError, match='every series is empty'):
        multi_series_svg(list(range(5)), {'a': [None] * 5}, 'y')
