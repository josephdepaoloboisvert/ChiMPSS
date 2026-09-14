"""Unit tests for chimpss.analysis.convergence.

The replay path is tested against a hand-built cache with known verdicts, so
that the window-policy semantics are pinned independently of any real run. The
null-calibration path is tested against distributions whose separation is known
by construction.
"""
import numpy as np
import pytest

from chimpss.analysis import convergence as cv


def make_cache(n_blocks, jsd_value=0.01, fro_value=0.001, overrides=None):
    """A cache in which every pair agrees, except where overridden."""
    N = n_blocks
    fro = {m: np.full((N, N), fro_value) for m in cv.METRICS}
    jsd = {m: np.full((N, N), jsd_value) for m in cv.METRICS}
    for d in (fro, jsd):
        for m in cv.METRICS:
            np.fill_diagonal(d[m], 0.0)
    for (a, b, metric, f, j) in (overrides or []):
        fro[metric][a, b] = fro[metric][b, a] = f
        jsd[metric][a, b] = jsd[metric][b, a] = j
    return cv.PairwiseCache(list(range(N)), fro, jsd)


# ---------------------------------------------------------------------------
# replay -- window policy semantics
# ---------------------------------------------------------------------------

def test_replay_all_pass_when_every_pair_agrees():
    cache = make_cache(40)
    r = cv.replay(cache, window_blocks=3, require_gate=False, total_n_sims=40)
    assert r.first is not None
    assert len(r.passing) > 0


def test_replay_trailing_window_is_narrower_than_history():
    """One distant bad block breaks a history window but not a trailing one."""
    cache = make_cache(40, overrides=[(5, 39, 'contact', 0.9, 0.9)])
    trailing = cv.replay(cache, window_blocks=3, require_gate=False, total_n_sims=40)
    assert 39 in trailing.passing          # block 5 is outside the last 3
    history = cv.replay(cache, window_fraction=1.0, require_gate=False, total_n_sims=40)
    assert 39 not in history.passing       # block 5 is inside the full history


def test_replay_wider_window_never_passes_more_blocks():
    cache = make_cache(40, overrides=[(10, 30, 'torsion', 0.9, 0.9)])
    counts = [len(cv.replay(cache, window_blocks=k, require_gate=False,
                            total_n_sims=40).passing) for k in (1, 3, 5, 10, 20)]
    assert counts == sorted(counts, reverse=True)


def test_replay_nan_pair_fails_the_metric():
    """A NaN comparison -- what an all-NaN contact matrix produces -- must fail."""
    cache = make_cache(40)
    cache.jsd['contact'][20, 19] = cache.jsd['contact'][19, 20] = np.nan
    r = cv.replay(cache, window_blocks=1, require_gate=False, total_n_sims=40)
    assert 20 not in r.passing


def test_replay_exclude_removes_a_poisoning_block():
    cache = make_cache(40, overrides=[(30, 31, 'contact', 0.9, 0.9)])
    before = cv.replay(cache, window_blocks=2, require_gate=False, total_n_sims=40)
    after = cv.replay(cache, window_blocks=2, require_gate=False, total_n_sims=40,
                      exclude={30})
    assert 31 not in before.passing
    assert 31 in after.passing


def test_replay_gate_blocks_when_discard_too_high():
    cache = make_cache(40)
    high = [0.9] * 40
    r = cv.replay(cache, window_blocks=3, discard=high, total_n_sims=40)
    assert r.passing == []


def test_replay_gate_allows_when_discard_low():
    cache = make_cache(40)
    low = [0.3] * 40
    r = cv.replay(cache, window_blocks=3, discard=low, total_n_sims=40)
    assert r.passing


def test_replay_respects_minimum_fraction():
    """Nothing may pass before minimum_fraction of the requested blocks."""
    cache = make_cache(40)
    r = cv.replay(cache, window_blocks=1, require_gate=False,
                  total_n_sims=40, minimum_fraction=0.5)
    assert all(b >= 20 for b in r.passing)


def test_replay_threshold_is_honoured():
    cache = make_cache(40, jsd_value=0.08)
    assert cv.replay(cache, window_blocks=3, require_gate=False,
                     total_n_sims=40, jsd_thresh=0.10).passing
    assert not cv.replay(cache, window_blocks=3, require_gate=False,
                         total_n_sims=40, jsd_thresh=0.05).passing


def test_replay_records_the_window_it_used():
    cache = make_cache(40)
    r = cv.replay(cache, window_blocks=4, require_gate=False, total_n_sims=40)
    assert r['per_block'][20]['window'] == (16, 19)


# ---------------------------------------------------------------------------
# PairwiseCache
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    cache = make_cache(6)
    path = tmp_path / 'pairwise.npz'
    np.savez_compressed(path, have=np.array(cache.blocks),
                        **{f'fro_{m}': cache.fro[m] for m in cv.METRICS},
                        **{f'jsd_{m}': cache.jsd[m] for m in cv.METRICS})
    back = cv.PairwiseCache.load(str(path))
    assert back.blocks == cache.blocks
    assert back.pair('torsion', 1, 2) == cache.pair('torsion', 1, 2)


def test_cache_has_reports_membership():
    cache = cv.PairwiseCache([0, 2, 5], {}, {})
    assert cache.has(2) and not cache.has(1)


def test_cache_default_path_is_outside_saved_variables():
    """saved_variables/ is globbed and int()ed; a file there breaks analysis."""
    p = cv.PairwiseCache.default_path('/runs/x')
    assert 'saved_variables' not in p
    assert p.endswith('analysis/pairwise.npz')


# ---------------------------------------------------------------------------
# null calibration
# ---------------------------------------------------------------------------

def test_calibrated_threshold_is_the_quantile():
    null = np.linspace(0.0, 1.0, 101)
    assert cv.calibrated_threshold(null, q=0.99) == pytest.approx(0.99, abs=0.02)


def test_calibrated_threshold_safety_factor_scales():
    null = np.linspace(0.0, 1.0, 101)
    a = cv.calibrated_threshold(null, q=0.9, safety=1.0)
    b = cv.calibrated_threshold(null, q=0.9, safety=2.0)
    assert b == pytest.approx(2.0 * a)


# ---------------------------------------------------------------------------
# covariance overlap
# ---------------------------------------------------------------------------

def test_covariance_overlap_is_one_for_identical_ensembles():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 12, 3)).astype(np.float32)
    assert cv.covariance_overlap(X, X) == pytest.approx(1.0, abs=1e-6)


def test_covariance_overlap_high_for_same_distribution():
    rng = np.random.default_rng(1)
    A = rng.normal(0, 1, (500, 12, 3)).astype(np.float32)
    B = rng.normal(0, 1, (500, 12, 3)).astype(np.float32)
    assert cv.covariance_overlap(A, B) > 0.7


def test_covariance_overlap_drops_for_different_amplitude():
    rng = np.random.default_rng(2)
    A = rng.normal(0, 1, (500, 12, 3)).astype(np.float32)
    B = rng.normal(0, 6, (500, 12, 3)).astype(np.float32)
    assert cv.covariance_overlap(A, B) < cv.covariance_overlap(A, A)


def test_covariance_overlap_symmetric():
    rng = np.random.default_rng(3)
    A = rng.normal(0, 1, (200, 8, 3)).astype(np.float32)
    B = rng.normal(0, 2, (200, 8, 3)).astype(np.float32)
    assert cv.covariance_overlap(A, B) == pytest.approx(cv.covariance_overlap(B, A), abs=1e-9)


# ---------------------------------------------------------------------------
# cluster populations
# ---------------------------------------------------------------------------

def test_cluster_populations_separates_planted_clusters():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 0.1, (200, 6, 3)).astype(np.float32)
    b = (rng.normal(0, 0.1, (200, 6, 3)) + 12.0).astype(np.float32)
    out = cv.cluster_populations({0: a, 1: b}, n_clusters=2, max_frames_per_block=150)
    # each block should sit almost entirely in one cluster, and not the same one
    assert out['histograms'][0].max() > 0.95
    assert out['histograms'][1].max() > 0.95
    assert np.argmax(out['histograms'][0]) != np.argmax(out['histograms'][1])


def test_cluster_populations_flags_collapse_as_single_occupancy():
    """A collapsed resample is many copies of one structure -> one cluster."""
    rng = np.random.default_rng(5)
    healthy = rng.normal(0, 3.0, (300, 6, 3)).astype(np.float32)
    collapsed = np.repeat(rng.normal(0, 0.001, (1, 6, 3)), 300, axis=0).astype(np.float32)
    out = cv.cluster_populations({0: healthy, 1: collapsed}, n_clusters=6,
                                 max_frames_per_block=200)
    assert out['occupancy'][1] == 1
    assert out['occupancy'][0] > 1


# ---------------------------------------------------------------------------
# resolve_window -- the shared window policy used by both the live criterion
# and the retrospective report
# ---------------------------------------------------------------------------

from chimpss.fultonmarket.retro_convergence import resolve_window


def test_resolve_window_trailing_blocks():
    assert resolve_window(50, 0.3, 0.75, window_blocks=3) == 47
    assert resolve_window(2, 0.3, 0.75, window_blocks=10) == 0      # clamped


def test_resolve_window_trailing_fraction():
    assert resolve_window(80, 0.9, 0.75, window_fraction=0.25) == 60


def test_resolve_window_published_rule_uses_the_floor():
    """Above max_equil_fraction the 0.25 floor binds, pinning the trailing quarter."""
    assert resolve_window(80, 0.90, 0.75) == 60
    assert resolve_window(80, 0.99, 0.75) == 60


def test_resolve_window_published_rule_widens_as_discard_falls():
    """The coupling that makes the gate and the matrix checks oppose each other."""
    wide = resolve_window(80, 0.30, 0.75)
    narrow = resolve_window(80, 0.85, 0.75)
    assert wide < narrow


def test_resolve_window_published_rule_is_not_monotone_in_sim_no():
    """A downward revision of t0 re-admits blocks that had aged out."""
    before = resolve_window(60, 0.80, 0.75)
    after = resolve_window(61, 0.40, 0.75)
    assert after < before


def test_resolve_window_trailing_policies_are_monotone():
    for kw in ({'window_blocks': 5}, {'window_fraction': 0.25}):
        vals = [resolve_window(n, 0.5, 0.75, **kw) for n in range(10, 90)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_resolve_window_blocks_take_precedence_over_fraction():
    assert resolve_window(80, 0.5, 0.75, window_blocks=2, window_fraction=0.9) == 78


# ---------------------------------------------------------------------------
# realign_cache
#
# ca_cache originally superposed each block onto its own first frame, leaving
# every block in a different global orientation. Clustering then separated
# blocks by rotation rather than conformation (adjacent-block population JSD
# 0.987, healthy blocks occupying 1.4 of 20 clusters). These pin the fix.
# ---------------------------------------------------------------------------

def _rotate(X, theta, axis=2):
    c, s = np.cos(theta), np.sin(theta)
    R = {2: np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
         0: np.array([[1, 0, 0], [0, c, -s], [0, s, c]])}[axis]
    return X @ R.T


def test_realign_removes_a_block_wide_rotation():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (50, 10, 3))
    ca = {0: base.astype(np.float32),
          1: _rotate(base, 0.9).astype(np.float32)}      # same conformations, rotated
    out = cv.realign_cache(ca)
    assert np.allclose(out[0], out[1], atol=1e-4)


def test_realign_preserves_genuine_conformational_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (50, 10, 3))
    b = a + rng.normal(0, 2.0, a.shape)                  # really different
    out = cv.realign_cache({0: a.astype(np.float32), 1: b.astype(np.float32)})
    assert not np.allclose(out[0], out[1], atol=1e-2)


def test_realign_is_translation_invariant():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, (30, 8, 3))
    out = cv.realign_cache({0: base.astype(np.float32),
                            1: (base + 25.0).astype(np.float32)})
    assert np.allclose(out[0], out[1], atol=1e-4)


def test_realign_does_not_introduce_reflection():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, (40, 9, 3))
    mirrored = base * np.array([1.0, 1.0, -1.0])
    out = cv.realign_cache({0: base.astype(np.float32), 1: mirrored.astype(np.float32)})
    assert not np.allclose(out[0], out[1], atol=1e-2)


def test_realign_centres_every_block():
    rng = np.random.default_rng(4)
    ca = {i: (rng.normal(0, 1, (20, 7, 3)) + 10 * i).astype(np.float32) for i in range(3)}
    out = cv.realign_cache(ca)
    for v in out.values():
        assert np.allclose(v.mean(axis=1), 0.0, atol=1e-4)


def test_realign_empty_cache_is_a_noop():
    assert cv.realign_cache({}) == {}


def test_clustering_separates_blocks_when_frames_differ_only_by_rotation():
    """The bug this fix addresses, reproduced then corrected."""
    rng = np.random.default_rng(5)
    # A rigid elongated template with only small per-frame jitter. Rotation then
    # displaces the whole block far relative to its internal spread, which is
    # the regime a real protein block is in. (With isotropic noise of comparable
    # width the rotated copy is statistically identical and no clustering can
    # separate the blocks -- the artefact needs structure to show up.)
    template = np.array([[i * 3.0, 0.0, 0.0] for i in range(8)])
    base = template + rng.normal(0, 0.1, (120, 8, 3))
    misaligned = {0: base.astype(np.float32), 1: _rotate(base, 1.2).astype(np.float32)}
    bad = cv.cluster_populations(misaligned, n_clusters=4, max_frames_per_block=100)
    # rotation alone puts the two blocks in different clusters
    assert np.argmax(bad['histograms'][0]) != np.argmax(bad['histograms'][1])
    good = cv.cluster_populations(cv.realign_cache(misaligned), n_clusters=4,
                                  max_frames_per_block=100)
    # after realignment they are the same ensemble, so populations agree
    from scipy.spatial.distance import jensenshannon
    assert float(jensenshannon(good['histograms'][0] + 1e-12,
                               good['histograms'][1] + 1e-12, base=2)) < 0.3
