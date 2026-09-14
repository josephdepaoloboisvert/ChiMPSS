"""Unit tests for chimpss.analysis.equilibration.

The detectors are validated against synthetic series with a *known* truncation
point, because the whole reason this module exists is that the incumbent
detector is unreliable on real data -- so "it agrees with pymbar" is not a
test, it is the thing under question.
"""
import numpy as np
import pytest

from chimpss.analysis import equilibration as eq


def make_series(n_burn, n_eq, amplitude=5.0, noise=0.2, seed=0):
    """Exponential transient of length ``n_burn`` followed by stationary noise."""
    rng = np.random.default_rng(seed)
    tau = max(n_burn / 5.0, 1.0)
    burn = amplitude * np.exp(-np.arange(n_burn) / tau) + rng.normal(0, noise, n_burn)
    return np.concatenate([burn, rng.normal(0, noise, n_eq)])


# ---------------------------------------------------------------------------
# long_run_variance
# ---------------------------------------------------------------------------

def test_long_run_variance_white_noise_matches_sample_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 4000)
    assert eq.long_run_variance(x, 0) == pytest.approx(x.var(), rel=0.05)
    # for uncorrelated data a window should not change the estimate much
    assert eq.long_run_variance(x, 20) == pytest.approx(x.var(), rel=0.25)


def test_long_run_variance_exceeds_sample_variance_when_correlated():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 8000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, e.size):          # AR(1), phi = 0.9 -> long-run var >> sample var
        x[i] = 0.9 * x[i - 1] + e[i]
    assert eq.long_run_variance(x, 60) > 2.0 * eq.long_run_variance(x, 0)


def test_long_run_variance_never_negative():
    rng = np.random.default_rng(3)
    x = np.array([1.0, -1.0] * 50) + rng.normal(0, 0.01, 100)   # anti-correlated
    assert eq.long_run_variance(x, 30) >= 0.0


def test_long_run_variance_degenerate_inputs():
    assert np.isnan(eq.long_run_variance(np.array([1.0])))
    assert eq.long_run_variance(np.zeros(50), 5) == 0.0


# ---------------------------------------------------------------------------
# MSER family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['mser_5', 'mser_w5', 'mser_sqrt'])
@pytest.mark.parametrize('n_burn', [50, 200, 400])
def test_autocorrelation_aware_mser_recovers_truncation_point(name, n_burn):
    """The variants that model autocorrelation should land within 15% of n."""
    n = 1000
    x = make_series(n_burn, n - n_burn, seed=n_burn)
    t0 = eq.DETECTORS[name](x)
    assert abs(t0 - n_burn) < 0.15 * n, f'{name}: t0={t0}, expected ~{n_burn}'


@pytest.mark.parametrize('n_burn', [50, 200, 400])
def test_uncorrelated_mser_truncates_early(n_burn):
    """White's original rule under-estimates t0 on correlated data.

    This is the documented weakness that motivates the windowed variants
    (Clark, Cole & Michel, JCTC 2025): assuming independence under-estimates
    the variance of the mean, so the rule stops discarding too soon.  Pinned
    here as expected behaviour rather than treated as a bug -- it is the
    reason ``mser_sqrt`` is the recommended default.
    """
    n = 1000
    x = make_series(n_burn, n - n_burn, seed=n_burn)
    assert eq.mser(x, window=0) <= n_burn + 0.05 * n


def test_mser_respects_max_fraction():
    x = make_series(100, 900)
    assert eq.mser(x, max_fraction=0.2) < 0.2 * x.size


def test_mser_on_pure_noise_truncates_early():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 1000)
    assert eq.mser(x, window='sqrt') < 0.25 * x.size


def test_mser_short_series_returns_zero():
    assert eq.mser(np.array([1.0, 2.0, 3.0])) == 0


# ---------------------------------------------------------------------------
# Gelman-Rubin
# ---------------------------------------------------------------------------

def test_rhat_near_one_for_converged_chains():
    rng = np.random.default_rng(5)
    chains = [rng.normal(0, 1, 600) for _ in range(4)]
    assert eq.gelman_rubin(chains) < 1.05
    assert eq.rank_normalized_rhat(chains) < 1.05


def test_rhat_large_for_disagreeing_chains():
    rng = np.random.default_rng(6)
    chains = [rng.normal(m, 1, 600) for m in (0.0, 4.0, 8.0)]
    assert eq.gelman_rubin(chains) > 1.5
    assert eq.rank_normalized_rhat(chains) > 1.5


def test_rank_normalized_rhat_catches_scale_difference():
    """Chains sharing a mean but differing in variance: folding is what catches it."""
    rng = np.random.default_rng(7)
    chains = [rng.normal(0, s, 800) for s in (1.0, 1.0, 6.0)]
    assert eq.rank_normalized_rhat(chains) > 1.05


def test_rhat_requires_at_least_two_chains():
    with pytest.raises(ValueError):
        eq.gelman_rubin([np.arange(100.0)])


def test_detect_equil_rhat_finds_shared_transient():
    rng = np.random.default_rng(8)
    chains = [np.concatenate([np.linspace(6, 0, 200) + rng.normal(0, .3, 200),
                              rng.normal(0, .3, 800)]) for _ in range(3)]
    t0 = eq.detect_equil_rhat(chains, threshold=1.02)
    assert 0 < t0 < 400


def test_detect_equil_rhat_returns_minus_one_when_never_converged():
    rng = np.random.default_rng(9)
    chains = [rng.normal(m, 0.05, 500) for m in (0.0, 5.0, 10.0)]   # never agree
    assert eq.detect_equil_rhat(chains, threshold=1.01) == -1


# ---------------------------------------------------------------------------
# Stabilising wrappers
# ---------------------------------------------------------------------------

def test_monotone_is_a_running_maximum():
    assert eq.monotone([10, 50, 12, 300, 45, 280, 8]) == [10, 50, 50, 300, 300, 300, 300]


def test_monotone_prevents_window_regression():
    """The real failure: a low reading re-admits blocks that had aged out."""
    raw = [400, 420, 180, 430]          # third reading collapses backwards
    assert eq.monotone(raw)[2] >= eq.monotone(raw)[1]


def test_freeze_after_holds_value():
    assert eq.freeze_after([10, 50, 12, 300, 45], 3) == [10, 50, 12, 300, 300]


def test_freeze_after_beyond_length_is_identity():
    seq = [1, 2, 3]
    assert eq.freeze_after(seq, 99) == seq


def test_trailing_window_matches_quarter_rule():
    assert eq.trailing_window(80, 0.25) == 60
    assert eq.trailing_window(48, 0.25) == 36


def test_trailing_window_is_monotone_in_sim_no():
    vals = [eq.trailing_window(s, 0.25) for s in range(10, 100)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# compare_detectors
# ---------------------------------------------------------------------------

def test_compare_detectors_returns_every_detector():
    x = make_series(150, 850)
    out = eq.compare_detectors(x)
    assert set(out) == set(eq.DETECTORS)
    for name, rec in out.items():
        assert rec['error'] is None, f'{name} raised: {rec["error"]}'
        assert 0 <= rec['discard'] <= 1


def test_compare_detectors_records_failure_rather_than_raising():
    def boom(_s, **_k):
        raise RuntimeError('nope')
    out = eq.compare_detectors(np.arange(100.0), detectors={'boom': boom})
    assert out['boom']['t0'] is None
    assert 'RuntimeError' in out['boom']['error']


# ---------------------------------------------------------------------------
# bound_pool
#
# Pool size, not replica mixing, is what predicts MBAR weight collapse. The
# walker-diversity hypothesis was tested on NTSR1_ML-301.hmr.rep0 and refuted
# (collapsed blocks had MORE walkers reaching 310 K, p=0.9996); pool size
# separates cleanly (Spearman rho=0.798, p=4e-18).
# ---------------------------------------------------------------------------

def test_bound_pool_caps_a_late_t0():
    # block 29 of NTSR1 rep0: t0=269 of 270 frames left a 1-frame pool
    assert eq.bound_pool(269, 270, 35) == 235


def test_bound_pool_leaves_a_healthy_t0_untouched():
    assert eq.bound_pool(50, 180, 35) == 50


def test_bound_pool_is_never_negative_on_short_series():
    """Early blocks simply do not have min_pool frames yet."""
    assert eq.bound_pool(19, 26, 35) == 0


def test_bound_pool_disabled_when_min_pool_falsy():
    for v in (None, 0):
        assert eq.bound_pool(269, 270, v) == 269


def test_bound_pool_guarantees_the_pool_when_data_allows():
    for t0, n in ((269, 270), (662, 691), (213, 220)):
        assert n - eq.bound_pool(t0, n, 35) >= 35


def test_bound_pool_cap_loosens_as_the_trajectory_grows():
    """The bound constrains early blocks and late-branch excursions only."""
    caps = [eq.bound_pool(10_000, n, 35) for n in (100, 200, 400, 800)]
    assert caps == sorted(caps)
    assert caps == [65, 165, 365, 765]


def test_bound_pool_composes_with_monotone():
    """Ratchet first, then cap: the safety bound must win."""
    raw = [100, 260, 180]
    ratcheted = eq.monotone(raw)
    capped = [eq.bound_pool(t, 270, 35) for t in ratcheted]
    assert max(capped) <= 235


# ---------------------------------------------------------------------------
# structural_series helpers
#
# The criterion detects equilibration on energy but tests structure. The RMSD
# kernel underneath the structural observable is validated here; the series
# builder itself needs real positions.npy and is exercised by the CLI.
# ---------------------------------------------------------------------------

def test_rmsd_is_zero_for_identical_frames():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (20, 3))
    assert eq._rmsd_after_superposition(x, x) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_is_invariant_to_translation():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (20, 3))
    assert eq._rmsd_after_superposition(x + 17.0, x) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_is_invariant_to_rotation():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (20, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert eq._rmsd_after_superposition(x @ R.T, x) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_grows_with_displacement():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (30, 3))
    small = eq._rmsd_after_superposition(x + rng.normal(0, 0.05, x.shape), x)
    large = eq._rmsd_after_superposition(x + rng.normal(0, 1.00, x.shape), x)
    assert small < large


def test_rmsd_does_not_mirror():
    """A reflection is not a rotation; Kabsch must not report it as identical."""
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (25, 3))
    mirrored = x * np.array([1.0, 1.0, -1.0])
    assert eq._rmsd_after_superposition(mirrored, x) > 1e-6
