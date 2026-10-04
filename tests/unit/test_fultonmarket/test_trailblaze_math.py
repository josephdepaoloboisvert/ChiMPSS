"""Unit tests for the stepping maths in chimpss.fultonmarket.trailblaze.

The module imports openmm only inside OpenMMRungSampler, so these tests run
without it: they check the step formula, the endpoint rules, the Metropolis
acceptance estimator against its Gaussian closed form, and the dry walk on
a constant heat capacity, which must come out geometric.
"""
import numpy as np
import pytest

from chimpss.fultonmarket import trailblaze as tb

CV = 1.7e5


def test_constants_and_aliases():
    assert tb.KB == pytest.approx(0.00831446261815324, rel=1e-9)
    assert tb.target_spacing(0.4) == pytest.approx(1.190, abs=1e-3)
    assert tb.predicted_acceptance(tb.target_spacing(0.4)) == pytest.approx(0.4)
    assert tb.sigma_from_cv(310.0, CV) == pytest.approx(tb.KB * 310.0 * np.sqrt(CV))


def test_pair_spacing_matches_pair_lengths():
    assert tb.pair_spacing(310.0, 311.0, 1000.0, 1010.0) == pytest.approx(
        0.5 * 2010.0 * (tb.beta(310.0) - tb.beta(311.0)))


# ---------------------------------------------------------------------------
# next_temperature
# ---------------------------------------------------------------------------

def test_next_temperature_is_exact_in_beta():
    T_o, sigma, dL = 310.0, 1050.0, 1.19
    T_n, crossed = tb.next_temperature(T_o, sigma, dL, 367.0)
    assert not crossed
    assert tb.beta(T_o) - tb.beta(T_n) == pytest.approx(dL / sigma, rel=1e-12)
    # first order: dT = dL k_B T^2 / sigma
    assert T_n - T_o == pytest.approx(dL * tb.KB * T_o ** 2 / sigma, rel=1e-2)


def test_next_temperature_step_scale_shrinks_the_step():
    full, _ = tb.next_temperature(310.0, 1050.0, 1.19, 367.0)
    small, _ = tb.next_temperature(310.0, 1050.0, 1.19, 367.0, step_scale=1 / 1.25)
    assert 0 < small - 310.0 < full - 310.0
    assert (small - 310.0) == pytest.approx((full - 310.0) / 1.25, rel=2e-3)


def test_next_temperature_caps_and_refuses():
    T_n, _ = tb.next_temperature(310.0, 1.0, 1.19, 367.0, max_dT=10.0)      # tiny sigma -> huge step
    assert T_n == 320.0
    T_n, _ = tb.next_temperature(310.0, 10.0, 1.19, 367.0, max_dT=10.0)     # denominator negative
    assert T_n == 320.0
    with pytest.raises(ValueError, match='min_dT'):
        tb.next_temperature(310.0, 1e6, 1.19, 367.0, min_dT=0.1)
    with pytest.raises(ValueError):
        tb.next_temperature(310.0, np.nan, 1.19, 367.0)
    with pytest.raises(ValueError):
        tb.next_temperature(370.0, 1000.0, 1.19, 367.0)


def test_next_temperature_endpoint_rules():
    # reaching T_max: placed there, crossed
    T_n, crossed = tb.next_temperature(366.5, 1300.0, 1.19, 367.0)
    assert T_n == 367.0 and crossed
    # a sliver shorter than half a step remains: split the remainder in two instead
    sigma = 1300.0
    full_step = tb.next_temperature(360.0, sigma, 1.19, 1000.0)[0] - 360.0
    T_max = 360.0 + 1.3 * full_step
    T_n, crossed = tb.next_temperature(360.0, sigma, 1.19, T_max)
    assert not crossed
    assert T_n == pytest.approx(360.0 + 0.65 * full_step)
    T_last, crossed = tb.next_temperature(T_n, sigma, 1.19, T_max)
    assert crossed and T_last == T_max
    # a remainder longer than half a step is left for the next rung
    T_max = 360.0 + 1.7 * full_step
    T_n, crossed = tb.next_temperature(360.0, sigma, 1.19, T_max)
    assert not crossed and T_n == pytest.approx(360.0 + full_step)


# ---------------------------------------------------------------------------
# pair_acceptance and stationary_window
# ---------------------------------------------------------------------------

def gaussian_samples(T, n, rng, c0=-1.0e6):
    return rng.normal(c0 + CV * tb.KB * T, tb.KB * T * np.sqrt(CV), size=n)


@pytest.mark.parametrize('dL', [0.5, 1.19, 2.0])
def test_pair_acceptance_matches_erfc(dL):
    rng = np.random.default_rng(int(dL * 100))
    T_o = 310.0
    r = np.exp(dL / np.sqrt(CV))
    T_n = T_o * r
    acc = tb.pair_acceptance(gaussian_samples(T_o, 2000, rng), gaussian_samples(T_n, 2000, rng), T_o, T_n)
    dL_exact = np.sqrt(CV) * (r ** 2 - 1.0) / (2.0 * r)
    assert acc == pytest.approx(tb.predicted_acceptance(dL_exact), abs=0.02)


def test_pair_acceptance_sign_and_subsampling():
    rng = np.random.default_rng(0)
    H = gaussian_samples(310.0, 300, rng)
    assert tb.pair_acceptance(H, H, 310.0, 310.0) == 1.0           # identical rungs always exchange
    hot = gaussian_samples(320.0, 300, rng)
    assert tb.pair_acceptance(H, hot, 310.0, 320.0) < 0.05         # far apart: hardly ever
    full = tb.pair_acceptance(H, hot + 2000.0, 310.0, 311.0)
    sub = tb.pair_acceptance(H, hot + 2000.0, 310.0, 311.0, max_pairs=20_000, rng=np.random.default_rng(1))
    assert sub == pytest.approx(full, abs=0.02)
    with pytest.raises(ValueError):
        tb.pair_acceptance(np.array([np.nan]), hot, 310.0, 311.0)


def test_stationary_window_detects_a_step():
    rng = np.random.default_rng(2)
    H = np.concatenate([rng.normal(5.0, 1.0, 30), rng.normal(0.0, 1.0, 120)])
    t0, g, neff = tb.stationary_window(H)
    assert 20 <= t0 <= 45
    assert g >= 1.0 and 0 < neff <= 150 - t0 + 1e-9


def test_stationary_window_fallback_without_pymbar(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_pymbar(name, *a, **k):
        if name.startswith('pymbar'):
            raise ImportError('pymbar hidden')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', no_pymbar)
    rng = np.random.default_rng(3)
    drift = np.concatenate([rng.normal(5.0, 1.0, 50), rng.normal(0.0, 1.0, 50)])
    t0, g, neff = tb.stationary_window(drift)
    assert t0 == 50 and g == 1.0 and neff == 50
    flat = rng.normal(0.0, 1.0, 100)
    assert tb.stationary_window(flat)[0] == 0
    assert tb.stationary_window(flat[:3]) == (0, 1.0, 3.0)


def test_stationary_window_keeps_a_minimum():
    rng = np.random.default_rng(4)
    H = np.concatenate([np.linspace(10, 0, 40), rng.normal(0.0, 1.0, 10)])
    t0, _, _ = tb.stationary_window(H, min_keep=20)
    assert H.size - t0 >= 20


# ---------------------------------------------------------------------------
# dry walk and ladder files
# ---------------------------------------------------------------------------

def test_dry_walk_on_constant_cv_is_geometric():
    dL = tb.target_spacing(0.4)
    T = tb.ladder_from_sigma(310.0, 367.0, lambda t: tb.sigma_from_cv(t, CV), dL)
    assert T[0] == 310.0 and T[-1] == 367.0
    assert np.all(np.diff(T) > 0)
    ratios = T[1:-2] / T[:-3]                     # the last two pairs may be evened out
    eps = dL * tb.KB * 310.0 / tb.sigma_from_cv(310.0, CV)
    assert ratios == pytest.approx(np.full(ratios.size, 1.0 / (1.0 - eps)), rel=1e-12)
    L_total = np.sqrt(CV) * np.log(367.0 / 310.0)
    assert T.size == pytest.approx(L_total / dL + 1, abs=1.5)
    assert 55 <= T.size <= 60
    with pytest.raises(RuntimeError):
        tb.ladder_from_sigma(310.0, 367.0, lambda t: tb.sigma_from_cv(t, CV), 0.01, max_rungs=100, min_dT=1e-4)


def test_load_ladder_file_validates(tmp_path):
    good = np.array([310.0, 311.0, 312.5])
    np.save(tmp_path / 'ok.npy', good)
    assert np.array_equal(tb.load_ladder_file(str(tmp_path / 'ok.npy')), good)
    (tmp_path / 'ok.txt').write_text('310\n311\n312.5\n')
    assert np.array_equal(tb.load_ladder_file(str(tmp_path / 'ok.txt')), good)
    for bad in ([310.0], [310.0, 309.0], [310.0, 310.004], [310.0, np.inf]):
        np.save(tmp_path / 'bad.npy', np.array(bad))
        with pytest.raises(ValueError):
            tb.load_ladder_file(str(tmp_path / 'bad.npy'))
