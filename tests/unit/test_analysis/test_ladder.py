"""Unit tests for chimpss.analysis.ladder.

Checked against synthetic parallel-tempering blocks with a known heat
capacity, where every quantity the module reports has a closed form: a
constant C_V makes sigma_H linear in T, the thermodynamic length of a
ratio-r ladder is sqrt(C_V) (r**2 - 1) / (2 r) for every pair, the cumulative
length is sqrt(C_V) ln(T / T_0), and the neighbour acceptance is erfc(dL / 2).
"""
import json
import os

import numpy as np
import pytest

from chimpss.analysis import ladder as L

R = L.R_KJ


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def geometric_ladder(T0, r, K):
    return T0 * r ** np.arange(K)


def exp_in_index_ladder(lo, hi, K):
    """The engine's ``geometric_distribution``: gaps grow e-fold along the ladder."""
    i = np.arange(K)
    return lo + (hi - lo) * (np.exp(i / (K - 1)) - 1.0) / (np.e - 1.0)


def gaussian_pt_blocks(T, cv, n_iter, n_blocks, seed=0, c0=-1.0e6, dup_iter0=True):
    """Float32 reduced potentials and state records for a PT run with constant C_V.

    ``H_k ~ N(c0 + cv R T_k, R T_k sqrt(cv))`` so that ``Var(H) = (R T)**2 cv``,
    i.e. ``C_V = cv`` in units of k_B and the mean rises with slope ``cv R``.
    Iteration 0 of every block carries the engine's restart signature: one
    configuration (drawn at the coldest rung) on every replica.
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(T, dtype=np.float64)
    K = T.size
    b = L.betas(T)
    mean = c0 + cv * R * T
    sd = R * T * np.sqrt(cv)
    energies, states = [], []
    for _ in range(n_blocks):
        n = n_iter if np.isscalar(n_iter) else int(rng.choice(n_iter))
        s = np.array([rng.permutation(K) for _ in range(n)], dtype=np.int32)
        Hk = rng.normal(mean, sd, size=(n, K))
        if dup_iter0:
            Hk[0] = rng.normal(mean[0], sd[0])
        H = np.take_along_axis(Hk, s, axis=1)     # replica r carries rung states[t, r]
        u = (H[:, :, None] * b[None, None, :]).astype(np.float32)
        energies.append(u)
        states.append(s)
    return energies, states


def write_fake_run(root, T, energies, states, name='run'):
    run_dir = os.path.join(root, name)
    for i, (u, s) in enumerate(zip(energies, states)):
        d = os.path.join(run_dir, 'saved_variables', str(i))
        os.makedirs(d)
        np.save(os.path.join(d, 'temperatures.npy'), np.asarray(T, dtype=np.float64))
        np.save(os.path.join(d, 'energies.npy'), u)
        np.save(os.path.join(d, 'states.npy'), s)
    return run_dir


CV = 1.7e5          # k_B units, the campaign's order of magnitude
T0 = 310.0


@pytest.fixture(scope='module')
def synthetic():
    r = np.exp(1.19 / np.sqrt(CV))         # dL = 1.19 per pair to first order
    T = geometric_ladder(T0, r, 8)
    energies, states = gaussian_pt_blocks(T, CV, n_iter=19, n_blocks=200, seed=1)
    Hk_raw = [L.by_state(L.enthalpy_from_reduced(u, T), s) for u, s in zip(energies, states)]
    return dict(T=T, r=r, energies=energies, states=states, Hk=[h[1:] for h in Hk_raw], Hk_raw=Hk_raw)


# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------

def test_gas_constant_matches_scipy():
    import scipy.constants
    assert L.R_KJ == pytest.approx(scipy.constants.R / 1000.0, rel=1e-9)


def test_length_for_acceptance_inverts_prediction():
    for dL in (0.3, 0.95, 1.19, 1.32, 2.5):
        assert L.length_for_acceptance(L.predicted_acceptance(dL)) == pytest.approx(dL, rel=1e-10)
    assert L.length_for_acceptance(0.5) == pytest.approx(0.954, abs=1e-3)
    assert L.length_for_acceptance(0.4) == pytest.approx(1.190, abs=1e-3)
    assert L.length_for_acceptance(0.35) == pytest.approx(1.322, abs=1e-3)


def test_length_for_acceptance_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            L.length_for_acceptance(bad)


def test_shenfeld_bounds_bracket_the_gaussian_prediction_at_practical_spacings():
    """The bounds are second order in the spacing: valid below dL ~ 1.7, not beyond."""
    dL = np.linspace(0.01, 1.6, 200)
    lower, upper = L.acceptance_bounds(dL)
    pred = L.predicted_acceptance(dL)
    assert np.all(lower <= pred + 1e-12)
    assert np.all(pred <= upper + 1e-12)
    lower_far, upper_far = L.acceptance_bounds(3.0)
    assert L.predicted_acceptance(3.0) > upper_far     # the expansion has broken down


# ---------------------------------------------------------------------------
# Enthalpies from saved arrays
# ---------------------------------------------------------------------------

def test_enthalpy_round_trip(synthetic):
    T, u, s = synthetic['T'], synthetic['energies'][0], synthetic['states'][0]
    H = L.enthalpy_from_reduced(u, T)
    # the exact H is recoverable from any column; the mean must match to float32 noise / sqrt(K)
    exact = np.asarray(u[:, :, 0], dtype=np.float64) / L.betas(T)[0]
    assert np.abs(H - exact).max() < 0.2


def test_enthalpy_rejects_non_pt_energies(synthetic):
    T, u = synthetic['T'], np.array(synthetic['energies'][0], dtype=np.float64)
    u[:, :, 3] *= 1.001
    with pytest.raises(ValueError, match='not pure-temperature'):
        L.enthalpy_from_reduced(u, T)


def test_enthalpy_marks_nonfinite_rows(synthetic):
    T, u = synthetic['T'], np.array(synthetic['energies'][0], dtype=np.float64)
    u[4, 2, 5] = np.nan
    H = L.enthalpy_from_reduced(u, T)
    assert np.isnan(H[4]).all()
    assert np.isfinite(H[[0, 1, 2, 3, 5]]).all()


def test_by_state_undoes_a_permutation():
    rng = np.random.default_rng(0)
    n, K = 5, 6
    s = np.array([rng.permutation(K) for _ in range(n)])
    Hk_true = rng.normal(size=(n, K))
    H = np.take_along_axis(Hk_true, s, axis=1)
    assert np.array_equal(L.by_state(H, s), Hk_true)


def test_by_state_rejects_non_permutation():
    s = np.array([[0, 1, 1]])
    with pytest.raises(ValueError, match='permutation'):
        L.by_state(np.zeros((1, 3)), s)


# ---------------------------------------------------------------------------
# Metric, lengths, placement on closed forms
# ---------------------------------------------------------------------------

def test_metric_profile_recovers_heat_capacity(synthetic):
    prof = L.metric_profile(synthetic['T'], synthetic['Hk'])
    assert prof['C_V'] == pytest.approx(np.full(8, CV), rel=0.05)
    assert prof['sigma_H'] == pytest.approx(R * synthetic['T'] * np.sqrt(CV), rel=0.05)
    assert prof['n_samples'][0] == 200 * 18
    # with no drift, total and within-block variance agree
    assert np.allclose(prof['sigma_total'] / prof['sigma_within'], 1.0, atol=0.02)


def test_metric_profile_within_variance_ignores_drift():
    T = geometric_ladder(T0, 1.003, 4)
    rng = np.random.default_rng(3)
    sd = R * T * np.sqrt(CV)
    blocks = [rng.normal(CV * R * T + 50 * sd * i, sd, size=(19, 4)) for i in range(20)]
    prof = L.metric_profile(T, blocks, variance='within')
    assert prof['sigma_H'] == pytest.approx(sd, rel=0.1)
    assert np.all(prof['sigma_total'] > 10 * prof['sigma_within'])


def test_pair_lengths_constant_on_a_geometric_ladder():
    r = 1.004
    T = geometric_ladder(T0, r, 30)
    sigma = R * T * np.sqrt(CV)
    dL = L.pair_lengths(T, sigma)
    expected = np.sqrt(CV) * (r ** 2 - 1.0) / (2.0 * r)
    assert dL == pytest.approx(np.full(29, expected), rel=1e-12)


def test_cumulative_length_is_logarithmic_for_constant_cv():
    T = np.linspace(310.0, 367.0, 4000)
    sigma = R * T * np.sqrt(CV)
    Lc = L.cumulative_length(T, sigma)
    assert Lc[0] == 0.0
    assert Lc == pytest.approx(np.sqrt(CV) * np.log(T / T[0]), rel=1e-6, abs=1e-9)


def test_pooled_acceptance_matches_erfc(synthetic):
    T, r = synthetic['T'], synthetic['r']
    dL = np.sqrt(CV) * (r ** 2 - 1.0) / (2.0 * r)
    expected = float(L.predicted_acceptance(dL))
    pooled = L.pair_acceptance_from_enthalpy(T, synthetic['Hk'], mode='pooled')
    simul = L.pair_acceptance_from_enthalpy(T, synthetic['Hk'], mode='simultaneous')
    assert pooled['acceptance'] == pytest.approx(np.full(7, expected), abs=0.02)
    assert simul['acceptance'] == pytest.approx(np.full(7, expected), abs=0.03)
    assert pooled['n_samples'][0] == 200 * 18 ** 2
    assert simul['n_samples'][0] == 200 * 18


def test_u_form_and_h_form_agree(synthetic):
    T = synthetic['T']
    u_form = L.pair_acceptance_from_energies(synthetic['energies'][:20], synthetic['states'][:20],
                                             skip=1, mode='pooled')['acceptance']
    h_form = L.pair_acceptance_from_enthalpy(T, synthetic['Hk'][:20], mode='pooled')['acceptance']
    assert u_form == pytest.approx(h_form, abs=0.01)


def test_acceptance_on_a_hand_built_case():
    """Two rungs, two replicas, three iterations, computed by hand in both forms."""
    T = np.array([300.0, 303.0])
    b = L.betas(T)
    Hk = np.array([[-1000.0, -900.0],
                   [-1050.0, -1000.0],
                   [-980.0, -1100.0]])          # rung order, per iteration
    states = np.array([[0, 1], [1, 0], [0, 1]])
    H = np.take_along_axis(Hk, states, axis=1)
    u = H[:, :, None] * b[None, None, :]
    delta = (b[0] - b[1]) * (Hk[:, 1] - Hk[:, 0])
    expected = np.minimum(1.0, np.exp(-delta)).mean()
    assert expected < 1.0  # the third iteration accepts with probability 1, the others do not
    got_h = L.pair_acceptance_from_enthalpy(T, Hk, mode='simultaneous')['acceptance'][0]
    got_u = L.pair_acceptance_from_energies(u, states, skip=0, mode='simultaneous')['acceptance'][0]
    assert got_h == pytest.approx(expected, rel=1e-12)
    assert got_u == pytest.approx(expected, rel=1e-9)
    pooled_expected = np.minimum(1.0, np.exp(-(b[0] - b[1]) * (Hk[None, :, 1] - Hk[:, None, 0]))).mean()
    assert L.pair_acceptance_from_enthalpy(T, Hk, mode='pooled')['acceptance'][0] == pytest.approx(pooled_expected)


def test_respace_equalises_an_exponential_in_index_ladder():
    T = exp_in_index_ladder(310.0, 367.0, 110)
    sigma = R * T * np.sqrt(CV)
    before = L.pair_lengths(T, sigma)
    assert np.std(before) / np.mean(before) > 0.2          # the campaign's non-uniformity
    out = L.respace(T, sigma, n_rungs=110)
    after = out['pair_dL']
    assert out['n_rungs'] == 110
    assert out['temperatures'][0] == 310.0 and out['temperatures'][-1] == 367.0
    assert np.all(np.diff(out['temperatures']) > 0)
    assert np.std(after) / np.mean(after) < 1e-3
    assert after.sum() == pytest.approx(before.sum(), rel=2e-3)   # same total length
    assert not out['extrapolated']


def test_respace_by_target_acceptance_sets_the_rung_count():
    T = np.linspace(310.0, 367.0, 50)
    sigma = R * T * np.sqrt(CV)
    out = L.respace(T, sigma, target_acceptance=0.4)
    L_total = np.sqrt(CV) * np.log(367.0 / 310.0)
    K_expected = int(np.ceil(L_total / L.length_for_acceptance(0.4))) + 1
    assert out['n_rungs'] == K_expected
    assert out['L_total'] == pytest.approx(L_total, rel=1e-5)
    assert out['dL_actual'] == pytest.approx(L_total / (K_expected - 1), rel=1e-5)
    assert np.all(out['predicted_acceptance'] >= 0.4 - 0.01)
    # a constant-C_V profile gives a constant-ratio ladder
    ratios = out['temperatures'][1:] / out['temperatures'][:-1]
    assert np.std(ratios) / np.mean(ratios) < 1e-4


def test_respace_requires_exactly_one_target():
    T = np.linspace(310.0, 367.0, 10)
    sigma = R * T * np.sqrt(CV)
    with pytest.raises(ValueError):
        L.respace(T, sigma)
    with pytest.raises(ValueError):
        L.respace(T, sigma, n_rungs=10, target_dL=1.0)


def test_respace_collision_guard():
    T = np.linspace(310.0, 367.0, 10)
    sigma = R * T * np.sqrt(CV)
    with pytest.raises(ValueError, match='collide'):
        L.respace(T, sigma, n_rungs=20000)


def test_respace_flags_extrapolation():
    T = np.linspace(320.0, 360.0, 10)
    sigma = R * T * np.sqrt(CV)
    out = L.respace(T, sigma, n_rungs=20, T_min=310.0, T_max=367.0)
    assert out['extrapolated']
    assert out['temperatures'][0] == 310.0 and out['temperatures'][-1] == 367.0


def test_smooth_sigma_recovers_a_linear_heat_capacity():
    T = np.linspace(310.0, 367.0, 60)
    a, b = 1.0e5, 250.0
    sigma = R * T * np.sqrt(a + b * T)
    rng = np.random.default_rng(5)
    noisy = sigma * (1 + 0.02 * rng.standard_normal(T.size))
    fit, info = L.smooth_sigma(T, noisy, n_samples=np.full(T.size, 1000))
    assert info['model'] == 'cv_linear'
    assert fit == pytest.approx(sigma, rel=0.01)
    assert info['cv_change_fraction'] == pytest.approx((a + b * 367) / (a + b * 310) - 1, abs=0.02)
    # the intercept extrapolates 310 K back to zero, so it needs quiet data to pin down
    quiet = sigma * (1 + 0.002 * rng.standard_normal(T.size))
    _, info_q = L.smooth_sigma(T, quiet)
    assert info_q['a'] == pytest.approx(a, rel=0.05)
    assert info_q['b'] == pytest.approx(b, rel=0.2)
    same, info_none = L.smooth_sigma(T, noisy, model='none')
    assert np.array_equal(same, noisy) and info_none['model'] == 'none'


def test_virtual_acceptance_equals_pooled_at_sampled_rungs(synthetic):
    T, Hk = synthetic['T'], synthetic['Hk'][:40]
    pooled = L.pair_acceptance_from_enthalpy(T, Hk, mode='pooled')['acceptance']
    v = L.virtual_pair_acceptance(T[2], T[3], T, Hk)
    assert v['acceptance'] == pytest.approx(pooled[2], rel=1e-10)
    assert v['ess_fraction'] == pytest.approx(1.0)
    assert v['source_rungs'] == (2, 3)
    # a pair half as far apart accepts more often
    mid = 0.5 * (T[2] + T[3])
    closer = L.virtual_pair_acceptance(T[2], mid, T, Hk)
    assert closer['acceptance'] > pooled[2]
    assert closer['ess_fraction'] > 0.5


# ---------------------------------------------------------------------------
# Loading runs, reports, block audit
# ---------------------------------------------------------------------------

def test_load_run_skips_blocks_with_another_ladder(tmp_path):
    T8 = geometric_ladder(T0, 1.004, 8)
    T9 = np.sort(np.append(T8, 320.0))
    e8, s8 = gaussian_pt_blocks(T8, CV, n_iter=[17, 19], n_blocks=2, seed=2)
    e9, s9 = gaussian_pt_blocks(T9, CV, n_iter=19, n_blocks=3, seed=3)
    run_dir = write_fake_run(tmp_path, T8, e8, s8)
    for i, (u, s) in enumerate(zip(e9, s9), start=2):
        d = os.path.join(run_dir, 'saved_variables', str(i))
        os.makedirs(d)
        np.save(os.path.join(d, 'temperatures.npy'), T9)
        np.save(os.path.join(d, 'energies.npy'), u)
        np.save(os.path.join(d, 'states.npy'), s)
    data = L.load_run(run_dir, skip=1)
    assert data.blocks_used == [2, 3, 4]
    assert set(data.blocks_skipped) == {0, 1}
    assert data.temperatures.shape == (9,)
    assert data.n_samples_per_rung == 3 * 18
    sub = L.load_run(run_dir, blocks=['3', '4'], skip=2)
    assert sub.blocks_used == [3, 4] and sub.n_samples_per_rung == 2 * 17


def test_load_run_handles_mixed_block_lengths(tmp_path):
    T = geometric_ladder(T0, 1.004, 6)
    e, s = gaussian_pt_blocks(T, CV, n_iter=[16, 17, 19], n_blocks=12, seed=4)
    run_dir = write_fake_run(tmp_path, T, e, s)
    data = L.load_run(run_dir)
    assert sorted(set(data.n_iterations)) == sorted(set(h.shape[0] for h in e))
    assert data.n_samples_per_rung == sum(h.shape[0] - 1 for h in e)


def test_ladder_report_is_json_serialisable_and_flags_the_restart(tmp_path):
    r = np.exp(1.19 / np.sqrt(CV))
    T = geometric_ladder(T0, r, 8)
    e, s = gaussian_pt_blocks(T, CV, n_iter=19, n_blocks=30, seed=6)
    run_dir = write_fake_run(tmp_path, T, e, s)
    rep = L.ladder_report(run_dir, targets=(0.35, 0.4))
    json.dumps(rep)
    summ = rep['summary']
    assert summ['n_rungs'] == 8 and len(rep['pairs']) == 7 and len(rep['rungs']) == 8
    assert summ['K_needed'].keys() == {'0.35', '0.40'}
    assert summ['A_meas_mean'] == pytest.approx(0.4, abs=0.05)
    assert summ['dL_cv'] < 0.05
    assert summ['bounds_violations'] == 0
    assert abs(summ['calibration_c'] - 1.0) < 0.1
    prof = summ['iteration_profile_sigma']
    assert prof[0] < -3.0                      # identical configuration on every replica
    assert abs(prof[1]) < 0.5
    text = L.format_table(rep, max_rows=3)
    assert 'rungs needed' in text and '... 4 more pairs' in text


def test_block_audit_accumulates_a_trailing_window(tmp_path):
    T = geometric_ladder(T0, 1.004, 6)
    e, s = gaussian_pt_blocks(T, CV, n_iter=19, n_blocks=4, seed=7)
    run_dir = write_fake_run(tmp_path, T, e, s)
    hist = os.path.join(run_dir, 'analysis', 'ladder')
    os.makedirs(hist)
    for b in range(4):
        out = L.block_audit(run_dir, b, trailing=3, threshold=0.5)
        with open(os.path.join(hist, f'{b:04d}.json'), 'w') as fh:
            json.dump(out, fh)
    assert out['trailing_blocks'] == [3, 2, 1]
    assert out['n_iterations_used'] == 18
    assert len(out['A_trailing']) == 5
    assert 'pairs_below_threshold' in out
    assert out['iteration_profile_sigma'][0] < -3.0


def test_parse_blocks():
    assert L.parse_blocks(None) is None
    assert L.parse_blocks(['3', '7:10', '20']) == [3, 7, 8, 9, 20]


# ---------------------------------------------------------------------------
# Adaptive insertion test
# ---------------------------------------------------------------------------

def per_iteration_acceptance(true_acc, n_iter, K, seed, spread=0.15):
    """Per-iteration neighbour acceptance around a true value, as the reporter would record it."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n_iter, K, K))
    idx = np.arange(K - 1)
    a[:, idx, idx + 1] = np.clip(rng.normal(true_acc, spread, size=(n_iter, K - 1)), 0, 1)
    return a


def test_mean_rule_splits_a_sound_ladder_on_five_iterations():
    a = per_iteration_acceptance(0.40, n_iter=5, K=60, seed=1)
    res = L.insertion_candidates(a, 0.30, rule='mean')
    assert 3 <= res['insert_after'].size <= 20           # noise alone flags a good fraction
    assert res['n_iters'] == 5 and res['acceptance'].shape == (59,)
    assert np.array_equal(res['upper'], res['acceptance'])


def test_ucb_rule_waits_and_then_spares_a_sound_ladder():
    a = per_iteration_acceptance(0.40, n_iter=5, K=60, seed=1)
    assert L.insertion_candidates(a, 0.30, rule='ucb', min_iters=10)['insert_after'].size == 0
    a = per_iteration_acceptance(0.40, n_iter=18, K=60, seed=2)
    res = L.insertion_candidates(a, 0.30, rule='ucb', min_iters=10)
    assert res['insert_after'].size <= 1
    assert np.all(res['upper'] >= res['acceptance'])


def test_ucb_rule_still_catches_a_bad_pair():
    a = per_iteration_acceptance(0.40, n_iter=18, K=12, seed=3)
    a[:, 5, 6] = np.clip(np.random.default_rng(4).normal(0.08, 0.05, 18), 0, 1)
    res = L.insertion_candidates(a, 0.30, rule='ucb', min_iters=10)
    assert 5 in res['insert_after']
    assert res['insert_after'].size <= 2


def test_insertion_candidates_validates():
    with pytest.raises(ValueError):
        L.insertion_candidates(np.zeros((3, 4, 4)), 0.3, rule='median')
    with pytest.raises(ValueError):
        L.insertion_candidates(np.zeros((3, 4)), 0.3)
    one = L.insertion_candidates(np.full((1, 4, 4), 0.1), 0.3, rule='ucb', min_iters=1)
    assert one['insert_after'].size == 0 and np.all(np.isinf(one['upper']))
