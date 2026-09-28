"""Unit tests for chimpss.analysis.convergence_panel.

Every gate is exercised on synthetic ensembles whose answer is known by
construction: a stationary process must pass and keep passing as it lengthens,
a drifting one must fail, and a two-basin ensemble with shifting populations
must fail the population gate while a stable one passes it.
"""
import numpy as np
import pytest

from chimpss.analysis import convergence_panel as cp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ar1(n, rho, rng, sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def make_tol(n_pc=10, sd_pc=3.0):
    C = {'Inactive': np.full(n_pc, -17.0), 'Intermediate': np.full(n_pc, -7.0),
         'Active': np.full(n_pc, 5.0)}
    for k in C:
        C[k][1:] = 0.0
    W = {k: np.full(n_pc, sd_pc) for k in C}
    return cp.Tolerances(same_state_rmsd=0.7, family_same_state_rmsd=0.7, diff_state_rmsd=2.0,
                         centroids=C, within_state_sd=W, receptor='test', receptor_n_structures=4,
                         state_counts={'Inactive': 180, 'Intermediate': 12, 'Active': 500})


def synth_ensemble(n_blocks, frames_per_block, n_atoms, rng, drift_per_block=0.0,
                   centre_pc1=-7.0, tau_blocks=3.0):
    """Frames scattered 0.6 A RMSD about a slowly wandering mean.

    ``drift_per_block`` moves the mean along one Cartesian direction per block;
    ``P`` is a matching projection whose PC1 tracks the same drift.
    """
    blocks = np.repeat(np.arange(n_blocks), frames_per_block)
    rho = np.exp(-1.0 / tau_blocks)
    wander = ar1(n_blocks, rho, rng, sd=0.05)             # A, block-mean wander
    base = rng.normal(0, 5, (n_atoms, 3))
    X = np.empty((blocks.size, n_atoms, 3))
    P = np.empty((blocks.size, 10))
    for b in range(n_blocks):
        sel = blocks == b
        shift = np.zeros(3)
        shift[0] = wander[b] + drift_per_block * b
        X[sel] = base + shift + rng.normal(0, 0.6, (sel.sum(), n_atoms, 3))
        P[sel, 0] = centre_pc1 + 4.0 * shift[0] + rng.normal(0, 1.0, sel.sum())
        P[sel, 1:] = rng.normal(0, 0.8, (sel.sum(), 9))
    return X, P, blocks


# ---------------------------------------------------------------------------
# series statistics
# ---------------------------------------------------------------------------

def test_tau_int_is_one_for_white_noise():
    rng = np.random.default_rng(0)
    taus = [cp.tau_int(rng.normal(size=2000)) for _ in range(5)]
    assert np.median(taus) < 1.4


def test_tau_int_recovers_ar1_scale():
    rng = np.random.default_rng(1)
    rho = 0.9                                   # tau_int = (1+rho)/(1-rho) = 19
    tau = np.median([cp.tau_int(ar1(20000, rho, rng)) for _ in range(3)])
    assert 13 < tau < 26


def test_ols_corrected_inflates_p_for_autocorrelated_noise():
    """An AR(1) series with no trend: the naive p is anticonservative, the
    corrected one is not."""
    rng = np.random.default_rng(2)
    naive, corr = [], []
    for _ in range(200):
        o = cp.ols_corrected(ar1(120, 0.9, rng))
        naive.append(o['p_naive'] < 0.05)
        corr.append(o['p_corr'] < 0.05)
    # n/tau is ~6 here, so the corrected test is still somewhat anticonservative
    # (Geyer's estimate is noisy at that depth); the claim is the size of the
    # correction, not exact calibration.
    assert np.mean(naive) > 0.25                # far above the nominal 5 %
    assert np.mean(corr) < 0.25
    assert np.mean(corr) < 0.5 * np.mean(naive)


def test_rhat_near_one_for_stationary_and_large_for_drift():
    rng = np.random.default_rng(3)
    assert cp.rhat_split(rng.normal(size=400)) < 1.05
    assert cp.rhat_split(np.linspace(0, 5, 400) + rng.normal(size=400)) > 1.5


def test_fp_block_se_plateau_exceeds_naive_for_correlated_series():
    rng = np.random.default_rng(4)
    x = ar1(4096, 0.9, rng)
    sizes, se, plateau = cp.fp_block_se(x)
    assert plateau > 2 * se[0]


def test_bimodality_coefficient_separates_one_and_two_modes():
    rng = np.random.default_rng(5)
    assert cp.bimodality_coefficient(rng.normal(size=5000)) < cp.BIMODALITY_CUTOFF
    two = np.concatenate([rng.normal(-3, 1, 2500), rng.normal(3, 1, 2500)])
    assert cp.bimodality_coefficient(two) > cp.BIMODALITY_CUTOFF


def test_jsd_1d_is_zero_for_identical_and_bounded():
    rng = np.random.default_rng(6)
    a = rng.normal(size=1000)
    e = np.linspace(-5, 5, 51)
    assert cp.jsd_1d(a, a, e) == 0.0
    assert 0 < cp.jsd_1d(a, a + 3, e) <= 1.0


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def test_interleaved_split_partitions_all_blocks():
    rng = np.random.default_rng(7)
    ub = np.arange(40, 161)
    a, b = cp._interleaved(ub, 8, rng)
    assert sorted(np.concatenate([a, b]).tolist()) == ub.tolist()
    assert abs(a.size - b.size) <= 16


def test_equilibrated_blocks_default_drops_first_quarter():
    blocks = np.repeat(np.arange(100), 3)
    ub = cp.equilibrated_blocks(blocks)
    assert ub[0] == 25 and ub.size == 75


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_mean_structure_drift_stationary_is_at_the_noise_floor():
    rng = np.random.default_rng(8)
    X, P, blocks = synth_ensemble(80, 20, 50, rng)
    ub = cp.equilibrated_blocks(blocks, 0)
    r = cp.mean_structure_drift(X, blocks, ub, n_boot=50, n_null=4)
    assert r['rmsd'] < 0.3
    assert r['rmsd'] < 3 * r['null_p95'] + 0.1


def test_mean_structure_drift_detects_a_planted_shift():
    rng = np.random.default_rng(9)
    X, P, blocks = synth_ensemble(80, 20, 50, rng, drift_per_block=0.02)   # 1.6 A over the run
    ub = cp.equilibrated_blocks(blocks, 0)
    r = cp.mean_structure_drift(X, blocks, ub, n_boot=50, n_null=4)
    # 1.6 A of drift across the run puts the half means ~0.8 A apart, above the
    # 0.69 A experimental tolerance.  (The interleaved floor is not a clean
    # reference under a monotone drift -- chunks carry the drift too -- which
    # is why the panel gates on the tolerance and reports the floor.)
    assert r['rmsd'] > 0.6
    assert r['boot_q95'] > 0.69


def test_state_stability_passes_within_one_state():
    rng = np.random.default_rng(10)
    X, P, blocks = synth_ensemble(80, 20, 50, rng)
    ub = cp.equilibrated_blocks(blocks, 0)
    r = cp.state_stability(P, blocks, ub, make_tol())
    assert r['same_state'] and r['state'] == 'Intermediate' and r['pass_']


def test_state_stability_fails_when_the_ensemble_crosses_states():
    rng = np.random.default_rng(11)
    X, P, blocks = synth_ensemble(80, 20, 50, rng)
    P = P.copy()
    P[blocks >= 40, 0] += 12.0                  # second half sits at the Active centroid
    ub = cp.equilibrated_blocks(blocks, 0)
    r = cp.state_stability(P, blocks, ub, make_tol())
    assert not r['same_state'] and not r['pass_']


def test_state_stability_thin_state_borrows_the_tighter_spread():
    tol = make_tol()
    tol.within_state_sd['Intermediate'][:] = 50.0    # absurdly loose
    tol.within_state_sd['Inactive'][:] = 2.0
    tol.within_state_sd['Active'][:] = 4.0
    rng = np.random.default_rng(12)
    X, P, blocks = synth_ensemble(80, 20, 50, rng)
    ub = cp.equilibrated_blocks(blocks, 0)
    r = cp.state_stability(P, blocks, ub, tol)
    assert r['tolerance'][0] == pytest.approx(2.0)


def test_ensemble_width_stationary_and_widened():
    rng = np.random.default_rng(13)
    X, P, blocks = synth_ensemble(80, 20, 50, rng)
    ub = cp.equilibrated_blocks(blocks, 0)
    r0 = cp.ensemble_width(X, blocks, ub, 50, sub=150, n_null=3)
    X2 = X.copy()
    sel = blocks >= 40
    X2[sel] = X2[sel].mean(0) + 1.6 * (X2[sel] - X2[sel].mean(0))   # second half 60 % wider
    r1 = cp.ensemble_width(X2, blocks, ub, 50, sub=150, n_null=3)
    assert r0['w1'] < r0['null_w1_p95'] * 2 + 0.05
    assert r1['w1'] > 3 * r0['w1']


def test_basin_count_one_for_unimodal_and_two_for_bimodal():
    rng = np.random.default_rng(14)
    P1 = rng.normal(size=(4000, 10))
    assert cp.basin_count(P1)['n_basins'] == 1
    P2 = P1.copy()
    P2[:2000, 0] -= 4
    P2[2000:, 0] += 4
    assert cp.basin_count(P2)['n_basins'] == 2


def test_population_free_energies_pass_and_fail():
    rng = np.random.default_rng(15)
    n_blocks, fpb = 80, 30
    blocks = np.repeat(np.arange(n_blocks), fpb)
    P = rng.normal(size=(blocks.size, 10))
    # stable 50/50 two-basin ensemble
    P[:, 0] += np.where(rng.random(blocks.size) < 0.5, -4, 4)
    ub = cp.equilibrated_blocks(blocks, 0)
    ok = cp.population_free_energies(P, blocks, ub, 2, n_boot=50)
    assert ok['pass_']
    # populations 80/20 -> 20/80 between halves: dG ~ 0.85 kcal/mol
    P2 = rng.normal(size=(blocks.size, 10))
    frac = np.where(blocks < 40, 0.8, 0.2)
    P2[:, 0] += np.where(rng.random(blocks.size) < frac, -4, 4)
    bad = cp.population_free_energies(P2, blocks, ub, 2, n_boot=50)
    assert not bad['pass_']
    assert all(abs(b['dG']) > cp.KT_310_KCAL for b in bad['basins'])


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def test_panel_insufficient_below_sixteen_blocks():
    rng = np.random.default_rng(16)
    X, P, blocks = synth_ensemble(12, 10, 30, rng)
    r = cp.panel(X, P, blocks, make_tol(), b0=0)
    assert r['verdict'] == 'INSUFFICIENT'


def test_panel_converged_for_stationary_and_keeps_passing_as_it_lengthens():
    rng = np.random.default_rng(17)
    X, P, blocks = synth_ensemble(120, 20, 50, rng)
    tol = make_tol()
    verdicts = [cp.panel(X[blocks < n], P[blocks < n], blocks[blocks < n], tol, b0=0)['verdict']
                for n in (40, 80, 120)]
    assert verdicts == ['CONVERGED'] * 3


def test_panel_not_converged_for_a_drifting_mean():
    rng = np.random.default_rng(18)
    X, P, blocks = synth_ensemble(80, 20, 50, rng, drift_per_block=0.03)
    r = cp.panel(X, P, blocks, make_tol(), b0=0)
    assert r['verdict'] == 'NOT CONVERGED'
    assert 'mean_structure' in r['failed']


def test_panel_reports_diagnostics_but_does_not_gate_on_them():
    """A stationary but strongly autocorrelated run has R-hat well above 1.1
    on block means; the panel must still read CONVERGED."""
    rng = np.random.default_rng(19)
    X, P, blocks = synth_ensemble(100, 20, 50, rng, tau_blocks=12.0)
    r = cp.panel(X, P, blocks, make_tol(), b0=0)
    assert 'diagnostics' in r and 'pc1' in r['diagnostics']
    assert r['verdict'] == 'CONVERGED'


def test_prefix_curve_records_each_prefix():
    rng = np.random.default_rng(20)
    X, P, blocks = synth_ensemble(64, 15, 40, rng)
    c = cp.prefix_curve(X, P, blocks, make_tol(), [32, 48, 64], b0_fraction=0.25)
    assert [r['N'] for r in c] == [32, 48, 64]
    assert all(r['verdict'] in ('CONVERGED', 'NOT CONVERGED', 'INSUFFICIENT') for r in c)


def test_format_panel_mentions_every_gate():
    rng = np.random.default_rng(21)
    X, P, blocks = synth_ensemble(60, 15, 40, rng)
    txt = cp.format_panel(cp.panel(X, P, blocks, make_tol(), b0=0))
    for word in ('mean-structure', 'state stability', 'ensemble width', 'basin', 'verdict'):
        assert word in txt


# ---------------------------------------------------------------------------
# source-pool accounting
# ---------------------------------------------------------------------------

def test_pool_overlap_and_disjoint_predecessor():
    b = {10: (100, 150, 300), 11: (105, 160, 300), 12: (140, 170, 300), 20: (155, 200, 300)}
    assert cp.pool_overlap(b, 10, 11) == pytest.approx(46 / 56)
    assert cp.pool_overlap(b, 10, 20) == 0.0
    assert cp.disjoint_predecessor(b, 20) == 10
    assert cp.disjoint_predecessor(b, 11) is None


def test_pool_bounds_reads_resample_indices(tmp_path):
    run = tmp_path / 'run'
    for b, lo, hi in ((0, 0, 8), (1, 5, 17)):
        d = run / 'saved_variables' / str(b)
        d.mkdir(parents=True)
        ind = np.array([[s, i] for i in range(lo, hi + 1) for s in range(3)])
        np.save(d / 'resampled_indcs.npy', ind)
    bounds = cp.pool_bounds(str(run))
    assert bounds[0][:2] == (0, 8) and bounds[1][:2] == (5, 17)
    assert bounds[0][2] == 27
