"""Tests for per-component PCA weighting.

get_traj_PCA returns the *cumulative* explained-variance curve, because that is
what the explained_variance_threshold cut consumes. determine_equilibration used
it directly as per-component weights, which inverts the intended weighting: the
trailing components, which carry almost no variance and report t0 = 0, ended up
with the largest weights.

Measured on NTSR1_ML-301.hmr.rep0 (40 components, 729 frames, stride 10): PC1
carried 32 % of the variance and reported t0 = 53, while each ~0.5 % tail
component received 3.1x PC1's weight. The consensus came out at t0 = 50 frames
(6.9 %) instead of 320 (43.9 %).
"""
import numpy as np
import pytest

from chimpss.fultonmarket.utils import per_component_variance


# ---------------------------------------------------------------------------
# per_component_variance
# ---------------------------------------------------------------------------

def test_differences_a_cumulative_curve():
    cum = np.array([0.5, 0.75, 0.9, 1.0])
    assert per_component_variance(cum) == pytest.approx([0.5, 0.25, 0.15, 0.10])


def test_weights_are_normalised():
    cum = np.array([0.2892, 0.4251, 0.4811, 0.5295, 0.565, 0.5914])
    assert per_component_variance(cum).sum() == pytest.approx(1.0)


def test_first_component_keeps_the_largest_weight():
    """PCA orders components by variance, so PC1 must always weigh most."""
    cum = np.cumsum([0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
    w = per_component_variance(cum)
    assert w[0] == max(w)
    assert np.all(np.diff(w) <= 1e-12)          # monotonically non-increasing


def test_cumulative_used_directly_inverts_the_ordering():
    """Pin the defect: the naive form makes the last component outweigh the first."""
    cum = np.cumsum([0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
    naive = cum / cum.sum()
    assert naive[-1] > naive[0]                 # backwards
    fixed = per_component_variance(cum)
    assert fixed[-1] < fixed[0]                 # correct


def test_recovers_the_production_consensus():
    """The real 40-component case: 6.9 % as coded, 43.9 % corrected."""
    cum = np.array([0.2892, 0.4251, 0.4811, 0.5295, 0.5650, 0.5914, 0.6148,
                    0.6359, 0.6549] + list(np.linspace(0.6720, 0.8996, 31)))
    t0 = np.array([53, 31, 61, 58, 36, 62, 12] + [0] * 33)
    naive = int(np.sum(t0 * (cum / cum.sum()))) * 10
    fixed = int(np.sum(t0 * per_component_variance(cum))) * 10
    assert naive < 100                          # implausibly early
    assert 250 < fixed < 400                    # in family with the energy detectors
    assert fixed > 3 * naive


def test_handles_a_flat_tail_without_negative_weights():
    """A cumulative curve that plateaus must not yield negative weights."""
    cum = np.array([0.6, 0.85, 0.9, 0.9, 0.9, 0.9])
    w = per_component_variance(cum)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0)


def test_tolerates_float_noise_in_a_plateau():
    cum = np.array([0.6, 0.9, 0.9 - 1e-15, 0.9])
    assert np.all(per_component_variance(cum) >= 0.0)


def test_degenerate_inputs():
    assert per_component_variance(np.array([])).size == 0
    flat = per_component_variance(np.zeros(4))
    assert flat == pytest.approx([0.25, 0.25, 0.25, 0.25])   # uniform fallback


def test_single_component_takes_all_the_weight():
    assert per_component_variance(np.array([0.87])) == pytest.approx([1.0])


def test_weighting_is_scale_invariant():
    """Only the shape of the curve matters, not its absolute level."""
    cum = np.cumsum([0.4, 0.3, 0.2, 0.1])
    assert per_component_variance(cum) == pytest.approx(per_component_variance(cum * 0.5))
