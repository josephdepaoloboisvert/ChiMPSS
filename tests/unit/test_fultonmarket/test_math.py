"""
Unit tests for pure-math functions in chimpss.fultonmarket.utils.

Gated with openmm because utils.py has 'from openmm import *' at module level.
These tests run on Expanse (full conda env) but are skipped on CI.

Functions tested: geometric_distribution, frobenius_norm, jsd_distance_matrices, rmsd.
"""
import numpy as np
import pytest

pytest.importorskip("openmm", reason="openmm not installed — skipping fultonmarket math tests")

from chimpss.fultonmarket.utils import (
    frobenius_norm,
    geometric_distribution,
    jsd_distance_matrices,
    rmsd,
)

# ── geometric_distribution ────────────────────────────────────────────────────

def test_geometric_distribution_length():
    result = geometric_distribution(300, 400, 5)
    assert len(result) == 5


def test_geometric_distribution_endpoints():
    result = geometric_distribution(300.0, 400.0, 5)
    assert abs(result[0] - 300.0) < 1e-9
    assert abs(result[-1] - 400.0) < 1e-9


def test_geometric_distribution_monotone():
    result = geometric_distribution(310, 370, 8)
    for i in range(len(result) - 1):
        assert result[i] < result[i + 1]


def test_geometric_distribution_single_element():
    result = geometric_distribution(310, 370, 2)
    assert len(result) == 2
    assert abs(result[0] - 310.0) < 1e-9
    assert abs(result[-1] - 370.0) < 1e-9


def test_geometric_distribution_equal_endpoints():
    result = geometric_distribution(300, 300, 4)
    for v in result:
        assert abs(v - 300) < 1e-9


# ── frobenius_norm ────────────────────────────────────────────────────────────

def _sym_matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    return (m + m.T) / 2


def test_frobenius_norm_identical():
    m = _sym_matrix(6)
    assert frobenius_norm(m, m) == pytest.approx(0.0, abs=1e-10)


def test_frobenius_norm_shape_mismatch():
    a = np.zeros((4, 4))
    b = np.zeros((5, 5))
    with pytest.raises(ValueError):
        frobenius_norm(a, b)


def test_frobenius_norm_positive():
    a = _sym_matrix(6, seed=1)
    b = _sym_matrix(6, seed=2)
    assert frobenius_norm(a, b) > 0


def test_frobenius_norm_normalised_vs_raw():
    a = _sym_matrix(6, seed=3)
    b = _sym_matrix(6, seed=4)
    normed = frobenius_norm(a, b, normalise=True)
    raw = frobenius_norm(a, b, normalise=False)
    n = 6
    n_pairs = n * (n - 1) / 2
    raw_val = float(np.sqrt(np.sum((a - b) ** 2)))
    assert normed == pytest.approx(raw_val / n_pairs, rel=1e-6)
    assert raw == pytest.approx(raw_val, rel=1e-6)


def test_frobenius_norm_symmetry():
    a = _sym_matrix(6, seed=5)
    b = _sym_matrix(6, seed=6)
    assert frobenius_norm(a, b) == pytest.approx(frobenius_norm(b, a), rel=1e-9)


# ── jsd_distance_matrices ─────────────────────────────────────────────────────

def test_jsd_identical_matrices():
    m = _sym_matrix(10)
    val = jsd_distance_matrices(m, m)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_jsd_different_matrices_positive():
    a = _sym_matrix(10, seed=7)
    b = _sym_matrix(10, seed=8)
    val = jsd_distance_matrices(a, b)
    assert val > 0


def test_jsd_bounded():
    a = _sym_matrix(10, seed=9)
    b = _sym_matrix(10, seed=10)
    val = jsd_distance_matrices(a, b)
    assert 0.0 <= val <= 1.0


def test_jsd_symmetry():
    a = _sym_matrix(10, seed=11)
    b = _sym_matrix(10, seed=12)
    assert jsd_distance_matrices(a, b) == pytest.approx(jsd_distance_matrices(b, a), rel=1e-6)


# ── rmsd ─────────────────────────────────────────────────────────────────────

def test_rmsd_identical_1d():
    a = np.array([1.0, 2.0, 3.0])
    assert rmsd(a, a) == pytest.approx(0.0, abs=1e-10)


def test_rmsd_known_1d():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    result = rmsd(a, b)
    expected = np.sqrt((1.0 ** 2) / 3)
    assert result == pytest.approx(expected, rel=1e-6)


def test_rmsd_2d_shape():
    a = np.random.default_rng(0).random((4, 3))
    b = np.random.default_rng(1).random((4, 3))
    result = rmsd(a, b)
    assert hasattr(result, "__len__")
    assert len(result) == 4


def test_rmsd_2d_identical():
    a = np.random.default_rng(0).random((3, 5))
    result = rmsd(a, a)
    for v in result:
        assert v == pytest.approx(0.0, abs=1e-10)
