"""Geometry tests for chimpss.analysis.motifs.

These cover the pure-math helpers (dihedral, Kabsch superposition) and the
NaN contract of the five motif measurements.  The BW-resolution path needs
GPCRdb network access and is exercised by the stage-06 job scripts instead.
"""
import numpy as np
import pytest

from chimpss.analysis.motifs import (
    MOTIF_COLUMNS,
    _dihedral,
    _kabsch,
    dry_ionic_lock,
    summarize,
    tm7_h8_distance,
    tm36_distance,
    w648_chi1,
)

# ---------------------------------------------------------------------------
# Dihedral
# ---------------------------------------------------------------------------

def test_dihedral_trans_is_180():
    p0 = np.array([1.0, 1.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([-1.0, 0.0, 0.0])
    assert abs(_dihedral(p0, p1, p2, p3)) == pytest.approx(180.0, abs=1e-6)


def test_dihedral_cis_is_zero():
    p0 = np.array([1.0, 1.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    assert _dihedral(p0, p1, p2, p3) == pytest.approx(0.0, abs=1e-6)


def test_dihedral_is_signed_and_antisymmetric():
    p0 = np.array([1.0, 1.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3a = np.array([0.0, -1.0, 1.0])
    p3b = np.array([0.0, -1.0, -1.0])
    a = _dihedral(p0, p1, p2, p3a)
    b = _dihedral(p0, p1, p2, p3b)
    assert a == pytest.approx(-b, abs=1e-6)
    assert a != pytest.approx(0.0, abs=1e-6)


def test_dihedral_range_is_minus180_to_180():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pts = rng.normal(size=(4, 3))
        v = _dihedral(*pts)
        assert -180.0 <= v <= 180.0


# ---------------------------------------------------------------------------
# Kabsch
# ---------------------------------------------------------------------------

def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_kabsch_recovers_a_rigid_transform_exactly():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(12, 3)) * 5.0
    rot_true = _random_rotation(rng)
    trans = rng.normal(size=3) * 10.0
    mobile = ref @ rot_true.T + trans

    rot, mob_com, ref_com = _kabsch(mobile, ref)
    moved = (mobile - mob_com) @ rot.T + ref_com

    assert np.allclose(moved, ref, atol=1e-9)


def test_kabsch_never_reflects():
    """A mirrored copy must not superpose to zero RMSD."""
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(10, 3)) * 5.0
    mirrored = ref.copy()
    mirrored[:, 2] *= -1.0

    rot, mob_com, ref_com = _kabsch(mirrored, ref)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)

    moved = (mirrored - mob_com) @ rot.T + ref_com
    rmsd = np.sqrt(((moved - ref) ** 2).sum() / len(ref))
    assert rmsd > 1e-3


# ---------------------------------------------------------------------------
# NaN contract -- unresolved labels must degrade, not raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [tm36_distance, tm7_h8_distance, w648_chi1,
                                dry_ionic_lock])
def test_missing_labels_return_nan(fn):
    assert np.isnan(fn(None, {}))


def test_partially_resolved_labels_return_nan():
    assert np.isnan(tm36_distance(None, {'3.50': 140}))
    assert np.isnan(tm7_h8_distance(None, {'8.50': 320}))
    assert np.isnan(dry_ionic_lock(None, {'3.50': 140}))


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_is_nan_safe():
    arr = np.full((5, len(MOTIF_COLUMNS)), np.nan)
    arr[:, 0] = [1.0, 2.0, 3.0, np.nan, 5.0]

    out = summarize(arr)
    assert out['tm36_distance']['n'] == 4
    assert out['tm36_distance']['mean'] == pytest.approx(2.75)
    assert out['dry_ionic_lock']['n'] == 0
    assert out['dry_ionic_lock']['mean'] is None


def test_summarize_handles_empty_input():
    out = summarize(np.empty((0, len(MOTIF_COLUMNS))))
    for col in MOTIF_COLUMNS:
        assert out[col]['n'] == 0
        assert out[col]['mean'] is None
