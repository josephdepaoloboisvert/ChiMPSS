"""
Regression tests for MBAR importance-resampling probability handling.

Three production runs (53855971, 53855972, 53856114) were killed after 17 h to
1 d 22 h of sampling by ``ValueError: probabilities do not sum to 1`` raised
from ``np.random.choice`` inside ``resample_with_MBAR``. The cause was the
order of two lines: the raw weight column was divided by
``np.nan_to_num(probs).sum()`` while it still contained NaN, and NaN was only
replaced with zero afterwards. When MBAR fails to converge and returns an
all-NaN or all-zero column the denominator is zero, so the division produces
inf/NaN and the vector sums to zero.
"""
import numpy as np
import pytest

from chimpss.fultonmarket.utils import (
    MBARWeightsDegenerate,
    _sanitize_probabilities,
)

TOL = np.sqrt(np.finfo(float).eps)  # np.random.choice's own tolerance


@pytest.mark.parametrize("raw", [
    pytest.param(np.full(64, np.nan),                                  id="all-nan"),
    pytest.param(np.zeros(64),                                         id="all-zero"),
    pytest.param(np.full(64, -1.0),                                    id="all-negative"),
    pytest.param(np.array([np.nan, np.inf, -np.inf, 0.0]),             id="no-finite-mass"),
])
def test_degenerate_columns_return_none(raw):
    """A column with no usable mass is reported, not silently renormalized."""
    assert _sanitize_probabilities(raw) is None


@pytest.mark.parametrize("raw", [
    pytest.param(np.where(np.arange(64) % 3 == 0, np.nan, 1.0),        id="mixed-nan"),
    pytest.param(np.concatenate([[np.inf], np.ones(63)]),              id="contains-posinf"),
    pytest.param(np.linspace(-1.0, 1.0, 64),                           id="contains-negative"),
    pytest.param(np.full(64, 1.0 / 64),                                id="uniform"),
    pytest.param(np.exp(-np.arange(64) / 3.0),                         id="peaked"),
    pytest.param(np.eye(64)[0],                                        id="single-nonzero"),
])
def test_usable_columns_are_valid_probability_vectors(raw):
    """Anything with usable mass comes back as something choice() accepts."""
    probs = _sanitize_probabilities(raw)
    assert probs is not None
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert abs(probs.sum() - 1.0) <= TOL
    np.random.choice(len(probs), size=8, replace=True, p=probs)  # must not raise


def test_does_not_mutate_caller_weights():
    """
    ``probs = weights[:, k]`` is a numpy view. The old in-place ``probs /= ...``
    overwrote that column of the caller's weight matrix, so the weights returned
    via ``return_weights`` -- and ``resampled_weights``, read back out of the
    same array afterwards -- were the normalized values rather than the raw
    MBAR weights.
    """
    weights = np.tile(np.linspace(1.0, 2.0, 8), (3, 1)).T.copy()
    before = weights.copy()
    _sanitize_probabilities(weights[:, 0])
    assert np.array_equal(weights, before)


def test_the_exact_old_failure_is_now_handled():
    """The all-NaN column that killed three jobs."""
    raw = np.full(1024, np.nan)

    # what the old code did, reproduced
    old = raw / np.nan_to_num(raw).sum()
    old = np.nan_to_num(old)
    with pytest.raises(ValueError, match="probabilities do not sum to 1"):
        np.random.choice(len(old), size=4, replace=True, p=old)

    # what the new code does
    assert _sanitize_probabilities(raw) is None


def test_degenerate_error_is_catchable_and_specific():
    assert issubclass(MBARWeightsDegenerate, RuntimeError)


# ---------------------------------------------------------------------------
# Two-sided ESS gate
#
# _sanitize_probabilities rejects a weight column only when it carries no
# probability mass at all. A column that has collapsed onto one frame, and a
# column that is perfectly uniform, both sanitize cleanly -- so neither was
# caught. Over 75 evaluated blocks of NTSR1_ML-301.hmr.rep0 the collapsed case
# occurred 5 times (6.7%), each producing distance matrices built from 1-10
# distinct configurations.
# ---------------------------------------------------------------------------

def _collapsed_column(n=2000, dominant=0.9962):
    """One frame carrying essentially all the mass -- block 3 / 29 in miniature."""
    w = np.full(n, (1.0 - dominant) / (n - 1))
    w[0] = dominant
    return w


def _uniform_column(n=2000):
    return np.full(n, 1.0 / n)


def _healthy_column(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 0.6, n)
    return w / w.sum()


def _run(weights, **kw):
    """Drive resample_with_MBAR with a prescribed weight column."""
    from unittest import mock
    from chimpss.fultonmarket import utils as U
    with mock.patch.object(U, 'compute_MBAR_weights', return_value=weights):
        return U.resample_with_MBAR(objs=[np.arange(weights.size)],
                                    u_kln=np.zeros((1, weights.size)),
                                    N_k=np.array([weights.size]),
                                    size=1000, **kw)


def test_collapsed_column_rejected_when_gate_enabled():
    from chimpss.fultonmarket.utils import MBARWeightsDegenerate
    with pytest.raises(MBARWeightsDegenerate, match='collapsed'):
        _run(_collapsed_column(), min_ess=25)


def test_collapsed_column_still_passes_when_gate_disabled():
    """min_ess=None must preserve the previous behaviour exactly."""
    out = _run(_collapsed_column(), min_ess=None)
    assert out is not None


def test_healthy_column_survives_the_gate():
    out = _run(_healthy_column(), min_ess=25)
    assert out is not None


def test_uniform_column_warns_but_is_not_rejected():
    """A uniform draw is unweighted, not garbage -- report it, keep it."""
    msgs = []
    out = _run(_uniform_column(), min_ess=25,
               _printf=lambda m, level='all': msgs.append(str(m)))
    assert out is not None
    assert any('uniform' in m for m in msgs), msgs


def test_gate_threshold_separates_real_collapsed_from_real_healthy():
    """ESS values measured in production: collapsed 1.0-10, healthy 50-150."""
    from chimpss.fultonmarket.utils import MBARWeightsDegenerate
    collapsed_ess = 1.0 / np.sum(_collapsed_column() ** 2)
    healthy_ess = 1.0 / np.sum(_healthy_column() ** 2)
    assert collapsed_ess < 25 < healthy_ess
    with pytest.raises(MBARWeightsDegenerate):
        _run(_collapsed_column(), min_ess=25)
    assert _run(_healthy_column(), min_ess=25) is not None


# ---------------------------------------------------------------------------
# compute_MBAR_weights: solver-protocol retry
# ---------------------------------------------------------------------------
from unittest import mock

from chimpss.fultonmarket import utils as U

class _FakeMBAR:
    """Stands in for pymbar.MBAR.  In failure mode it does what pymbar does:
    logs the complaint to the 'pymbar' logger and returns a weight matrix
    whose columns do not sum to one."""
    calls = []

    def __init__(self, u_kln, N_k, initialize='BAR', solver_protocol=None):
        import logging
        _FakeMBAR.calls.append(solver_protocol)
        self._fail = ((solver_protocol is not None and getattr(_FakeMBAR, 'fail_adaptive', False))
                      or (solver_protocol is None and getattr(_FakeMBAR, 'fail_default', False)))
        if self._fail:
            logging.getLogger('pymbar').warning('WARNING: Did not converge to within specified tolerance.')
        self._n, self._k = u_kln.shape[1], u_kln.shape[0]

    def weights(self):
        W = np.full((self._n, self._k), 1.0 / self._n)
        return W * 1e-3 if self._fail else W


def _reset_fake(fail_default=False, fail_adaptive=False):
    _FakeMBAR.calls = []
    _FakeMBAR.fail_default = fail_default
    _FakeMBAR.fail_adaptive = fail_adaptive


def test_mbar_weights_sanity_check():
    good = np.full((12, 3), 1.0 / 12)
    assert U.mbar_weights_are_sane(good)
    assert not U.mbar_weights_are_sane(good * 1e-3)          # columns do not sum to one
    assert not U.mbar_weights_are_sane(np.where(np.eye(12, 3) > 0, np.nan, good))
    bad = good.copy(); bad[0, 0] = -0.5; bad[1, 0] = 0.5 + 1 / 12
    assert not U.mbar_weights_are_sane(bad)                 # negative weight


def test_compute_mbar_weights_uses_adaptive_protocol_first():
    _reset_fake()
    with mock.patch.object(U, 'MBAR', _FakeMBAR):
        W = U.compute_MBAR_weights(np.zeros((3, 12)), np.array([4, 4, 4]))
    assert W.shape == (12, 3)
    assert _FakeMBAR.calls == [U.ADAPTIVE_PROTOCOL]


def test_compute_mbar_weights_falls_back_to_default_protocol():
    """pymbar reports failure through logging and returns an unnormalised
    matrix; both must be caught, and the default protocol tried next."""
    _reset_fake(fail_adaptive=True)
    msgs = []
    with mock.patch.object(U, 'MBAR', _FakeMBAR):
        W = U.compute_MBAR_weights(np.zeros((3, 12)), np.array([4, 4, 4]), _printf=msgs.append)
    assert W.shape == (12, 3) and np.allclose(W.sum(axis=0), 1.0)
    assert _FakeMBAR.calls == [U.ADAPTIVE_PROTOCOL, None]
    assert any('retrying' in m for m in msgs)


def test_compute_mbar_weights_detects_failure_from_weights_alone():
    """Even with no message at all, a matrix whose columns do not sum to one
    is a failed solve."""
    class _Silent(_FakeMBAR):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
        def weights(self):
            return np.full((self._n, self._k), 1.0 / self._n) * (1e-3 if self._fail else 1.0)
    _reset_fake(fail_adaptive=True, fail_default=True)
    import logging
    with mock.patch.object(U, 'MBAR', _Silent), mock.patch.object(logging.getLogger('pymbar'), 'warning', lambda *a, **k: None):
        with pytest.raises(U.MBARWeightsDegenerate):
            U.compute_MBAR_weights(np.zeros((3, 12)), np.array([4, 4, 4]))


def test_compute_mbar_weights_raises_when_both_protocols_fail():
    _reset_fake(fail_default=True, fail_adaptive=True)
    with mock.patch.object(U, 'MBAR', _FakeMBAR):
        with pytest.raises(U.MBARWeightsDegenerate):
            U.compute_MBAR_weights(np.zeros((3, 12)), np.array([4, 4, 4]))
    assert len(_FakeMBAR.calls) == 2
