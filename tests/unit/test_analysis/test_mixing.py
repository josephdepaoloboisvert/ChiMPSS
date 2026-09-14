"""Unit tests for chimpss.analysis.mixing.

Validated against chains with analytically known mixing times and against a
hand-built ladder traversal, because the point of the module is to produce
numbers nobody has measured before -- "it runs" is not evidence.
"""
import numpy as np
import pytest

from chimpss.analysis import mixing as mx


def two_state_chain(p, n_it=4000, n_rep=20, seed=0):
    """Symmetric two-state chain with switch probability p -> lambda_2 = 1-2p."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n_it, n_rep), dtype=int)
    for it in range(1, n_it):
        flip = rng.random(n_rep) < p
        s[it] = np.where(flip, 1 - s[it - 1], s[it - 1])
    return s


# ---------------------------------------------------------------------------
# transition_matrix
# ---------------------------------------------------------------------------

def test_transition_matrix_is_row_stochastic():
    T = mx.transition_matrix(two_state_chain(0.1))
    assert np.allclose(T.sum(axis=1), 1.0)


def test_transition_matrix_recovers_known_probability():
    p = 0.1
    T = mx.transition_matrix(two_state_chain(p, n_it=20000))
    assert T[0, 1] == pytest.approx(p, abs=0.01)
    assert T[1, 0] == pytest.approx(p, abs=0.01)


def test_transition_matrix_handles_unvisited_state():
    """An unvisited row must not divide by zero; it becomes absorbing."""
    s = np.zeros((50, 4), dtype=int)
    T = mx.transition_matrix(s, n_states=5)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert T[3, 3] == 1.0


def test_transition_matrix_rejects_degenerate_input():
    with pytest.raises(ValueError):
        mx.transition_matrix(np.zeros((1, 4), dtype=int))


# ---------------------------------------------------------------------------
# mixing_time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('p', [0.05, 0.1, 0.25])
def test_mixing_time_matches_analytic_tau2(p):
    expected = -1.0 / np.log(1 - 2 * p)
    got = mx.mixing_time(mx.transition_matrix(two_state_chain(p, n_it=20000)))
    assert got == pytest.approx(expected, rel=0.15)


def test_mixing_time_near_zero_for_iid_states():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 10, size=(2000, 20))
    assert mx.mixing_time(mx.transition_matrix(s)) < 1.0


def test_mixing_time_infinite_when_frozen():
    s = np.tile(np.arange(10), (500, 1))
    assert mx.mixing_time(mx.transition_matrix(s)) == float('inf')


# ---------------------------------------------------------------------------
# round_trips
# ---------------------------------------------------------------------------

def _sawtooth(n_sweeps, K=10):
    leg = list(range(K)) + list(range(K - 2, -1, -1))
    return np.array((leg * n_sweeps) + [0])[:, None]


@pytest.mark.parametrize('n', [1, 3, 5])
def test_round_trips_counts_full_sweeps(n):
    rt = mx.round_trips(_sawtooth(n), n_states=10)
    assert rt['total_round_trips'] == n
    assert rt['half_legs'] == 2 * n


def test_round_trips_zero_when_walker_never_reaches_top():
    s = np.tile(np.arange(5), (20, 1))[:, :1]      # never exceeds state 4 of 10
    rt = mx.round_trips(s, n_states=10)
    assert rt['total_round_trips'] == 0
    assert rt['mean_iterations_per_leg'] == float('inf')


def test_round_trips_scales_with_walkers():
    single = _sawtooth(2)
    many = np.repeat(single, 4, axis=1)
    assert (mx.round_trips(many, n_states=10)['total_round_trips']
            == 4 * mx.round_trips(single, n_states=10)['total_round_trips'])


# ---------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------

def test_state_occupancy_counts_replicas_per_iteration():
    s = np.array([[0, 0, 1], [0, 1, 1]])
    assert list(mx.state_occupancy(s, 0)) == [2, 1]


def test_walkers_reaching_state_counts_distinct_walkers():
    s = np.array([[0, 1, 2], [1, 0, 2], [1, 1, 0]])
    assert mx.walkers_reaching_state(s, 0) == 3
    assert mx.walkers_reaching_state(s, 2) == 1


def test_acceptance_from_states_bounds():
    assert mx.acceptance_from_states(np.tile(np.arange(5), (10, 1))) == 0.0
    alt = np.array([[0, 1], [1, 0]] * 10)
    assert mx.acceptance_from_states(alt) == 1.0
