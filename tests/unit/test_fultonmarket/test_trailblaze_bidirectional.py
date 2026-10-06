"""Unit tests for downward and bidirectional trailblazing with the fake rung sampler.

A down walk starts at T_max, steps down and finishes with the same ascending
record, ``temperatures.npy`` and ``seeds/rung_XXX.npz`` layout as an up walk,
so FultonMarket reads either.  Reconciliation of an up and a down walk places
the ladder by the direction-free integral rule and seeds it from the down walk.
"""
import json
import os

import numpy as np
import pytest

from chimpss.fultonmarket import trailblaze as tb

from .test_trailblaze_walk import CV, FakeSampler, make_seed


class PreparingSampler(FakeSampler):
    """A fake sampler with a ``prepare`` that records how it was asked to heat the seed."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prepare_calls = []
        self.last_prepare = {}

    def prepare(self, T, seed, T_from=None):
        self.prepare_calls.append((T, T_from))
        stages = [] if T_from is None else tb.ramp_stages(T_from, T, 10.0)
        self.last_prepare = {'stages': stages, 'ramp_K': 10.0}
        return make_seed(tag=T)


def quiet(*_):
    pass


# ---------------------------------------------------------------------------
# ramp stages and legacy inputs
# ---------------------------------------------------------------------------

def test_ramp_stages():
    assert tb.ramp_stages(310.0, 367.0, 10.0) == [310.0, 320.0, 330.0, 340.0, 350.0, 360.0]
    assert tb.ramp_stages(367.0, 310.0, 10.0) == [367.0, 357.0, 347.0, 337.0, 327.0, 317.0]
    assert tb.ramp_stages(300.0, 310.0, 10.0) == []           # one jump, as the up walk has always done
    assert tb.ramp_stages(310.0, 310.0, 10.0) == []
    with pytest.raises(ValueError):
        tb.ramp_stages(310.0, 367.0, 0.0)


def test_inputs_match_reads_legacy_defaults():
    now = {'a': 1, 'sigma_source': 'pooled', 'direction': 'up'}
    assert tb.inputs_match({'a': 1}, now)                                   # rep4-style record
    assert not tb.inputs_match({'a': 1}, {**now, 'direction': 'down'})
    assert not tb.inputs_match({'a': 1, 'sigma_source': 'slope'}, now)
    assert not tb.inputs_match({'a': 1, 'extra': 2}, now)


# ---------------------------------------------------------------------------
# the down walk
# ---------------------------------------------------------------------------

def test_down_walk_finishes_ascending_with_seeds_to_match(tmp_path):
    sampler = PreparingSampler(seed=21, n=400)
    walker = tb.Trailblazer(sampler, 310.0, 367.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                            printf=quiet)
    rec = walker.run(make_seed())
    T = rec.temperatures
    assert rec.complete and rec.direction == 'down' and rec.meta['direction'] == 'down'
    assert T[0] == 310.0 and T[-1] == 367.0 and np.all(np.diff(T) > 0)
    assert 50 <= T.size <= 70
    # walked from the top: the first sampler call is T_max, the last T_min
    assert sampler.calls[0] == 367.0 and sampler.calls[-1] == 310.0
    assert sampler.prepare_calls == [(367.0, 310.0)]          # heated from the cold end by default
    assert rec.prepare['T'] == 367.0 and rec.prepare['T_from'] == 310.0
    assert rec.prepare['stages'] == [310.0, 320.0, 330.0, 340.0, 350.0, 360.0]
    # walk order is kept on every rung and runs opposite to the index
    assert [r.walk_index for r in rec.rungs] == list(range(T.size - 1, -1, -1))
    assert np.array_equal(np.load(tmp_path / tb.LADDER_NAME), T)
    assert walker.store.has_all(T.size) and walker.walk_store.has_all(T.size)
    for k, r in enumerate(rec.rungs):
        seed, H = walker.store.load(k)
        assert seed.positions[0, 0] == pytest.approx(r.T)       # the seed at rung_k was sampled at T_k
        assert r.seed_path == walker.store.path(k)
    # pair acceptances resolve to ascending pairs: rung k holds the pair (k, k+1) for a down walk
    acc = rec.pair_acc()
    assert acc.shape == (T.size - 1,) and np.all(np.isfinite(acc))
    assert np.all((acc >= 0.3) & (acc <= 0.75)) and abs(np.median(acc) - 0.4) < 0.05
    assert rec.rungs[-1].acc_est is None and rec.rungs[0].acc_est is not None
    assert np.all(np.isfinite(rec.pair_dL()))
    ratios = T[3:] / T[2:-1]                                   # the even-out pairs sit at the bottom
    assert np.std(ratios) / np.mean(ratios) < 0.05
    assert rec.meta['L_total'] == pytest.approx(np.sqrt(CV) * np.log(367.0 / 310.0), rel=0.05)


def test_down_walk_seed_from_t_max_skips_the_ramp(tmp_path):
    sampler = PreparingSampler(seed=22, n=200)
    rec = tb.Trailblazer(sampler, 310.0, 320.0, target_acc=0.4, direction='down', seed_from=320.0,
                         out_dir=str(tmp_path), printf=quiet).run(make_seed())
    assert rec.complete
    assert sampler.prepare_calls == [(320.0, None)]
    assert rec.prepare['T_from'] == 320.0 and rec.prepare['stages'] == []


def test_down_walk_remaps_rejections(tmp_path):
    # the second call (rung 1 of the walk, just below T_max) returns a sigma 3x too small
    sampler = FakeSampler(seed=23, sigma_scale={1: 1 / 3})
    rec = tb.Trailblazer(sampler, 364.0, 367.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                         printf=quiet).run(make_seed())
    assert rec.complete and rec.rejected
    K = len(rec.rungs)
    for rej in rec.rejected:
        assert rej['after_rung'] == K - 1 - rej['after_walk_index']
    assert rec.rejected[0]['after_walk_index'] == 0            # rejected right after the hottest rung
    assert rec.rejected[0]['after_rung'] == K - 1
    assert rec.rungs[-2].rejects >= 1                           # the rung seeded from the top carries them


def test_down_walk_resumes_from_a_half_written_record(tmp_path):
    killed = FakeSampler(seed=24, fail_after=5)
    walker = tb.Trailblazer(killed, 310.0, 367.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                            printf=quiet)
    with pytest.raises(RuntimeError, match='wall-clock'):
        walker.run(make_seed())
    partial = tb.TrailblazeRecord.load(str(tmp_path / tb.RECORD_NAME))
    assert not partial.complete and len(partial.rungs) == 5
    assert np.all(np.diff(partial.temperatures) < 0)           # still in walk order
    assert not os.path.exists(tmp_path / tb.LADDER_NAME)
    assert walker.walk_store.has_all(5) and not os.path.exists(walker.store.path(0))

    messages = []
    resumed = FakeSampler(seed=24)
    resumed.rng = np.random.default_rng(7)
    rec = tb.Trailblazer(resumed, 310.0, 367.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                         printf=messages.append).run(make_seed())
    assert any('resuming after rung 4' in m for m in messages)
    assert rec.complete and np.all(np.diff(rec.temperatures) > 0)
    assert resumed.calls[0] < partial.rungs[-1].T               # continued below the last walked rung
    assert rec.temperatures[-5:].tolist() == sorted(partial.temperatures.tolist())
    assert tb.SeedStore(str(tmp_path)).has_all(len(rec.rungs))

    untouched = FakeSampler(seed=25)
    again = tb.Trailblazer(untouched, 310.0, 367.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                           printf=quiet).run(make_seed())
    assert again.complete and untouched.calls == []


def test_down_walk_refinalises_after_a_kill_between_link_and_save(tmp_path):
    sampler = FakeSampler(seed=26, n=200)
    rec = tb.Trailblazer(sampler, 310.0, 316.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                         printf=quiet).run(make_seed())
    K = len(rec.rungs)
    # undo the finalisation on disk: record in walk order, incomplete, rung_* links gone
    d = rec.to_dict()
    d['complete'] = False
    d['rungs'] = sorted(d['rungs'], key=lambda r: r['walk_index'])
    for r in d['rungs']:
        r['seed_path'] = None
    with open(tmp_path / tb.RECORD_NAME, 'w') as fh:
        json.dump(d, fh)
    store = tb.SeedStore(str(tmp_path))
    for k in range(K):
        os.remove(store.path(k))
    os.remove(tmp_path / tb.LADDER_NAME)
    untouched = FakeSampler(seed=27, n=200)
    again = tb.Trailblazer(untouched, 310.0, 316.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                           printf=quiet).run(make_seed())
    assert untouched.calls == []                               # nothing resampled
    assert again.complete and again.temperatures.tolist() == rec.temperatures.tolist()
    assert store.has_all(K) and os.path.exists(tmp_path / tb.LADDER_NAME)


def test_direction_is_part_of_the_inputs_and_legacy_records_still_resume(tmp_path):
    sampler = FakeSampler(seed=28)
    tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.4, out_dir=str(tmp_path), printf=quiet).run(make_seed())
    with pytest.raises(RuntimeError, match='different inputs'):
        tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                       printf=quiet).run(make_seed())
    rec = tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.4, direction='down', out_dir=str(tmp_path),
                         printf=quiet, force=True).run(make_seed())
    assert rec.complete and rec.direction == 'up'              # the cached up ladder is returned

    # a record written before sigma_source/direction existed: an incomplete up walk, version 1
    legacy_dir = tmp_path / 'legacy'
    killed = FakeSampler(seed=29, fail_after=3)
    with pytest.raises(RuntimeError, match='wall-clock'):
        tb.Trailblazer(killed, 310.0, 367.0, target_acc=0.4, out_dir=str(legacy_dir), printf=quiet).run(make_seed())
    path = legacy_dir / tb.RECORD_NAME
    with open(path) as fh:
        d = json.load(fh)
    d['version'] = 1
    d['inputs'].pop('sigma_source')
    d['inputs'].pop('direction')
    d.pop('prepare', None)
    for r in d['rungs']:
        for k in ('walk_index', 'source', 'source_index', 'source_T', 'sigma_fit', 'cv_slope'):
            r.pop(k, None)
    with open(path, 'w') as fh:
        json.dump(d, fh)
    rec = tb.Trailblazer(FakeSampler(seed=29), 310.0, 367.0, target_acc=0.4, out_dir=str(legacy_dir),
                         printf=quiet).run(make_seed())
    assert rec.complete and rec.direction == 'up' and rec.temperatures[:3].size == 3
    with pytest.raises(ValueError, match='both'):
        tb.Trailblazer(FakeSampler(), 310.0, 367.0, direction='both')
    with pytest.raises(ValueError):
        tb.Trailblazer(FakeSampler(), 310.0, 367.0, direction='sideways')


def test_pair_acc_resolves_ascending_pairs_for_both_directions(tmp_path):
    up = tb.Trailblazer(FakeSampler(seed=30, n=300), 310.0, 315.0, target_acc=0.4, printf=quiet).run(make_seed())
    down = tb.Trailblazer(FakeSampler(seed=31, n=300), 310.0, 315.0, target_acc=0.4, direction='down',
                          printf=quiet).run(make_seed())
    assert up.pair_acc().tolist() == [r.acc_est for r in up.rungs[1:]]
    assert down.pair_acc().tolist() == [r.acc_est for r in down.rungs[:-1]]
    assert up.pair_dL().size == len(up.rungs) - 1 and down.pair_dL().size == len(down.rungs) - 1
    assert tb.TrailblazeRecord(inputs={}).pair_acc().size == 0


def test_seed_store_link_and_velocity_scaling(tmp_path):
    src = tb.SeedStore(str(tmp_path), prefix=tb.WALK_PREFIX)
    src.save(3, make_seed(tag=1.0), np.arange(4.0))
    dst = tb.SeedStore(str(tmp_path))
    dst.link(0, src.path(3))
    seed, H = dst.load(0)
    assert np.allclose(seed.positions, 1.0) and np.array_equal(H, np.arange(4.0))
    assert os.path.samefile(dst.path(0), src.path(3)) or np.allclose(seed.velocities, 0.0)
    src.save(4, tb.Seed(positions=np.ones((7, 3)), box_vectors=np.eye(3), velocities=np.full((7, 3), 2.0)), np.ones(2))
    dst.link(1, src.path(4), velocity_scale=0.5)
    assert np.allclose(dst.load(1)[0].velocities, 1.0)
    assert not os.path.samefile(dst.path(1), src.path(4))
    dst.link(1, src.path(4), velocity_scale=0.5)                # re-runnable
    assert not any('.tmp.' in f for f in os.listdir(dst.dir))
