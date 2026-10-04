"""Unit tests for the Trailblazer walk with a fake rung sampler.

The sampler is a function that draws Gaussian enthalpies for a known heat
capacity, so the walk's output has a closed form (a geometric ladder ending
exactly at T_max with the target acceptance on every pair), and its
bookkeeping -- rejection, stretching, resume from a half-written record,
cached completion, input mismatch -- can be exercised without a GPU.
"""
import os

import numpy as np
import pytest

from chimpss.fultonmarket import trailblaze as tb

CV = 1.7e5
N_ATOMS = 7


def make_seed(tag=0.0):
    return tb.Seed(positions=np.full((N_ATOMS, 3), tag), box_vectors=np.eye(3) * 6.0,
                   velocities=np.zeros((N_ATOMS, 3)))


class FakeSampler:
    """Gaussian enthalpies with constant C_V; records every call; can misbehave on demand."""

    def __init__(self, seed=0, n=120, sigma_scale=None, fail_after=None):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.sigma_scale = sigma_scale or {}
        self.fail_after = fail_after
        self.calls = []

    def describe(self):
        return {'kind': 'fake', 'n': self.n}

    def __call__(self, T, seed):
        self.calls.append(T)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError('simulated wall-clock kill')
        scale = self.sigma_scale.get(len(self.calls), 1.0)
        H = self.rng.normal(-1.0e6 + CV * tb.KB * T, scale * tb.KB * T * np.sqrt(CV), size=self.n)
        return H, make_seed(tag=T), {'g': 1.0, 'neff': float(self.n)}


def test_walk_produces_a_geometric_ladder_to_t_max(tmp_path):
    sampler = FakeSampler(seed=1, n=400)
    walker = tb.Trailblazer(sampler, 310.0, 367.0, target_acc=0.4, out_dir=str(tmp_path), printf=lambda *_: None)
    rec = walker.run(make_seed())
    T = rec.temperatures
    assert rec.complete
    assert T[0] == 310.0 and T[-1] == 367.0 and np.all(np.diff(T) > 0)
    assert 50 <= T.size <= 70
    ratios = T[1:-2] / T[:-3]
    assert np.std(ratios) / np.mean(ratios) < 0.05           # sigma noise only
    accs = np.array([r.acc_est for r in rec.rungs[1:]])
    assert np.all((accs >= 0.3) & (accs <= 0.75))
    assert abs(np.median(accs) - 0.4) < 0.05
    # 400 samples per rung leave ~2.5 % noise on sigma; a stray rejection is possible, a habit is not
    assert sum(r.rejects for r in rec.rungs) <= 2
    assert os.path.exists(tmp_path / tb.LADDER_NAME)
    assert np.array_equal(np.load(tmp_path / tb.LADDER_NAME), T)
    assert walker.store.has_all(T.size)
    assert np.unique(np.round(T, 2)).size == T.size
    meta = rec.meta
    assert meta['n_rungs'] == T.size and meta['min_acc'] == 0.3 and meta['complete']
    assert meta['L_total'] == pytest.approx(np.sqrt(CV) * np.log(367.0 / 310.0), rel=0.05)
    assert sampler.calls[0] == 310.0 and sampler.calls[-1] == 367.0


def test_walk_rejects_and_shrinks_when_acceptance_is_low(tmp_path):
    # the second call returns a sigma 3x too small, so the step overshoots and the pair is rejected
    sampler = FakeSampler(seed=2, sigma_scale={1: 1 / 3})
    walker = tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.4, out_dir=str(tmp_path), printf=lambda *_: None)
    rec = walker.run(make_seed())
    assert rec.complete
    assert rec.rejected and rec.rejected[0]['reason'].startswith('below min_acc')
    assert rec.rungs[1].rejects >= 1
    assert rec.rungs[1].step_scale == pytest.approx(1.25 ** (-rec.rungs[1].rejects))
    assert rec.rungs[1].acc_est >= 0.3


def test_walk_rejects_and_stretches_when_acceptance_is_high(tmp_path):
    # A sigma 3x too large at rung 0 makes the first step 3x too short.  Against a broad
    # cold rung the measured acceptance saturates near 0.7 (half its samples are too hot
    # to ever exchange), so the ceiling is lowered to 0.6 to exercise the stretch path.
    sampler = FakeSampler(seed=3, sigma_scale={1: 3.0})
    walker = tb.Trailblazer(sampler, 310.0, 315.0, target_acc=0.4, max_acc=0.6, out_dir=str(tmp_path),
                            printf=lambda *_: None)
    rec = walker.run(make_seed())
    assert rec.complete
    assert any(r['reason'].startswith('above max_acc') for r in rec.rejected)
    assert rec.rungs[1].step_scale > 1.0
    assert rec.rungs[1].acc_est <= 0.6 or rec.rungs[1].rejects == walker.max_retries


def test_walk_gives_up_after_max_retries(tmp_path):
    # every proposal is hopeless: constant sigma 5x too small -> acceptance ~0 until the step shrinks a lot
    sampler = FakeSampler(seed=4, sigma_scale={i: 0.2 for i in range(1, 100)})
    messages = []
    walker = tb.Trailblazer(sampler, 310.0, 311.0, target_acc=0.4, max_retries=3, out_dir=str(tmp_path),
                            printf=messages.append)
    rec = walker.run(make_seed())
    assert rec.complete
    assert any('WARNING accepting' in m for m in messages)
    assert max(r.rejects for r in rec.rungs) == 3


def test_walk_stops_at_max_rungs(tmp_path):
    sampler = FakeSampler(seed=5)
    walker = tb.Trailblazer(sampler, 310.0, 367.0, target_acc=0.4, max_rungs=5, out_dir=str(tmp_path),
                            printf=lambda *_: None)
    with pytest.raises(RuntimeError, match='max_rungs'):
        walker.run(make_seed())


def test_walk_resumes_from_a_half_written_record(tmp_path):
    killed = FakeSampler(seed=6, fail_after=5)
    walker = tb.Trailblazer(killed, 310.0, 367.0, target_acc=0.4, out_dir=str(tmp_path), printf=lambda *_: None)
    with pytest.raises(RuntimeError, match='wall-clock'):
        walker.run(make_seed())
    partial = tb.TrailblazeRecord.load(str(tmp_path / tb.RECORD_NAME))
    assert not partial.complete and len(partial.rungs) == 5
    assert not os.path.exists(tmp_path / tb.LADDER_NAME)
    assert not any('.tmp.' in f for f in os.listdir(tmp_path))

    messages = []
    resumed_sampler = FakeSampler(seed=6)
    resumed_sampler.rng = np.random.default_rng(99)
    walker2 = tb.Trailblazer(resumed_sampler, 310.0, 367.0, target_acc=0.4, out_dir=str(tmp_path),
                             printf=messages.append)
    rec = walker2.run(make_seed())
    assert any('resuming after rung 4' in m for m in messages)
    assert rec.complete and rec.temperatures[:5].tolist() == partial.temperatures.tolist()
    assert resumed_sampler.calls[0] > partial.rungs[-1].T          # it did not resample rung 0..4
    # the seed handed to the first resumed call is the one saved for rung 4
    seed4, H4 = walker2.store.load(4)
    assert seed4.positions[0, 0] == pytest.approx(partial.rungs[-1].T)
    assert H4.size == 120

    # a complete record is returned without touching the sampler
    untouched = FakeSampler(seed=7)
    rec3 = tb.Trailblazer(untouched, 310.0, 367.0, target_acc=0.4, out_dir=str(tmp_path),
                          printf=lambda *_: None).run(make_seed())
    assert rec3.complete and untouched.calls == [] and rec3.temperatures.tolist() == rec.temperatures.tolist()


def test_walk_refuses_a_record_with_other_inputs(tmp_path):
    sampler = FakeSampler(seed=8)
    tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.4, out_dir=str(tmp_path), printf=lambda *_: None).run(make_seed())
    with pytest.raises(RuntimeError, match='different inputs'):
        tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.5, out_dir=str(tmp_path), printf=lambda *_: None).run(make_seed())
    rec = tb.Trailblazer(sampler, 310.0, 312.0, target_acc=0.5, out_dir=str(tmp_path), printf=lambda *_: None,
                         force=True).run(make_seed())
    assert rec.complete


def test_walk_in_memory_without_out_dir():
    sampler = FakeSampler(seed=9)
    rec = tb.Trailblazer(sampler, 310.0, 313.0, target_acc=0.4, printf=lambda *_: None).run(make_seed())
    assert rec.complete and all(r.seed_path is None for r in rec.rungs)


def test_record_round_trip(tmp_path):
    rec = tb.TrailblazeRecord(inputs={'a': 1}, rungs=[tb.Rung(T=310.0, sigma=1.0, mean_H=2.0, n_samples=3, g=1.0,
                                                               neff=3.0, acc_est=None, acc_pred=None, dL=None,
                                                               step_scale=1.0, rejects=0, seed_path=None, wall_s=0.1)])
    path = str(tmp_path / 'r.json')
    rec.save(path)
    back = tb.TrailblazeRecord.load(path)
    assert back.to_dict() == rec.to_dict()
    assert tb.TrailblazeRecord.load_if_complete(path) is None
    back.complete = True
    back.save(path)
    assert tb.TrailblazeRecord.load_if_complete(path).complete
    assert tb.TrailblazeRecord.load_if_complete(str(tmp_path / 'missing.json')) is None


def test_seed_store_round_trip(tmp_path):
    store = tb.SeedStore(str(tmp_path))
    seed = make_seed(tag=3.5)
    store.save(2, seed, np.arange(5.0))
    assert not store.has_all(3) and os.path.exists(store.path(2))
    back, H = store.load(2)
    assert np.allclose(back.positions, 3.5) and np.array_equal(H, np.arange(5.0))
    assert back.velocities is not None and back.velocities.shape == (N_ATOMS, 3)
    store.save(2, tb.Seed(positions=seed.positions, box_vectors=seed.box_vectors), np.ones(2))
    assert store.load(2)[0].velocities is None


def test_trailblazer_validates_arguments():
    with pytest.raises(ValueError):
        tb.Trailblazer(FakeSampler(), 367.0, 310.0)
    with pytest.raises(ValueError):
        tb.Trailblazer(FakeSampler(), 310.0, 367.0, target_acc=0.4, min_acc=0.5)


def test_pooled_sigma_tames_noisy_rung_estimates(tmp_path):
    """With 40 samples per rung the single-rung sigma scatters ~11 %; pooling five rungs must
    make the realised spacing markedly more uniform than stepping on each rung's own sigma."""
    def walk(window, seed):
        sampler = FakeSampler(seed=seed, n=40)
        rec = tb.Trailblazer(sampler, 310.0, 340.0, target_acc=0.4, min_acc=0.05, max_acc=0.95,
                             sigma_window=window, printf=lambda *_: None).run(make_seed())
        T = rec.temperatures
        ratios = T[1:-2] / T[:-3]
        return np.std(ratios) / np.mean(ratios), rec
    cv_single = np.mean([walk(1, s)[0] for s in range(3)])
    cv_pooled, rec = [walk(5, s) for s in range(3)][0]
    assert cv_pooled < 0.6 * cv_single
    assert rec.inputs['sigma_window'] == 5
    assert all(r.sigma_step is not None for r in rec.rungs)
    # the pooled value is the neff-weighted pooled variance of the trailing window
    r = rec.rungs[7]
    w = rec.rungs[3:8]
    expect = np.sqrt(sum(x.neff * x.sigma ** 2 for x in w) / sum(x.neff for x in w))
    assert r.sigma_step == pytest.approx(expect)


class WindowLimitedSampler(FakeSampler):
    """Means follow the true heat capacity but the sampled fluctuation is 10 % too small,
    the signature of a short single-trajectory window that misses slow modes."""

    def __call__(self, T, seed):
        self.calls.append(T)
        H = self.rng.normal(-1.0e6 + CV * tb.KB * T, 0.9 * tb.KB * T * np.sqrt(CV), size=self.n)
        return H, make_seed(tag=T), {'g': 1.0, 'neff': float(self.n)}


def test_slope_sigma_source_recovers_the_true_metric():
    dL = tb.target_spacing(0.4)
    eps_true = dL * tb.KB * 310.0 / tb.sigma_from_cv(310.0, CV)
    ratio_true = 1.0 / (1.0 - eps_true)
    walks = {}
    for source in ('pooled', 'slope'):
        sampler = WindowLimitedSampler(seed=11, n=400)
        rec = tb.Trailblazer(sampler, 310.0, 340.0, target_acc=0.4, min_acc=0.05, max_acc=0.95,
                             sigma_source=source, printf=lambda *_: None).run(make_seed())
        T = rec.temperatures
        walks[source] = np.median(T[11:-2] / T[10:-3])      # once the slope window is full
        assert rec.inputs['sigma_source'] == source
    # stepping on the under-estimated variance spaces the rungs ~10 % too far apart;
    # the slope route lands on the true ratio
    assert walks['pooled'] - 1.0 == pytest.approx(1.1 * (ratio_true - 1.0), rel=0.05)
    assert walks['slope'] - 1.0 == pytest.approx(ratio_true - 1.0, rel=0.03)


def test_slope_source_falls_back_until_the_window_exists():
    sampler = FakeSampler(seed=12, n=400)
    rec = tb.Trailblazer(sampler, 310.0, 314.0, target_acc=0.4, min_acc=0.05, max_acc=0.95,
                         sigma_source='slope', printf=lambda *_: None).run(make_seed())
    assert rec.rungs[0].cv_slope is None and rec.rungs[2].cv_slope is None
    assert rec.rungs[3].cv_slope is not None and rec.rungs[3].cv_slope == pytest.approx(CV * tb.KB, rel=0.1)
    with pytest.raises(ValueError):
        tb.Trailblazer(sampler, 310.0, 314.0, sigma_source='variance')
