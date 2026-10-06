"""GPU regression test: trailblaze a short ladder on the RS1125 membrane system.

Marked slow and gpu; needs openmm, openmmtools and an accelerator.  Walks
310 -> 313 K with deliberately short sampling (minutes, not an hour), then
checks the walk's outputs, its cache, and that FultonMarket's own entry
point resolves the same ladder without re-walking.
"""
import json
import os
import time

import numpy as np
import pytest

pytest.importorskip('openmm')
pytest.importorskip('openmmtools')

HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(os.path.dirname(HERE), '..', 'test_data', 'RS_sets')
PDB = os.path.join(RS, 'RS1125.pdb')
SYSTEM = os.path.join(RS, 'RS1125_sys.xml')
STATE = os.path.join(RS, 'RS1125_state.xml')

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not all(os.path.exists(p) for p in (PDB, SYSTEM, STATE)),
                                 reason='RS1125 fixtures missing')]

SHORT = dict(target_acc=0.4, min_acc=0.3, first_equil_ps=4.0, equil_ps=2.0, sample_ps=10.0,
             sample_interval_ps=0.5, max_sample_ps=20.0, neff_min=10)


def test_walk_on_rs1125_and_cache_reuse(tmp_path):
    from chimpss.fultonmarket import FultonMarket
    from chimpss.fultonmarket.trailblaze import LADDER_NAME, RECORD_NAME, SeedStore

    market = FultonMarket(input_pdb=PDB, input_system=SYSTEM, input_state=STATE, T_min=310.0, T_max=313.0,
                          ladder='trailblaze', trailblaze_kwargs=SHORT)
    t = time.time()
    temps, meta = market.trailblaze(str(tmp_path), dt=2.0)
    walked = time.time() - t
    T = np.asarray(temps)

    assert T[0] == 310.0 and T[-1] == 313.0 and np.all(np.diff(T) > 0)
    assert 3 <= T.size <= 8
    assert meta['n_rungs'] == T.size and meta['complete']
    assert meta['acc_min'] >= 0.3 or meta['acc_min'] is None
    with open(tmp_path / 'trailblaze' / RECORD_NAME) as fh:
        rec = json.load(fh)
    sig = [r['sigma'] for r in rec['rungs']]
    assert all(300.0 < s < 3000.0 for s in sig)         # ~45k atoms: a few hundred to ~1000 kJ/mol
    assert all(r['n_samples'] >= 10 for r in rec['rungs'])
    store = SeedStore(str(tmp_path / 'trailblaze'))
    assert store.has_all(T.size)
    seed, H = store.load(T.size - 1)
    assert seed.positions.shape == (market.n_atoms, 3) and seed.velocities.shape == (market.n_atoms, 3)
    assert np.array_equal(np.load(tmp_path / 'trailblaze' / LADDER_NAME), T)
    assert market.n_replicates == T.size and len(market.init_positions) == T.size

    # a second call must come from the cache, in seconds, without touching the seeds
    mtimes = {i: os.path.getmtime(store.path(i)) for i in range(T.size)}
    t = time.time()
    temps2, meta2 = market.trailblaze(str(tmp_path), dt=2.0)
    assert time.time() - t < min(5.0, walked)
    assert temps2 == temps and meta2['n_rungs'] == meta['n_rungs']
    assert {i: os.path.getmtime(store.path(i)) for i in range(T.size)} == mtimes


def test_run_resolves_the_cached_ladder_without_walking(tmp_path, monkeypatch):
    """FultonMarket.run() with ladder='trailblaze' must pick up the cache and set the thresholds."""
    from chimpss.fultonmarket import FultonMarket

    market = FultonMarket(input_pdb=PDB, input_system=SYSTEM, input_state=STATE, T_min=310.0, T_max=312.0,
                          ladder='trailblaze', trailblaze_kwargs=SHORT)
    temps, _ = market.trailblaze(str(tmp_path), dt=2.0)

    # drive only the part of run() that resolves the ladder
    market.output_dir = str(tmp_path)
    market.dt = 2.0
    market.sim_no = 0
    market.init_overlap_thresh = None
    market.term_overlap_thresh = None
    market.insertion_rule = None
    calls = []
    monkeypatch.setattr(market, '_trailblaze_rank0', lambda: (calls.append(1), (temps, market.ladder_meta))[1])
    market._resolve_ladder()
    market._resolve_overlap_thresholds()
    assert calls == [1]
    assert [t._value for t in market.temperatures] == temps
    assert market.init_overlap_thresh == 0.3 and market.term_overlap_thresh == 0.3


def test_down_walk_on_rs1125_heats_then_walks_down(tmp_path):
    from chimpss.fultonmarket import FultonMarket
    from chimpss.fultonmarket.trailblaze import LADDER_NAME, RECORD_NAME, SeedStore, TrailblazeRecord

    market = FultonMarket(input_pdb=PDB, input_system=SYSTEM, input_state=STATE, T_min=310.0, T_max=313.0,
                          ladder='trailblaze', trailblaze_kwargs={**SHORT, 'direction': 'down', 'ramp_K': 2.0,
                                                                  'ramp_ps': 2.0})
    temps, meta = market.trailblaze(str(tmp_path), dt=2.0)
    T = np.asarray(temps)
    assert meta['direction'] == 'down' and meta['complete']
    assert T[0] == 310.0 and T[-1] == 313.0 and np.all(np.diff(T) > 0)
    assert 3 <= T.size <= 8
    rec = TrailblazeRecord.load_if_complete(str(tmp_path / 'trailblaze' / RECORD_NAME))
    assert rec is not None and rec.direction == 'down'
    assert [r.walk_index for r in rec.rungs] == list(range(T.size - 1, -1, -1))
    assert rec.prepare['T'] == 313.0 and rec.prepare['T_from'] == 310.0
    assert rec.prepare['stages'] == [310.0, 312.0]               # 3 K in 2 K stages, then the 1 K jump to 313
    store = SeedStore(str(tmp_path / 'trailblaze'))
    assert store.has_all(T.size)
    seed, H = store.load(0)
    assert seed.positions.shape == (market.n_atoms, 3) and seed.velocities.shape == (market.n_atoms, 3)
    assert H.size >= 10
    assert np.array_equal(np.load(tmp_path / 'trailblaze' / LADDER_NAME), T)
    assert all(np.isfinite(rec.pair_acc()))


def test_both_directions_on_rs1125_reconcile_and_seed_block_zero(tmp_path, monkeypatch):
    from chimpss.fultonmarket import FultonMarket
    from chimpss.fultonmarket.trailblaze import (
        DOWN_DIR, LADDER_NAME, RECONCILE_NAME, RECORD_NAME, UP_DIR, SeedStore, TrailblazeRecord)

    market = FultonMarket(input_pdb=PDB, input_system=SYSTEM, input_state=STATE, T_min=310.0, T_max=312.0,
                          ladder='trailblaze', trailblaze_kwargs={**SHORT, 'direction': 'both', 'max_acc': 0.95})
    temps, meta = market.trailblaze(str(tmp_path), dt=2.0)
    T = np.asarray(temps)
    tb_dir = tmp_path / 'trailblaze'
    assert meta['direction'] == 'both' and meta['complete'] and meta['n_rungs'] == T.size
    up = TrailblazeRecord.load_if_complete(str(tb_dir / UP_DIR / RECORD_NAME))
    down = TrailblazeRecord.load_if_complete(str(tb_dir / DOWN_DIR / RECORD_NAME))
    assert up is not None and down is not None and up.direction == 'up' and down.direction == 'down'
    assert down.prepare['T_from'] == 312.0 and down.prepare['stages'] == []    # seeded from the hot rung, no ramp
    assert T[0] == 310.0 and T[-1] == 312.0 and np.all(np.diff(T) > 0)
    assert abs(T.size - len(up.rungs)) <= 2 and abs(T.size - len(down.rungs)) <= 2
    assert np.array_equal(np.load(tb_dir / LADDER_NAME), T)
    assert os.path.exists(tb_dir / RECONCILE_NAME)
    with open(tb_dir / RECONCILE_NAME) as fh:
        d = json.load(fh)
    assert d['K_consensus'] == T.size and d['K_up'] == len(up.rungs) and d['K_down'] == len(down.rungs)
    store = SeedStore(str(tb_dir))
    assert store.has_all(T.size)
    rec = TrailblazeRecord.load_if_complete(str(tb_dir / RECORD_NAME))
    assert all(r.source == 'down' for r in rec.rungs)

    # block 0 of a production run on this ladder starts every replica from its consensus seed
    market.output_dir = str(tmp_path)
    market.dt = 2.0
    market.sim_no = 0
    market.init_overlap_thresh = None
    market.term_overlap_thresh = None
    market.insertion_rule = None
    market._resolve_ladder()
    market._resolve_overlap_thresholds()
    assert market.n_replicates == T.size and market.init_overlap_thresh == 0.3
    market._build_sampler_states()
    states = market.sampler_states
    assert len(states) == T.size
    for k, ss in enumerate(states):
        assert ss.positions.shape == (market.n_atoms, 3)
        assert ss.velocities is not None and ss.velocities.shape == (market.n_atoms, 3)

    # the cache is returned without walking again
    t = time.time()
    temps2, meta2 = market.trailblaze(str(tmp_path), dt=2.0)
    assert time.time() - t < 5.0 and temps2 == temps and meta2['direction'] == 'both'
