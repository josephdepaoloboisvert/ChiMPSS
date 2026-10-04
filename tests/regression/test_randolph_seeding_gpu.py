"""GPU regression test: every replica starts a block from its own configuration.

Until 2026-10 Randolph handed openmmtools a single SamplerState, which the
sampler round-robins onto every replica, so each block began with all
replicas on the state-0 configuration (identical enthalpy at iteration 0 on
every saved run).  This builds a three-rung Randolph with three distinct
sampler states, runs one iteration, and checks that the energies the
reporter stored for iteration 0 differ between replicas and match the
configurations handed in.
"""
import os

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


def test_each_replica_starts_from_its_own_sampler_state(tmp_path):
    import openmm.unit as unit
    from openmmtools.states import SamplerState, ThermodynamicState

    from chimpss.fultonmarket import FultonMarket, Randolph
    from chimpss.fultonmarket.trailblaze import OpenMMRungSampler

    market = FultonMarket(input_pdb=PDB, input_system=SYSTEM, input_state=STATE, T_min=310.0, T_max=312.0,
                          n_replicates=3)
    base = SamplerState.from_context(market.context)

    # three configurations that differ by a few picoseconds of dynamics at 310 K
    sampler = OpenMMRungSampler(market.system, dt_fs=2.0, equil_ps=1.0, sample_ps=1.0, sample_interval_ps=0.5,
                                max_sample_ps=1.0, neff_min=1, first_equil_ps=1.0)
    seed = sampler.seed_from_sampler_state(base)
    seeds = []
    for _ in range(3):
        _, seed, _ = sampler(310.0, seed)
        seeds.append(seed)
    states = [SamplerState(positions=s.positions * unit.nanometer, box_vectors=s.box_vectors * unit.nanometer,
                           velocities=s.velocities * (unit.nanometer / unit.picosecond)) for s in seeds]
    expected_U = []
    ref = ThermodynamicState(system=market.system, temperature=310.0 * unit.kelvin, pressure=1.0 * unit.bar)
    for s in seeds:
        ss = sampler.sampler_state(s)
        sampler._chunk._global_citation_silence = True
        # potential energy of the handed-in configuration, via the same reduced potential the sampler uses
        from openmmtools import cache
        ctx, integ = cache.global_context_cache.get_context(ref)
        ss.apply_to_context(ctx)
        expected_U.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert len({round(u, 1) for u in expected_U}) == 3          # genuinely distinct configurations

    r = Randolph(sampler_states=states, reference_thermo_state=ref, sim_no=0, sim_time=0.0001,
                 temperatures=[310.0 * unit.kelvin, 311.0 * unit.kelvin, 312.0 * unit.kelvin],
                 output_dir=str(tmp_path), output_ncdf=str(tmp_path / 'out.ncdf'),
                 checkpoint_ncdf=str(tmp_path / 'out_checkpoint.ncdf'), iter_length=0.0005, dt=2.0)
    try:
        r.simulation.run(1)
        u0, _, _ = r.reporter.read_energies(iteration=0)        # (replica, state) reduced potentials
    finally:
        try:
            r.reporter.close()
        except Exception:
            pass
    H0 = u0[:, 0] / (1.0 / (0.00831446261815324 * 310.0))      # enthalpy of each replica at iteration 0
    assert np.unique(np.round(H0, 1)).size == 3, 'iteration 0 must not carry one configuration on every replica'
    # H = U + pV; pV is ~26 kJ/mol here and identical per box, so H and U differ by one constant
    order_H = np.argsort(H0)
    order_U = np.argsort(expected_U)
    assert np.array_equal(order_H, order_U)
