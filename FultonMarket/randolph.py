# Package Imports
import glob, itertools, jax, math, mpiplus, os, sys
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import mdtraj as md
import netCDF4 as nc
import numpy as np
import jax.numpy as jnp
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmmtools import cache, states, mcmc, multistate
from openmmtools.multistate import ParallelTemperingSampler, MultiStateReporter
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.utils import get_fastest_platform
from openmmtools.utils.utils import TrackedQuantity
from pymbar import timeseries, MBAR
from pymbar.timeseries import detect_equilibration
import scipy.constants as cons
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List
import warnings
warnings.filterwarnings('ignore')

#Custom Imports
from .FultonMarketUtils import (
    printf, truncate_ncdf, build_sampler_states,
    _interpolate_new_states, _interpolate_new_positions,
)
np.seterr(divide='ignore', invalid='ignore')


class Randolph():
    """
    Parallel tempering replica exchange sampler, wrapping OpenMMTools
    ParallelTemperingSampler with adaptive temperature ladder management.

    During each cycle, acceptance rates between adjacent replicas are evaluated
    and new intermediate states are inserted wherever the rate falls below the
    specified threshold. The simulation is then restarted from the current
    positions with the expanded ladder, preserving all trajectory data via
    periodic NetCDF truncation.

    Parameters
    ----------
    sampler_states : list of SamplerState
        Initial positions, box vectors, and (optionally) velocities for each
        replica.
    reference_thermo_state : ThermodynamicState
        Reference thermodynamic state. The full ladder is built internally
        by ParallelTemperingSampler.
    sim_no : int
        Index of the current sub-simulation (0-based). Controls whether the
        initial or terminal acceptance rate threshold is applied and whether
        velocities are expected to be present.
    sim_time : float
        Total aggregate simulation time for this sub-simulation in
        nanoseconds.
    temperatures : np.ndarray
        1-D array of temperatures in Kelvin defining the replica ladder.
    output_dir : str
        Path to the directory where NetCDF output files are written.
    output_ncdf : str
        Path to the primary trajectory NetCDF file.
    checkpoint_ncdf : str
        Path to the checkpoint NetCDF file.
    iter_length : float
        Time between replica swap attempts in nanoseconds.
    dt : float
        Integration timestep in femtoseconds.
    """

    def __init__(self,
                 sampler_states,
                 reference_thermo_state,
                 sim_no: int,
                 sim_time: float,
                 temperatures: np.ndarray,
                 output_dir: str,
                 output_ncdf: str,
                 checkpoint_ncdf: str,
                 iter_length: float,
                 dt: float):

        self.sampler_states = sampler_states
        self.reference_state = reference_thermo_state
        self.sim_no = sim_no
        self.sim_time = sim_time
        self.output_dir = output_dir
        self.output_ncdf = output_ncdf
        self.checkpoint_ncdf = checkpoint_ncdf
        self.temperatures = temperatures.copy()
        self.n_replicates = len(self.temperatures)
        self.iter_length = iter_length
        self.dt = dt

        self._configure_simulation_parameters()
        self._build_simulation()


    def main(self, init_overlap_thresh: float, term_overlap_thresh: float):
        """
        Run the simulation until the target number of cycles is complete.

        On the first sub-simulation (`sim_no == 0`), the initial acceptance
        rate threshold is applied. On all subsequent sub-simulations the
        terminal threshold is applied instead. Whenever an acceptance rate
        falls below the active threshold, new intermediate replicas are
        inserted and the cycle counter resets.

        Parameters
        ----------
        init_overlap_thresh : float
            Minimum acceptable exchange rate during the first sub-simulation.
            Triggers replica insertion if any adjacent pair falls below this
            value.
        term_overlap_thresh : float
            Minimum acceptable exchange rate for all subsequent
            sub-simulations.
        """
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        self.current_cycle = 0
        while self.current_cycle <= self.n_cycles:
            self._run_cycle()


    @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    def save_simulation(self, save_dir: str):
        """
        Persist trajectory data to disk and truncate the NetCDF files to
        reclaim disk space.

        Writes positions, velocities, box vectors, state indices, energies,
        and temperatures as numpy arrays into a numbered sub-directory under
        `save_dir`. Both the primary and checkpoint NetCDF files are then
        truncated in-place by replacing them with the truncated copies.

        Decorated with `@mpiplus.on_single_node` so that file I/O is only
        performed on MPI rank 0; the result is broadcast to all other ranks.

        Parameters
        ----------
        save_dir : str
            Root directory under which per-sub-simulation subdirectories are
            created (e.g. ``save_dir/0/``, ``save_dir/1/``, ...).

        Returns
        -------
        n_replicates : int
            Number of replicas after saving (may differ from the value at
            construction if states were inserted during this sub-simulation).
        temperatures : list of Quantity
            Updated temperature ladder in Kelvin.
        """
        save_no_dir = os.path.join(save_dir, str(self.sim_no))
        os.makedirs(save_no_dir, exist_ok=True)

        # Truncate primary trajectory and extract arrays
        ncdf_copy = os.path.join(self.output_dir, 'output_copy.ncdf')
        pos_memmap, velos, box_vectors, states, energies, temperatures = truncate_ncdf(
            self.output_ncdf, ncdf_copy, save_no_dir, self.reporter, False
        )
        np.save(os.path.join(save_no_dir, 'velocities.npy'), velos.data);       del velos
        np.save(os.path.join(save_no_dir, 'box_vectors.npy'), box_vectors.data); del box_vectors
        np.save(os.path.join(save_no_dir, 'states.npy'), states.data);           del states
        np.save(os.path.join(save_no_dir, 'energies.npy'), energies.data);       del energies
        np.save(os.path.join(save_no_dir, 'temperatures.npy'), temperatures)

        # Truncate checkpoint trajectory
        checkpoint_copy = os.path.join(self.output_dir, 'output_checkpoint_copy.ncdf')
        truncate_ncdf(self.checkpoint_ncdf, checkpoint_copy, save_no_dir, self.reporter, True)

        # Replace originals with truncated copies
        os.replace(ncdf_copy, self.output_ncdf)
        os.replace(checkpoint_copy, self.checkpoint_ncdf)

        try:
            self.reporter.close()
        except Exception:
            pass

        return len(temperatures), [t * unit.kelvin for t in temperatures]


    def _configure_simulation_parameters(self):
        """
        Derive all iteration and cycle counts from the aggregate simulation
        time, number of replicates, swap interval, and timestep.

        The simulation is divided into cycles of approximately 5 iterations
        each to allow periodic acceptance-rate evaluation and, if necessary,
        replica insertion without running the full sub-simulation before
        checking.

        Sets
        ----
        self.n_replicates : int
            Number of replicas (re-read from ``self.temperatures`` to catch
            any updates after state insertion).
        self.n_steps_per_iter : int
            MD steps between each replica swap attempt.
        self.n_iters : int
            Total number of swap iterations for this sub-simulation.
        self.n_cycles : int
            Number of evaluation cycles (each covering ~5 iterations).
        self.n_iters_per_cycle : int
            Number of iterations advanced per cycle.
        self.checkpoint_interval : int
            Number of iterations between checkpoint writes.
        """
        self.n_replicates = len(self.temperatures)
        sim_time_per_rep = self.sim_time / self.n_replicates
        printf(f'Simulation time per replicate : {np.round(sim_time_per_rep, 6)} ns')
        steps_per_rep = np.ceil(sim_time_per_rep * 1e6 / self.dt)
        printf(f'Steps per replicate           : {np.round(steps_per_rep, 0):.0f}')
        self.n_steps_per_iter = int(self.iter_length * 1e6 / self.dt)
        printf(f'Steps per iteration           : {self.n_steps_per_iter}')
        self.n_iters = int(np.ceil(steps_per_rep / self.n_steps_per_iter))
        printf(f'Total iterations              : {self.n_iters}')
        self.n_cycles = int(np.ceil(self.n_iters / 5))
        printf(f'Total cycles                  : {self.n_cycles}')
        self.n_iters_per_cycle = int(np.ceil(self.n_iters / self.n_cycles))
        printf(f'Iterations per cycle          : {self.n_iters_per_cycle}')
        self.checkpoint_interval = max(1, int(0.02 / self.iter_length))
        printf(f'Checkpoint interval           : {self.checkpoint_interval} iterations')
        printf(f'Temperature ladder ({self.n_replicates} reps) : '
               f'{[np.round(t._value, 1) for t in self.temperatures]} K')


    @mpiplus.on_single_node(0, sync_nodes=True)
    def _remove_ncdf(self):
        """
        Delete the primary output NetCDF file if it exists.

        Called before building a new simulation to ensure a clean slate.
        Decorated with `@mpiplus.on_single_node` so deletion only occurs on
        MPI rank 0.
        """
        if os.path.exists(self.output_ncdf):
            printf(f'Removing {self.output_ncdf}')
            os.remove(self.output_ncdf)


    def _build_simulation(self):
        """
        Construct the OpenMMTools sampler, reporter, and simulation object.

        Creates a LangevinDynamicsMove with the configured timestep and
        steps per iteration, then instantiates a ParallelTemperingSampler.
        The MultiStateReporter is configured to write all particle positions
        at the specified checkpoint interval.

        Sets
        ----
        self.simulation : ParallelTemperingSampler or ReplicaExchangeSampler
            The configured OpenMMTools multistate sampler, ready to run.
        self.reporter : MultiStateReporter
            NetCDF reporter attached to ``self.output_ncdf``.
        """
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond,
                                         collision_rate=1.0 / unit.picosecond,
                                         n_steps=self.n_steps_per_iter,
                                         reassign_velocities=False)
        
        self.simulation = ParallelTemperingSampler(mcmc_moves=move,
                                                   number_of_iterations=self.n_iters,
                                                   replica_mixing_scheme='swap-all')
        
        self.simulation._global_citation_silence = True
        
        self._remove_ncdf()
        
        atom_inds = tuple(range(self.reference_state.system.getNumParticles()))
        
        self.reporter = MultiStateReporter(self.output_ncdf,
                                           checkpoint_interval=self.checkpoint_interval,
                                           analysis_particle_indices=atom_inds)
        self.simulation.create(thermodynamic_state=self.reference_state,
                               sampler_states=self.sampler_states[0],
                               storage=self.reporter,
                               temperatures=self.temperatures,
                               n_temperatures=self.n_replicates)


    def _run_cycle(self):
        """
        Advance the simulation by one cycle and evaluate exchange acceptance
        rates.

        Runs ``self.n_iters_per_cycle`` iterations (extending if the sampler
        has already reached its target iteration count). After advancing,
        acceptance rates between all adjacent replica pairs are checked. If
        any pair falls below the active threshold, new intermediate states are
        inserted via ``_interpolate_states``, the reporter is closed, and the
        cycle counter resets to 0 so the expanded ladder runs from the
        beginning of a fresh simulation. If all rates are acceptable, the
        cycle counter is incremented.
        """
        thresh = self.init_overlap_thresh if self.sim_no == 0 else self.term_overlap_thresh

        printf(f'Cycle {self.current_cycle}: advancing {self.n_iters_per_cycle} iterations.')
        if self.simulation.is_completed:
            self.simulation.extend(self.n_iters_per_cycle)
        else:
            self.simulation.run(self.n_iters_per_cycle)

        insert_inds = self._eval_acc_rates(thresh)

        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.reporter.close()
            self.current_cycle = 0
            self._configure_simulation_parameters()
            self._build_simulation()
        else:
            self.current_cycle += 1


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _eval_acc_rates(self, acceptance_rate_thresh: float = 0.40) -> np.ndarray:
        """
        Read mixing statistics from the reporter and identify adjacent replica
        pairs with exchange rates below the given threshold.

        Acceptance rates are averaged over all recorded swap attempts
        (excluding the first frame, which may contain zeros). NaN values —
        arising from pairs with zero proposed swaps — are replaced with 0.

        Decorated with `@mpiplus.on_single_node` so reporter reads only occur
        on rank 0; results are broadcast to all ranks.

        Parameters
        ----------
        acceptance_rate_thresh : float
            Minimum acceptable exchange rate. Default 0.40.

        Returns
        -------
        insert_inds : np.ndarray of int
            Indices at which a new intermediate state should be inserted.
            An index ``k`` means a new state should be placed between the
            current states ``k-1`` and ``k``.
        """
        temperatures = [float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]]
        accepted, proposed = self.reporter.read_mixing_statistics()
        acc_rates = np.nan_to_num(np.mean(accepted[1:] / proposed[1:], axis=0))
        
        insert_inds = []
        for state in range(len(acc_rates) - 1):
            rate = acc_rates[state, state + 1]
            printf(f'Exchange {np.round(temperatures[state], 2)} K <-> {np.round(temperatures[state + 1], 2)} K : {rate:.4f}')
            if rate < acceptance_rate_thresh:
                insert_inds.append(state + 1)
        return np.array(insert_inds)


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _read_temps_from_reporter(self) -> np.ndarray:
        """
        Read the current temperature ladder directly from the reporter.

        Decorated with `@mpiplus.on_single_node` so the reporter read only
        occurs on rank 0; the result is broadcast to all ranks.

        Returns
        -------
        temperatures : np.ndarray
            1-D array of temperatures in Kelvin, one per replica, in state
            index order.
        """
        return np.array([float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]])


    def _interpolate_states(self, insert_inds: np.ndarray):
        """
        Insert new intermediate replicas at the specified positions in the
        temperature ladder.

        New temperatures are computed as the geometric mean of the adjacent
        pair (via ``_interpolate_new_states``). Positions, box vectors, and
        velocities for the new replicas are linearly interpolated from their
        neighbours (via ``_interpolate_new_positions``). The sampler states
        are then rebuilt with the expanded set of replicas.

        Parameters
        ----------
        insert_inds : np.ndarray of int
            State indices at which to insert new replicas, as returned by
            ``_eval_acc_rates``.
        """
        prev_temps = self._read_temps_from_reporter()
        self.temperatures, self.n_replicates = _interpolate_new_states(prev_temps, insert_inds)
        
        init_positions, init_box_vectors, init_velocities = self._load_inits()
        init_positions, init_box_vectors, init_velocities = _interpolate_new_positions(init_positions,
                                                                                       init_box_vectors,
                                                                                       init_velocities,
                                                                                       insert_inds,
                                                                                       self.n_replicates)
        
        self.sampler_states = build_sampler_states(self.n_replicates,
                                                   init_positions,
                                                   init_box_vectors,
                                                   init_velocities)


    def _load_inits(self):
        """
        Extract current positions, box vectors, and velocities from the
        sampler states.

        Velocities are only read when ``sim_no > 0`` and the first sampler
        state has a non-None velocities array; otherwise ``None`` is returned
        for velocities so that OpenMM will generate them from the Maxwell-
        Boltzmann distribution at the start of the next simulation.

        Returns
        -------
        init_positions : np.ndarray
            Positions array of shape (n_replicates, n_atoms, 3) in nanometers.
        init_box_vectors : np.ndarray
            Box vectors array of shape (n_replicates, 3, 3) in nanometers.
        init_velocities : np.ndarray or None
            Velocities array of shape (n_replicates, n_atoms, 3) in nm/ps, or
            None if not available.
        """
        n = len(self.sampler_states)
        init_positions  = np.array([self.sampler_states[i].positions._value.copy()   for i in range(n)])
        init_box_vectors = np.array([self.sampler_states[i].box_vectors._value.copy() for i in range(n)])
        
        if self.sim_no > 0 and self.sampler_states[0].velocities is not None:
            init_velocities = np.array([self.sampler_states[i].velocities._value.copy() for i in range(n)])
        else:
            init_velocities = None
        
        return init_positions, init_box_vectors, init_velocities