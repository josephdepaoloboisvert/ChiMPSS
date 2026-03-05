from openmm import *
from openmm.app import *
from openmmtools import cache
from openmmtools.utils import get_fastest_platform
from openmmtools.utils.utils import TrackedQuantity
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.multistate import ParallelTemperingSampler, ReplicaExchangeSampler, MultiStateReporter
import tempfile
import os, sys
sys.path.append('../MotorRow')
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import netCDF4 as nc
from typing import List
from datetime import datetime
import mdtraj as md
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FultonMarketUtils import *
from FultonMarketUtils import _interpolate_new_states, _interpolate_new_positions # Not being imported in previous line for some reason
import mpiplus
# from mpi4py import MPI

class Randolph():
    """
    """
    
    def __init__(self, 
                 sampler_states,
                 thermodynamic_states,
                 sim_no: int, 
                 sim_time: unit.Quantity, 
                 temperatures: np.array,
                 output_dir: str, 
                 output_ncdf: str, 
                 checkpoint_ncdf: str,
                 iter_length: unit.Quantity, 
                 dt: unit.Quantity,
                 spring_centers: np.array=None,
                 restrained_atom_indices: np.array=None):
        """
        """
        # Assign attributes
        self.sampler_states = sampler_states
        self.thermodynamic_states = thermodynamic_states
        self.sim_no = sim_no
        self.sim_time = sim_time
        self.output_dir = output_dir
        self.output_ncdf = output_ncdf
        self.checkpoint_ncdf = checkpoint_ncdf
        self.temperatures = temperatures.copy()
        self.n_replicates = len(self.temperatures)
        self.iter_length = iter_length
        self.dt = dt
        self.spring_centers = spring_centers
        if self.spring_centers is not None:
            raise NotImplementedError('Interpolation has been deprecated w/ restraints')
        self.restrained_atom_indices = restrained_atom_indices
        
        # Configure simulation parameters
        self._configure_simulation_parameters()
        
        # Build simulation
        self._build_simulation()

    
    def main(self, init_overlap_thresh: float, term_overlap_thresh: float):
        """
        """
        
        # Assign attributes
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        # Continue until self.n_cycles reached
        self.current_cycle = 0
        while self.current_cycle <= self.n_cycles:

            # Advance 1 cycle
            self._run_cycle()

            
    @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    def save_simulation(self, save_dir):
        """
        Save the important information from a simulation and then truncate the output.ncdf file to preserve disk space.
        """
        # Determine save no. 
        save_no_dir = os.path.join(save_dir, str(self.sim_no))
        if not os.path.exists(save_no_dir):
            os.mkdir(save_no_dir)

        # Truncate output.ncdf
        ncdf_copy = os.path.join(self.output_dir, 'output_copy.ncdf')
        pos_memmap, velos, box_vectors, states, energies, temperatures = truncate_ncdf(self.output_ncdf, ncdf_copy, save_no_dir, self.reporter, False)
        np.save(os.path.join(save_no_dir, 'velocities.npy'), velos.data)
        del velos
        np.save(os.path.join(save_no_dir, 'box_vectors.npy'), box_vectors.data)
        del box_vectors
        np.save(os.path.join(save_no_dir, 'states.npy'), states.data)
        del states
        np.save(os.path.join(save_no_dir, 'energies.npy'), energies.data)
        del energies
        np.save(os.path.join(save_no_dir, 'temperatures.npy'), temperatures)
        
        if self.spring_centers is not None: 
            np.save(os.path.join(save_no_dir, 'spring_centers.npy'), self.spring_centers)

        # Truncate output_checkpoint.ncdf
        checkpoint_copy = os.path.join(self.output_dir, 'output_checkpoint_copy.ncdf')
        truncate_ncdf(self.checkpoint_ncdf, checkpoint_copy, save_no_dir, self.reporter, True)

        # Write over previous .ncdf files
        os.system(f'mv {ncdf_copy} {self.output_ncdf}')
        os.system(f'mv {checkpoint_copy} {self.checkpoint_ncdf}')

        # Close reporter object
        try:
            self.reporter.close()
        except:
            pass    
        
        if self.spring_centers is not None:    
            return len(temperatures), [t*unit.kelvin for t in temperatures], self.spring_centers
        else:
            return len(temperatures), [t*unit.kelvin for t in temperatures]
        

    
    def _configure_simulation_parameters(self):
        """
        Configure simulation times to meet aggregate simulation time. 
        """            

        # Read number replicates if different than argument
        self.n_replicates = len(self.temperatures)
        
        # Configure times/steps
        sim_time_per_rep = self.sim_time / self.n_replicates
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated simulation per replicate to be', np.round(sim_time_per_rep, 6), 'nanoseconds', flush=True)
        
        steps_per_rep = np.ceil(sim_time_per_rep * 1e6 / self.dt)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per replicate to be', np.round(steps_per_rep,0), 'steps', flush=True)        
        
        self.n_steps_per_iter = self.iter_length * 1e6 / self.dt
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per iteration to be', np.round(self.n_steps_per_iter, 0), 'steps', flush=True) 
        
        self.n_iters = np.ceil(steps_per_rep / self.n_steps_per_iter)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iterations to be', self.n_iters, 'iterations', flush=True) 
        
        self.n_cycles = np.ceil(self.n_iters / 5)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of cycles to be', self.n_cycles, 'cycles', flush=True) 
        
        self.n_iters_per_cycle = np.ceil(self.n_iters / self.n_cycles)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iters per cycle to be', self.n_iters_per_cycle, 'iterations', flush=True) 

        self.checkpoint_interval = int(0.01 / self.iter_length)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated checkpoint interval to be', self.checkpoint_interval, 'iterations', flush=True) 


        # Configure replicates            
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated temperature of', self.n_replicates,
                                      'replicates to be', [np.round(t._value,1) for t in self.temperatures], flush=True)


    @mpiplus.on_single_node(0, sync_nodes=True)
    def _remove_ncdf(self):
        if os.path.exists(self.output_ncdf):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Removing', self.output_ncdf, flush=True)
            os.remove(self.output_ncdf)


    def _build_simulation(self):
        """
        """
        # Set up integrator
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond, collision_rate=1.0 / unit.picosecond, n_steps=self.n_steps_per_iter, reassign_velocities=False)
        
        # Set up simulation
        if self.spring_centers is not None:
            self.simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        else:
            self.simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        self.simulation._global_citation_silence = True

        # Remove existing .ncdf files
        self._remove_ncdf()

        # Setup reporter
        atom_inds = tuple([i for i in range(self.thermodynamic_states[0].system.getNumParticles())])
        self.reporter = MultiStateReporter(self.output_ncdf, checkpoint_interval=self.checkpoint_interval, analysis_particle_indices=atom_inds)
        
        # Create simulation obj    
        if self.spring_centers is not None:
            self.simulation.create(thermodynamic_states=self.thermodynamic_states, sampler_states=self.sampler_states, storage=self.reporter)
        else:
            self.simulation.create(thermodynamic_state=self.thermodynamic_states[0], sampler_states=self.sampler_states,
                                   storage=self.reporter, temperatures=self.temperatures, n_temperatures=self.n_replicates)   

    
    def _run_cycle(self):
        """
        Run one cycle
        """

        comm = mpiplus.get_mpicomm()
        
        # Take steps
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'CYCLE', self.current_cycle, 'advancing', self.n_iters_per_cycle, 'iterations', flush=True) 
        if self.simulation.is_completed:
            self.simulation.extend(self.n_iters_per_cycle)
        else:
            self.simulation.run(self.n_iters_per_cycle)

        # Eval acceptance rates
        if self.sim_no == 0:
            insert_inds = self._eval_acc_rates(self.init_overlap_thresh)
        else:
            insert_inds = self._eval_acc_rates(self.term_overlap_thresh)

        # Interpolate, if necessary
        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.reporter.close()
            self.current_cycle = 0
            self._configure_simulation_parameters()
            self._build_simulation()
        else:
            self.current_cycle += 1


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True) 
    def _eval_acc_rates(self, acceptance_rate_thresh: float=0.40):
        
        # Get temperatures
        temperatures = [float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]]
        
        # Get mixing statistics
        accepted, proposed = self.reporter.read_mixing_statistics()
        acc_rates = np.mean(accepted[1:] / proposed[1:], axis=0)
        acc_rates = np.nan_to_num(acc_rates) # Adjust for cases with 0 proposed swaps
    
        # Iterate through mixing statistics to flag acceptance rates that are too low
        insert_inds = [] # List of indices to apply new state. Ex: (a "1" means a new state between "0" and the previous "1" indiced state)
        for state in range(len(acc_rates)-1):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Mixing between', np.round(temperatures[state], 2), 'and', np.round(temperatures[state+1], 2), ':', acc_rates[state, state+1], flush=True) 
            rate = acc_rates[state, state+1]
            if rate < acceptance_rate_thresh:
                insert_inds.append(state+1)
        
        return np.array(insert_inds)

    
    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True) 
    def _read_temps_from_reporter(self): 
        return np.array([float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]])

        
    def _interpolate_states(self, insert_inds: np.array):
        
        # Determine new states
        prev_temps = self._read_temps_from_reporter()
        self.temperatures, self.n_replicates = _interpolate_new_states(prev_temps, insert_inds)
        init_positions, init_box_vectors, init_velocities = self._load_inits()
        init_positions, init_box_vectors, init_velocities = _interpolate_new_positions(init_positions, init_box_vectors, init_velocities, insert_inds, self.n_replicates)
        
        # Update Sampler States
        self.sampler_states = build_sampler_states(self.n_replicates, init_positions, init_box_vectors, init_velocities)

        
        # Add new restraints, if necessary, MAY BE DEPRECATED w/ MPI implementation
        # if self.spring_centers is not None:
        #     prev_spring_centers = self.spring_centers
        #     new_spring_centers = self.spring_centers
        #     for displacement, ind in enumerate(insert_inds):
        #         center_below = prev_spring_centers[ind - 1]
        #         center_above = prev_spring_centers[ind]
        #         print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state with new Spring Center', flush=True)
        #         new_center = 0.5*(center_above + center_below)
        #         new_spring_centers = np.insert(new_spring_centers, ind + displacement, new_center, axis=0)
        #     self.spring_centers = new_spring_centers
        #     assert self.spring_centers.shape[0] == len(self.temperatures)

        #     # Update Thermodynamic States
        #     self.system = self.thermodynamic_states[0].system
        #     self.system.removeForce(6)# Remove previous CustomExternalForce
        #     build_thermodynamic_states(self)


    def _load_inits(self):

        init_positions = np.array([self.sampler_states[i].positions._value.copy() for i in range(len(self.sampler_states))])
        init_box_vectors = np.array([self.sampler_states[i].box_vectors._value.copy() for i in range(len(self.sampler_states))])
        if self.sim_no > 0 and self.sampler_states[0].velocities is not None:
            init_velocities = np.array([self.sampler_states[i].velocities._value.copy() for i in range(len(self.sampler_states))])
        else:
            init_velocities = None

        return init_positions, init_box_vectors, init_velocities
            

