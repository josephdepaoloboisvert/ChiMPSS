#Package Imports
from openmm import *
from openmm.app import *
from openmmtools.states import SamplerState, ThermodynamicState
import numpy as np
import netCDF4 as nc
import os, sys, faulthandler, mpiplus

#Custom Imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FultonMarketUtils import *
from Randolph import Randolph
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysis'))
from FultonMarketAnalysis import FultonMarketAnalysis
from FultonMarketAnalysisUtils import PCA_convergence_detection, plot_MRC

#Set some things
np.seterr(divide='ignore', invalid='ignore')
faulthandler.enable()


class FultonMarket():
    """
    Unrestrained Replica Exchange

    Methods:
         __init__(self, input_pdb: str, input_system: str, input_state: str=None, T_min: float=300, T_max: float=367.447, n_replicates: int=12)
    
        run(self, total_sim_time: float, iter_length: float, dt: float=2.0, sim_length=50,
            init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/'))
    
        _set_init_positions(self)
    
        _set_init_box_vectors(self)
    
        _set_parameters(self)
    
        _build_states(self)
    
        _build_sampler_states(self)
    
        _build_thermodynamic_states(self)
    
        _save_sub_simulation(self)
    
        _load_initial_args(self)
    
        _configure_experiment_parameters(self, sim_length=50)
    
        _recover_arguments(self)
    """

    def __init__(self, 
                 input_pdb: str, 
                 input_system: str, 
                 input_state: str,
                 sele_str: str=None,
                 T_min: float=300, 
                 T_max: float=367.447, 
                 n_replicates: int=12):
        """
        Initialize a Fulton Market obj. 

        Parameters:
        -----------
            input_pdb (str):
                String path to pdb to run simulation. 

            input_system (str):
                String path to OpenMM system (.xml extension) file that contains parameters for simulation. 

            input_state (str):
                String path to OpenMM state (.xml extension) file that contains state for reference. 


        Returns:
        --------
            FultonMarket obj.
        """
        printf('Welcome to FultonMarket.')


        # Set attr
        self.temperatures = [temp*unit.kelvin for temp in geometric_distribution(T_min, T_max, n_replicates)]
        self.n_replicates = n_replicates
        self.sele_str = sele_str

        # Unpack .pdb
        self.input_pdb = input_pdb
        self.pdb = PDBFile(input_pdb)
        self._set_init_positions()
        printf(f'Found input_pdb: {input_pdb}')

        # Unpack .xml
        self.system = XmlSerializer.deserialize(open(input_system, 'r').read())
        self._set_init_box_vectors()
        printf(f'Found input_system: {input_system}')

        # Build state
        if input_state != None:
            integrator = LangevinIntegrator(300, 0.01, 2)
            sim = Simulation(self.pdb.topology, self.system, integrator)
            sim.loadState(input_state)
            self.context = sim.context
            printf(f'Found input_state: {input_state}')



    def _set_init_positions(self):

        # Repeat pdb positions
        self.init_positions = self.pdb.getPositions(asNumpy=True)
        self.n_atoms = self.init_positions.shape[0]
        self.init_positions = [self.init_positions for i in range(self.n_replicates)]

    

    def _set_init_box_vectors(self):

        # Repeat system box vectors
        self.init_box_vectors = self.system.getDefaultPeriodicBoxVectors()
        self.init_box_vectors = [self.init_box_vectors for i in range(self.n_replicates)]
        
        
    
    def run(self, 
            iter_length: float, 
            dt: float=2.0, 
            sim_length: int=50,
            convergence_thresh: float=None,
            resSeqs: np.array=None,
            total_sim_time: int=None, 
            init_overlap_thresh: float=0.5, 
            term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/')):
        
        """
        Run parallel tempering replica exchange.

        Parameters:
        -----------
            iter_length (float):
                Specify the amount of time between swapping replicates in nanoseconds. 
            
            dt (float):
                Timestep for simulation. Default is 2.0 femtoseconds.
            

            sim_length (int):
                Amount of time for each sub simulation in nanoseconds. This value dictates how often .ncdf objects are truncated, data is store, resampling occures, PCA analysis occurs, andconvergence criterion is evaluated. Default is 50, but 25 is recommended. 

            convergence_thresh (float):
                Amount of time the simulation needs to be converged according to the mean weighted reduced cartesians of resampled frames. Default is 0.350 . If this is not None, then this criterion will be used over total simulation time (see below). Default is None.

            resSeqs (np.array):
                Numpy array of resSeqs to use to compute the PCA and evaluate the mean weighted reduced cartesians. If convergence_thresh is not None, this option should be specified. Default is None.
            
            total_sim_time (int):
                Aggregate simulation time from all replicates in nanoseconds. Default is None. If this option is specified and convergence_thresh is None, then this criterion will be used to evaluate when the simulation is complete. 

            init_overlap_thresh (float):
                Acceptance rate threshold during first 50 ns simulation to cause restart. Default is 0.50. 

            term_overlap_thresh (float):
                Terminal acceptance rate. If the minimum acceptance rate every falls below this threshold simulation with restart. Default is 0.35.

            output_dir (str):
                String path to output directory to store files. Default is 'FultonMarket_output' in the current working directory.
        """

        # Set attr
        self.total_sim_time = total_sim_time #ns
        self.iter_length = iter_length #ns
        self.dt = dt
        self.sim_length = sim_length #ns
        self.resSeqs = resSeqs
        self.convergence_thresh = convergence_thresh
        if self.resSeqs is None and self.convergence_thresh is not None: 
            raise Exception(f'You must provide resSeqs if you intend to use the convergence threshold of {self.convergence_thresh} ns.')
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        # Prepare output
        self.output_dir = output_dir
        self.name = output_dir.split('/')[-1]
        self.output_ncdf = os.path.join(self.output_dir, 'output.ncdf')
        self.checkpoint_ncdf = os.path.join(self.output_dir, 'output_checkpoint.ncdf')
        self.save_dir = os.path.join(self.output_dir, 'saved_variables')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        printf(f'Found total simulation time of {self.total_sim_time} nanoseconds')
        printf(f'Found iteration length of {self.iter_length} nanoseconds')
        printf(f'Found timestep of {self.dt} femtoseconds')
        printf(f'Found length of simulation to be {self.sim_length} nanoseconds')
        printf(f'Found number of replicates {self.n_replicates}')
        printf(f'Found initial acceptance rate threshold {self.init_overlap_thresh}')
        printf(f'Found terminal acceptance rate threshold {self.term_overlap_thresh}')
        printf(f'Found output_dir {self.output_dir}')
        printf(f'Found Temperature Schedule {[np.round(T._value, 1) for T in self.temperatures]} Kelvin')
            

        # Loop through short 50 ns simulations to allow for .ncdf truncation
        self._configure_experiment_parameters()
        while not self.finished:

            # Initialize Randolph
            if self.sim_no > 0:
                self._load_initial_args() #sets positions, velocities, box_vecs, temperatures, and spring_constants

            # Build states
            self._build_states()

            # Set parameters
            self._set_parameters()

            
            self.simulation = Randolph(**self.params)
            
            # Run simulation
            self.simulation.main(init_overlap_thresh=init_overlap_thresh, term_overlap_thresh=term_overlap_thresh)

            # Save simulation
            self._save_sub_simulation()

            # Evaluate stop criterion
            self._evaluate_stopping_criterion()

            # Update counter
            self.sim_no += 1



    def _set_parameters(self):

        # Set parameters for Randolph
        self.params = dict(sampler_states=self.sampler_states,
                           thermodynamic_states=self.thermodynamic_states,
                           sim_no=self.sim_no,
                           sim_time=self.sim_length,
                           temperatures=self.temperatures,
                           output_dir=self.output_dir,
                           output_ncdf=self.output_ncdf,
                           checkpoint_ncdf=self.checkpoint_ncdf,
                           iter_length=self.iter_length,
                           dt=self.dt)

    

    def _build_states(self):
        
        # Build sampler and thermodyanic states individually
        self._build_sampler_states()
        self._build_thermodynamic_states()

    

    def _build_sampler_states(self):
        
        # Build sampler states
        if self.sim_no == 0:
            printf('Setting initial positions with the "Context" method')
            self.sampler_states = [SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors).from_context(self.context) for i in range(self.n_replicates)]
        else:
            printf('Setting initial positions with the "Velocity" method')
            self.sampler_states = build_sampler_states(self.n_replicates, self.init_positions, self.init_box_vectors, self.init_velocities)



    def _build_thermodynamic_states(self):
        
        # Build thermodynamic states
        if not hasattr(self, 'thermodynamic_states'):
            self.thermodynamic_states = [ThermodynamicState(system=self.system, temperature=self.temperatures[0], pressure=1.0*unit.bar)] 


    def _save_sub_simulation(self):
        
        # Save temperatures
        self.n_replicates, self.temperatures = self.simulation.save_simulation(self.save_dir)


    
    def _load_initial_args(self):
        
        # Get last directory
        load_no = self.sim_no - 1
        self.load_dir = os.path.join(self.save_dir, str(load_no))
        
        # Load args (not in correct shapes
        self.temperatures = np.load(os.path.join(self.load_dir, 'temperatures.npy'))
        self.temperatures = [t*unit.kelvin for t in self.temperatures]
        self.n_replicates = len(self.temperatures)
        printf(f'Changed n_replicates to {self.n_replicates}')


        # Load from .npy files
        try:
            box_vectors = np.load(os.path.join(self.load_dir, 'box_vectors.npy'))
            n_frames = box_vectors.shape[0]
            init_box_vectors = box_vectors[-1]
            try: # Try loading normally
                init_positions = np.load(os.path.join(self.load_dir, 'positions.npy'))[-1]
            except: # Try loading as a memory map
                init_positions = np.array(np.memmap(os.path.join(self.load_dir, 'positions.npy'), mode='r', dtype='float32', shape=(n_frames, self.n_replicates, self.n_atoms, 3))[-1])
            if os.path.exists(os.path.join(self.load_dir, 'velocities.npy')):
                init_velocities = np.load(os.path.join(self.load_dir, 'velocities.npy')) 
            else:
                init_velocities = None
            state_inds = np.load(os.path.join(self.load_dir, 'states.npy'))[-1]
        except:
            init_velocities, init_positions, init_box_vectors, state_inds = self._recover_arguments()
        
        # Reshape 
        reshaped_init_positions = np.empty((init_positions.shape))
        reshaped_init_box_vectors = np.empty((init_box_vectors.shape))
        for state in range(len(self.temperatures)):
            rep_ind = np.where(state_inds == state)[0]
            reshaped_init_box_vectors[state] = init_box_vectors[rep_ind] 
            reshaped_init_positions[state] = init_positions[rep_ind] 
            
        if init_velocities is not None:
            reshaped_init_velocities = np.empty((init_velocities.shape))
            for state in range(len(self.temperatures)):
                rep_ind = np.where(state_inds == state)[0]
                reshaped_init_velocities[state] = init_velocities[rep_ind] 
                
        # Convert to quantities    
        self.init_positions = convert_to_TrackedQuantity(reshaped_init_positions, unit.nanometer)
        self.init_box_vectors = convert_to_TrackedQuantity(reshaped_init_box_vectors, unit.nanometer)
        if init_velocities is not None:
            self.init_velocities = convert_to_TrackedQuantity(reshaped_init_velocities, (unit.nanometer / unit.picosecond))
        else:
            self.init_velocities = None


    
    def _configure_experiment_parameters(self):
        # Assert that no empty save directories have been made
        assert all([len(os.listdir(os.path.join(self.save_dir, dir))) >= 5 for dir in os.listdir(self.save_dir)]), "You may have an empty save directory, please remove empty or incomplete save directories before continuing :)"
        
        # Configure experiment parameters
        self.sim_no = len(os.listdir(self.save_dir))
        printf(f'Found n_sims_completed to be {self.sim_no}')
        if self.total_sim_time is not None:
            self.total_n_sims = np.ceil(self.total_sim_time / self.sim_length)
            printf(f'Calculated total_n_sims to be {self.total_n_sims}')

        self.finished = False
        if self.sim_no > 0:
            self.converged = False # Deprecated and changed to false, was previously _post_analysis()
    
            # Evaluate stopping criterion 
            self._evaluate_stopping_criterion()


    
    def _recover_arguments(self):
        ncfile = nc.Dataset(self.output_ncdf, 'r')
        
        # Read
        velocities = ncfile.variables['velocities'][-1].data
        positions = ncfile.variables['positions'][-1].data
        box_vectors = ncfile.variables['box_vectors'][-1].data
        state_inds = ncfile.variables['states'][-1].data
        
        ncfile.close()
        
        return velocities, positions, box_vectors, state_inds


    # @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    # def _post_analysis(self):
    #     """
    #     For post analysis:
    #         1. Make output directory
    #         2. Read replica expance output with FultonMarketAnalysis
    #         3. Detect equilibration
    #         4. Importance resampling (top 99.9%)
    #         5. Write out trajectory and mbar weights
    #         6. Compute PCA 
    #         7. Save out mean weighted reduced cartesian relative to endstate
    #     """

    #     # Make output directory, if needed
    #     if not hasattr(self, 'post_analysis_dir'):
    #         self.post_analysis_dir = os.path.join(self.output_dir, 'post_analysis')
    #     if not os.path.exists(self.post_analysis_dir):
    #         os.mkdir(self.post_analysis_dir)

    #     # Determine how much simulation has already been ran
    #     domain = int(self.sim_no * self.sim_length * self.iter_length * 1000 * 10)
    #     domain_save_dir = os.path.join(self.post_analysis_dir, str(domain))
    #     if not os.path.exists(domain_save_dir):
    #         os.mkdir(domain_save_dir)
    #     pdb_out = os.path.join(domain_save_dir, f'{self.name}.pdb')
    #     dcd_out = os.path.join(domain_save_dir, f'{self.name}.dcd')
    #     weights_out = os.path.join(domain_save_dir, f'{self.name}.npy')
    #     rc_out = os.path.join(domain_save_dir, f'{self.name}_mean_weighted_rc.npy')
    #     rc_err_out = os.path.join(domain_save_dir, f'{self.name}_mean_weighted_rc_err.npy')

    #     # Interact with FultonmarketAnalysis
    #     analysis = FultonMarketAnalysis(self.output_dir, self.input_pdb, resSeqs=self.resSeqs, sele_str=self.sele_str) 
    #     analysis.determine_equilibration()
    #     analysis.importance_resampling(n_samples=1000)
    #     analysis.plot_weights(savefig=os.path.join(domain_save_dir, 'weights_plot.png'))
    #     analysis.write_resampled_traj(pdb_out, dcd_out, weights_out)
    #     analysis.get_PCA()

    #     # Get mean weighted reduced cartesians
    #     domains = np.zeros(len(analysis.unshaped_energies), int)
    #     mean_weighted_rc = np.empty(len(domains))
    #     mean_weighted_rc_err = np.empty(len(domains))
    #     frame_counter = 0
    #     for i, e in enumerate(analysis.unshaped_energies):
        
    #         # Analyze simulation domain
    #         n_frames, n_states = e.shape[:2]
    #         sim_time_per_frame = self.iter_length * len(analysis.temperatures)
    #         sub_sim_length = sim_time_per_frame * analysis.energies.shape[0]
    #         domain = sub_sim_length + domains[i-1]
    #         domains[i] = domain
    #         frame_slice = n_frames + frame_counter
    #         frame_counter += n_frames
        
    #         # Get mean weighted rc
    #         mean_weighted_rc[i], mean_weighted_rc_err[i] = analysis.get_weighted_reduced_cartesian(rc_upper_limit=frame_slice, return_weighted_rc=True)

    #     # Save mean weighted rc
    #     np.save(rc_out, mean_weighted_rc)
    #     np.save(rc_err_out, mean_weighted_rc_err)
    #     plot_MRC(domains, mean_weighted_rc, mean_weighted_rc_err, savefig=os.path.join(domain_save_dir, 'MRC_plot.png'))

    #     # Detect equilibration
    #     converged = PCA_convergence_detection(mean_weighted_rc, mean_weighted_rc_err)
    #     sim_time_converged = converged.sum() * self.sim_length * self.iter_length * 1000
    #     printf(f'Detected {sim_time_converged} ns converged.')
    #     if self.convergence_thresh is not None and sim_time_converged >= self.convergence_thresh:
    #         printf(f'Convergence criterion of {self.convergence_thresh} met. Stopping here.') 
    #         return True
    #     else:
            
    #         if self.convergence_thresh is not None:
    #             print(f'Convergence criterion of {self.convergence_thresh} not met. Continuing...')
    #         return False


    
    def _evaluate_stopping_criterion(self):

        # If convergence threshold is specified, use that
        if self.convergence_thresh is not None:
            if self.converged:
                self.finished = True
        elif self.total_sim_time is not None:
            if self.sim_no >= self.total_n_sims:
                self.finished = True
        
        


