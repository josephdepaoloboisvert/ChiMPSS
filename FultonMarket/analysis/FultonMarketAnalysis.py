# Imports
import os, sys, math, glob, jax 
import jax.numpy as jnp
from datetime import datetime
import netCDF4 as nc
import numpy as np
from typing import List
from pymbar import timeseries, MBAR
import scipy.constants as cons
import mdtraj as md
#import dask.array as da
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from sklearn.decomposition import PCA
from pymbar.timeseries import detect_equilibration
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FultonMarketAnalysisUtils import *

printf = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))

jax.print_environment_info()
printf(f"Default JAX backend is {jax.default_backend()}")

class FultonMarketAnalysis():
    """
    Analysis class for Replica Exchange Simulations written with Fulton Market

    methods:
        init: input_dir
    """
    def __init__(self, 
                 input_dir:str, 
                 pdb: str, 
                 skip: int=10, 
                 scheduling: str='Temperature', 
                 resSeqs: List[int]=None, 
                 sele_str: str=None,
                 upper_limit: int=None, 
                 remove_harmonic: bool=False,
                 
                 spring_centers: np.array=None):
        """
        get Numpy arrays, determine indices of interpolations, and set state_inds
        """
        
        # Find directores
        if input_dir.endswith('/'):
            input_dir = input_dir[:-1]
        self.input_dir = input_dir
        self.stor_dir = os.path.join(input_dir, 'saved_variables')
        assert os.path.isdir(self.stor_dir), self.stor_dir
        printf(f"Found storage directory at {self.stor_dir}")
        self.storage_dirs = sorted(glob.glob(self.stor_dir + '/*'), key=lambda x: int(x.split('/')[-1]))
        self.pdb = pdb 
        self.top = md.load_pdb(self.pdb).topology
        if resSeqs is not None:
            self.resSeqs = [[self.top.residue(i).resSeq for i in range(self.top.n_residues)].index(resSeq) for resSeq in resSeqs] # convert to mdtraj resSeqs
        else:
            self.resSeqs = None
        self.sele_str = sele_str
        printf(f'Found ligand selection string of: {sele_str}')       
        
        # Load saved variables
        self.temperatures_list = [np.round(np.load(os.path.join(storage_dir, 'temperatures.npy'), mmap_mode='r'), decimals=2) for storage_dir in self.storage_dirs]
        self.temperatures = self.temperatures_list[-1]
        printf(f"Shapes of temperature arrays: {[(i, temp.shape) for i, temp in enumerate(self.temperatures_list)]}")
        self.state_inds = [np.load(os.path.join(storage_dir, 'states.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
        self.unshaped_energies = [np.load(os.path.join(storage_dir, 'energies.npy'), mmap_mode='r')[skip:] for storage_dir in self.storage_dirs]
        
        # Reshape lists 
        self.energies = self._reshape_list(self.unshaped_energies)

        # Compute positions/box_vectors map 
        self._get_postions_map()
        self.skip = skip        

        # Determine if interpolation occured and resample to fill in missing states
        self.scheduling = scheduling
        self._backfill()

        # Apply upper limit, if specified
        if upper_limit is not None:
            self.energies = self.energies[:upper_limit+1]
            self.map = self.map[:upper_limit+1]
            self.upper_limit = upper_limit
        printf(f'Shape of final energies determined to be: {self.energies.shape}')

        # Remove harmonic, if specified
        if remove_harmonic:
            assert spring_centers is not None
            self.spring_centers = spring_centers.copy()
            self._remove_harmonic()
        
        
    def get_state_energies(self, state_index: int=0):
        """
        get energies of each replicate in its own state (iters, state, state) -> (iters, state)
        Optionally reduce energies based on temperatures, and concatenate the list of arrays to a single array
        """
        
        state_energies = self.energies[:,state_index, state_index]
        
        return state_energies
    
    
    def get_average_energy(self, plot: bool=False, figsize: tuple=(6,6), equilibration_method: str='energy'):
        """
        """

        # Get average energies
        state_energies = np.empty((self.energies.shape[0], self.energies.shape[1]))
        for state in range(self.energies.shape[1]):
            state_energies[:,state] = self.energies[:,state,state]
        self.average_energies = state_energies.mean(axis=1)
        
        # Plot if specified:
        if plot:
            
            # Determine equilibration
            self.determine_equilibration(equilibration_method=equilibration_method)

            # Make plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(range(self.average_energies.shape[0]), self.average_energies, color='k')
            ax.vlines(self.t0, self.average_energies.min(), self.average_energies.max(), color='r', label='equilibration')
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_title('Mean energy')
            ax.set_ylabel('Energy (kJ/mol)')
            ax.set_xlabel('Iterations')
            fig.tight_layout()
            plt.show
            return self.average_energies, fig, ax
            
        else:
            return self.average_energies
        
        
    def plot_energy_distributions(self, figsize: tuple=(8,4), post_equil: bool=False):  

        if post_equil:
            if not hasattr(self, 't0'):
                self.determine_equilibration()
            state_energies = np.array([self.get_state_energies(state_index=state_no)[self.t0:] for state_no in range(self.energies.shape[1])]).T
        else:
            state_energies = np.array([self.get_state_energies(state_index=state_no) for state_no in range(self.energies.shape[1])]).T

        
        fig, ax = plt.subplots(figsize=figsize)
        
        
        sns.kdeplot(state_energies, ax=ax, legend=False)
        ax.set_xlabel('Energy (kJ/mol)')
        fig.tight_layout()
        plt.show()
        
        return fig, ax 
    
    
    def importance_resampling(self, n_samples:int=-1, equilibration_method: str='energy', specify_state:int=0, replace: bool=True):
        """
        """                  
        #Ensure equilibration has been detected
        if not hasattr(self, 't0'):
            self.determine_equilibration(equilibration_method=equilibration_method)
          
        # Get MBAR weights
        self.flat_inds = np.array([[state, ind] for ind in range(self.t0, self.energies.shape[0]) for state in range(self.energies.shape[1])])
        u_kln = np.array([self.energies[self.t0:,:,k].flatten() for k in range(self.energies.shape[2])])
        N_k = np.array([self.energies[self.t0:].shape[0] for i in range(self.energies.shape[2])])
        self.resampled_inds, self.weights, self.resampled_weights = resample_with_MBAR(objs=[self.flat_inds], u_kln=u_kln, N_k=N_k, size=n_samples, return_inds=False, return_weights=True, return_resampled_weights=True, specify_state=0, replace=replace)

    
    def reshape_weights(self):
        # Reshape weights
        self.reshaped_weights = np.zeros((len(self.temperatures), len(self.energies)))
        for ((state, frame), weight) in zip(self.flat_inds, self.weights[:,0]):
            self.reshaped_weights[state, frame] = weight

    
    def plot_weights(self, figsize: tuple=(25,10), savefig: str=None):
        """
        """

        # Reshape weights
        self.reshape_weights()

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.reshaped_weights[:, ::10])
        ax.vlines(self.flat_inds[:,1].min()/10, 0, self.temperatures.shape[0], color='red', label='equilibration')
        ax.set_yticks(np.arange(self.temperatures.shape[0])[::20], self.temperatures[::20])
        ax.set_xlabel('Replicate Simulation Time (ns)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Truncated MBAR weights')
        plt.legend(loc='upper left')
        plt.show()
        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)
            
    
    
    def write_resampled_traj(self, pdb_out: str, dcd_out: str, weights_out: str=None, inds_out: str=None, return_traj: bool=False, correction: bool=True):
        
        # Make sure resampling has already occured
        if not hasattr(self, 'resampled_inds'):
            self.importance_resampling()
            
        # Load pos, box_vec
        if not hasattr(self, 'positions'):
            self._load_positions_box_vecs()
        
        # Use the map to find the resampled configurations
        pos = np.empty((len(self.resampled_inds), self.positions[0].shape[2], 3))
        box_vec = np.empty((len(self.resampled_inds), 3, 3))
        for i, (state, iter) in enumerate(self.resampled_inds):
    
            # Use map
            sim_no, sim_iter, sim_rep_ind = self.map[iter, state].astype(int)
            
            pos[i] = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])
    
        # Write out trajectory
        self.traj = write_traj_from_pos_boxvecs(pos, box_vec, self.top, self.sele_str, correction=correction)
        self.traj[0].save_pdb(pdb_out)
        self.traj.save_dcd(dcd_out)
        printf(f'{self.traj.n_frames} frames written to {pdb_out}, {dcd_out}')
    
        # Save weights, if specified
        if weights_out is not None:
            np.save(weights_out, self.resampled_weights)
            printf(f'{self.traj.n_frames} mbar weights written to {weights_out}')
    
        # Save indices, if specifid
        if inds_out is not None:
            np.save(inds_out, self.resampled_inds)
            printf(f'{self.traj.n_frames} indices weights written to {inds_out}')
        
        if return_traj:
            return self.traj


    def state_trajectory(self, state_no=0, stride: int=1):
        """    
        State_no is the thermodynamics state to retrieve
        If pdb file is provided (top_file), then an MdTraj trajectory will be returned
        If top_file is None - the numpy array of positions will be returned
        """
        if not (hasattr(self, 'positions') or hasattr(self, 'box_vectors')):
            self._load_positions_box_vecs()
  
        # Use the map to find the resampled configurations
        inds = np.arange(0, self.energies.shape[0], stride)
        pos = np.empty((len(inds), self.positions[0].shape[2], 3))
        box_vec = np.empty((len(inds), 3, 3))
        for i, ind in enumerate(inds):
            
            # Use map
            sim_no, sim_iter, sim_rep_ind = self.map[ind, state_no].astype(int)
            pos[i] = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])
        
        # Apply pos, box_vec to mdtraj obj
        traj = write_traj_from_pos_boxvecs(pos, box_vec, self.top, self.sele_str)

        return traj

    def determine_equilibration(self, equilibration_method: str='energy', stride: int=10):
        """
        Automated equilibration detection
        suggests an equilibration index (with respect to the whole simulation) by detecting equilibration for the average energies
        starting from each of the first 50 frames
        returns the likely best index of equilibration
        """

        if hasattr(self, 't0'):
           return 
 
        # PCA
        if equilibration_method == 'PCA':
            self.get_PCA(state_no=0, stride=stride, explained_variance_threshold=0.9)
            
            # Iterate through PCs to detect equilibration
            equil_times = np.empty(self.n_components)
            for pc in range(self.n_components):
                equil_times[pc] = detect_PC_equil(pc, self.reduced_cartesian)
    
    
            # Save equilibration/uncorrelated inds to new variables
            self.t0 = np.sum(equil_times  * (self.explained_variance / self.explained_variance.sum())).astype(int) * stride

        elif equilibration_method == 'energy':
            self.t0 = detect_energy_equil(self.get_average_energy())
         
        elif equilibration_method == 'None':
            self.t0 = 0
        else:
            print('equilibration_method must be either PCA or None')

        printf(f'Equilibration detected at {np.round(self.t0 * self.energies.shape[1] / 1000, 3)} ns with method: {equilibration_method}')

    

    def get_PCA(self, state_no: int=None, stride: int=1, explained_variance_threshold: float=0.9):
            
            # Get state trajectory
            if state_no is None:
                traj = deepcopy(self.traj)
            else:
                traj = self.state_trajectory(state_no, stride)
                if hasattr(self, 'upper_limit'):
                    traj = traj[:self.upper_limit+1]            
        
            # Get protein or resSeqs of interest
            if self.resSeqs is not None:
                sele = traj.topology.select(f'resSeq {" ".join([str(resSeq) for resSeq in self.resSeqs])}')
            else:
                sele = traj.topology.select('protein')
            traj = traj.atom_slice(sele)
        
            # PCA
            pca, self.reduced_cartesian, self.explained_variance, self.n_components = get_traj_PCA(traj, explained_variance_threshold=0.9)
            printf(f'Computed reduced cartesian with shape: {self.reduced_cartesian.shape}')

    
    def get_weighted_reduced_cartesian(self, rc_upper_limit: None, return_weighted_rc: bool=False, use_state: bool=False, stride: int=1):
        """
        """
        if rc_upper_limit is None:
            rc_upper_limit = np.inf

        if use_state:
            state_flat_inds = np.array([[0, ind] for ind in range(self.energies.shape[0])])[::stride]
            state_weights = np.repeat(1, state_flat_inds.shape[0])
            self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err = calculate_weighted_rc(self.reduced_cartesian, state_flat_inds, rc_upper_limit, self.explained_variance, state_weights)
        else:
            self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err = calculate_weighted_rc(self.reduced_cartesian, self.resampled_inds, rc_upper_limit, self.explained_variance, self.resampled_weights)

        if return_weighted_rc:
            return self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err


    def truncate(self):
    
        # Reload sim positions without skip
        self._load_positions_box_vecs(skip=0) 
    
        # Get indices to remove
        keep_inds = get_truncation_atom_keep_inds(self.top)
        traj = md.load_pdb(self.pdb).atom_slice(keep_inds)
        self.top = deepcopy(traj.topology)
        
        # Iterate through save directories and remove positions from unused thermodynamic states
        for i, storage_dir in enumerate(self.storage_dirs[:-2]):
    
            # Check if already been truncated
            top_fn = os.path.join(storage_dir, 'topology.pdb')
            if os.path.exists(top_fn):
                continue
    
            # Truncate positions
            init_shape = self.positions[i].shape
            sim_pos = np.array(self.positions[i][:,:,keep_inds,:])
            printf(f'Reduced positions of {storage_dir} from {init_shape} to: {sim_pos.shape}')
    
            # Remove .npy file
            pos_fn = os.path.join(storage_dir, 'positions.npy')
            os.remove(pos_fn)
    
            # Write new positions
            mmap = np.memmap(pos_fn, mode='w+', dtype='float32', shape=sim_pos.shape)
            mmap[:] = sim_pos.copy()
            mmap.flush()
            traj[0].save_pdb(os.path.join(storage_dir, 'topology.pdb'))
            
        # Reload positions
        self._load_positions_box_vecs() 
    
    
    def _reshape_list(self, unshaped_list: List):
        """
        Return a list of reshaped arrays. 
        
        Parameters:
        ------------
            unshaped_list (List):
                List of numpy arrays that need the 1st axis reshaped 
        """
        reshaped_array = []
        for i in range(len(self.storage_dirs)):
            reshaped_array.append(self._reshape_array(unshaped_arr=unshaped_list[i], state_arr=self.state_inds[i]))
        return reshaped_array      

    
    
    def _reshape_array(self, unshaped_arr: np.array, state_arr: np.array):
        """
        Return a reshaped array. 

        Parameters:
        ------------
            unshaped_arr (np.array):
                Numpy arrays that need the 1st axis reshaped.

            state_arr (np.array)
                Numpy array with the state indices for each replicate
        """        
        # Reshape 1st axis
        reshaped_arr = np.empty(unshaped_arr.shape)
        for state in range(unshaped_arr.shape[1]):
            for iter_num in range(unshaped_arr.shape[0]):
                reshaped_arr[iter_num, state, :] = unshaped_arr[iter_num, np.where(state_arr[iter_num] == state)[0], :]
        
        return reshaped_arr
    
    
    
    def _get_postions_map(self):
        """
        Dimensions of map:  Dimensions are (sim_no, iteration, state) >>> corresponding[sim_no, state, iter]. EXAMPLE: self.map[1,0] could return [0, 1, 2] which means the correct positions for the 1st interation at the 0th state of the reshaped energies matrix can be cound at the the 2nd iteration of the 1st replicate of the 0th simulation.
        """
        
        
        # Make maps
        self.map = []
        for sim_no, sim_state_inds in enumerate(self.state_inds):
    
            # Build map for simulation
            sim_map = np.empty((self.energies[sim_no].shape[0], self.energies[sim_no].shape[1], 3), dtype=int)
    
            # Iterate through simulation iters, states to build map
            for sim_iter in range(sim_map.shape[0]):
                for sim_state in range(sim_map.shape[1]):
                    sim_map[sim_iter, sim_state, :] = np.array([sim_no, sim_iter, np.where(sim_state_inds[sim_iter] == sim_state)[0][0]], dtype=int)
    
            # Add sim_map
            self.map.append(sim_map.astype(int))


    def _determine_interpolation_inds(self):
        """
        determine the indices (with respect to the last simulation) which are missing from other simulations
        """
        # Set interpolation attribute
        interpolation_list = self.temperatures_list
        final_set = self.temperatures
            
        # Iterate through temperatures
        self.interpolation_inds = []
        for i, set_i in enumerate(interpolation_list):
            missing_sim_inds = []
            for i, s in enumerate(final_set):
                if s not in set_i:
                    missing_sim_inds.append(i)
        
            # Assert that interpolation made sense
            assert len(missing_sim_inds) + len(set_i) == len(final_set), f'{len(missing_sim_inds)}, {len(set_i)}, {len(final_set)}'
            self.interpolation_inds.append(missing_sim_inds)



    def _backfill(self):
        """
        
        """
        # Determine interpolation inds
        self._determine_interpolation_inds()
        printf(f'Detected interpolations at: {self.interpolation_inds}')

        # Determine which simulations to resample from
        filled_sims = [True if not self.interpolation_inds[i] else False for i in range(len(self.interpolation_inds))]
        filled_sim_inds = [i for i in range(len(filled_sims)) if filled_sims[i] == True]

        #Make an interpolation map
        interpolation_map = [np.arange(self.temperatures.shape[0]) for i in range(len(self.temperatures))] # in shape (state, state) >>> state_ind
        for i, interpolation_ind_set in enumerate(self.interpolation_inds):
            for ind in interpolation_ind_set:
                interpolation_map[i] = interpolation_map[i][interpolation_map[i] != ind]

        # Iterate throught simulations to backfill energies
        backfilled_energies = []
        backfilled_map = []
        for sim_no, sim_interpolate_inds in enumerate(self.interpolation_inds):
            
            #Create an array for this simulations energies, in the final simulation's shape on axis 1, 2
            sim_energies = np.zeros((self.energies[sim_no].shape[0], self.temperatures.shape[0], self.temperatures.shape[0]))
            sim_map = np.zeros((self.map[sim_no].shape[0], self.temperatures.shape[0], 3))

            #Fill this array with the values that exist
            for i, ind in enumerate(interpolation_map[sim_no]):
                sim_energies[:, ind, interpolation_map[sim_no]] = self.energies[sim_no][:, i, :]
                sim_map[:,ind] = self.map[sim_no][:,i]

            #Fill in rows and columns 
            for state_no in sim_interpolate_inds:

                # Get state-specific objects to resample from
                filled_energies = np.concatenate([self.energies[sim_no] for sim_no in filled_sim_inds])
                filled_map = np.concatenate([self.map[sim_no] for sim_no in filled_sim_inds])
                
                state_energies = filled_energies[:, state_no]
                state_map = filled_map[:, state_no]

                N_k = np.array([state_energies.shape[0]])

                # Resample
                res_energies, res_mappings, res_inds = resample_with_MBAR(objs=[state_energies, state_map], u_kln=np.array([state_energies[:,state_no]]), N_k=N_k, reshape_weights=state_energies.shape[0], return_inds=True, size=len(sim_energies))


                # Assign resampled configurations to empty rows/cols
                sim_energies[:, state_no] = res_energies.copy()
                sim_map[:, state_no] = res_mappings.copy()

                sim_energies[:, :, state_no] = [filled_energies[resampled_ind, :, state_no] for resampled_ind in res_inds]


            backfilled_energies.append(sim_energies)
            backfilled_map.append(sim_map)

        # Concatenate
        self.energies = np.concatenate(backfilled_energies, axis=0)
        self.map = np.concatenate(backfilled_map, axis=0).astype(int)
        self.n_frames = self.energies.shape[0]
    
    
    # def _load_positions_box_vecs(self):
        
    #     # Load
    #     self.positions = []
    #     self.box_vectors = []
    #     for i, storage_dir in enumerate(self.storage_dirs):
            
    #         # Get positions
    #         try:
    #             pos_i = np.load(os.path.join(storage_dir, 'positions.npy'), mmap_mode='r')[self.skip:]
    #         except:
    #             try:
    #                 pos_i = np.memmap(os.path.join(storage_dir, 'positions.npy'), mode='r', dtype='float32', shape=(self.unshaped_energies[i].shape[0] + self.skip, self.unshaped_energies[i].shape[1], self.top.n_atoms, 3))[self.skip:]
    #             except:
    #                 raise Exception(f"Issue opening {os.path.join(storage_dir, 'positions.npy')} with dtype float32 and shape {(self.unshaped_energies[i].shape[0] + self.skip, self.unshaped_energies[i].shape[1], self.top.n_atoms, 3)}")
    #         assert pos_i.shape[0] > 0, f'{storage_dir} is invalid, please delete and resume'            
    #         self.positions.append(pos_i)
    
    #         # Get box vectors
    #         self.box_vectors.append(np.load(os.path.join(storage_dir, 'box_vectors.npy'), mmap_mode='r')[self.skip:]) 


    def _load_positions_box_vecs(self, skip=None):

        # Set skip
        if skip is None:
            skip = self.skip
        
        # Load
        self.positions = []
        self.box_vectors = []
        truncated = False
        for i, storage_dir in enumerate(self.storage_dirs):
            
            # Get positions
            try:
                pos_i = np.load(os.path.join(storage_dir, 'positions.npy'), mmap_mode='r')[skip:]
            except:
                try:
                    top_fn = os.path.join(storage_dir, 'topology.pdb')
                    if os.path.exists(top_fn):
                        truncated = True
                        top = md.load_pdb(top_fn).topology
                        self.top = deepcopy(top) # Make this the new topology for writing files
                        shape = (self.unshaped_energies[i].shape[0] + self.skip, self.unshaped_energies[i].shape[1], top.n_atoms, 3)
                    else:
                        top = md.load_pdb(self.pdb).topology
                        shape = (self.unshaped_energies[i].shape[0] + self.skip, self.unshaped_energies[i].shape[1], top.n_atoms, 3)
    
                    if truncated:
                        keep_inds = get_truncation_atom_keep_inds(top)
                        pos_i = np.memmap(os.path.join(storage_dir, 'positions.npy'), mode='r', dtype='float32', shape=shape)[skip:, :, keep_inds]
                    else:
                        pos_i = np.memmap(os.path.join(storage_dir, 'positions.npy'), mode='r', dtype='float32', shape=shape)[skip:]
                except:
                    raise Exception(f"Issue opening {os.path.join(storage_dir, 'positions.npy')} with dtype float32 and shape {shape}")
            assert pos_i.shape[0] > 0, f'{storage_dir} is invalid, please delete and resume'            
            self.positions.append(pos_i)
            printf(f'Loaded positions of {storage_dir} with shape {self.positions[i].shape}')

            # Get box vectors
            self.box_vectors.append(np.load(os.path.join(storage_dir, 'box_vectors.npy'), mmap_mode='r')[skip:]) 
            printf(f'Loaded box vectors of {storage_dir} with shape {self.box_vectors[i].shape}')


    def _remove_harmonic(self, spring_constant=83.65):
    
        printf('Removing harmonic restraint from energies...')
        # Equilibration if necessary
        if not hasattr(self, 't0'):
            self.equilibration_method = 'energy'
            self.determine_equilibration()
    
        # Load positions
        if not hasattr(self, 'positions'):
            self._load_positions_box_vecs()
        
        # Get selection for spring centers
        if self.resSeqs is not None:
            sele = self.top.select(f'protein and resSeq {" ".join([str(resSeq) for resSeq in self.resSeqs])}')
        else:
            raise Exception('Did not provide a selection via resSeqs :(')
    
        # Iterate through frames
        for frame in range(self.t0, self.energies.shape[0]):
        
            if frame % 10 == 0:
                print(datetime.now(), frame)
        
            # Iterate through primary states
            for state1 in range(self.energies.shape[1]):
        
                # Get frame, state positions
                sim_no, sim_iter, sim_rep_ind = self.map[frame, state1]
                pos = jnp.array([self.positions[sim_no][sim_iter][sim_rep_ind][sele]])
                unitcell_lengths = jnp.array([self.box_vectors[sim_no][sim_iter][sim_rep_ind].sum(axis=1)])
        
                # Get spring centers
                spring_centers = jnp.array([self.spring_centers[0,sele]]) # THIS ASSUMES THAT THE SPRING CENTERS PROVIDED ARE EQUAL IN ALL THERMODYNAMIC STATES
        
                # Get translation in case of wrapping issue
                trans, _ = best_translation_by_unitcell_jax(unitcell_lengths, pos, spring_centers)
        
                # Correct 
                self.energies[frame, state1, :] -= get_restraint_energy_kT(pos[0], trans[0], spring_centers[0], self.temperatures, spring_constant) 
    
    
        printf(f'Harmonic restraint removed from EQUILIBRATED frames of energies with shape {self.energies.shape}')



