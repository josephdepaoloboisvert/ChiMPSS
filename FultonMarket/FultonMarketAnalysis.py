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
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.utils.utils import TrackedQuantity
from pymbar import timeseries, MBAR
from pymbar.timeseries import detect_equilibration
import scipy.constants as cons
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from .FultonMarketUtils import *

jax.print_environment_info()
printf(f"Default JAX backend is {jax.default_backend()}")


class FultonMarketAnalysis():
    """
    Simulation analysis for replica exchange trajectories produced by FultonMarket.

    Loads energies, state indices, and (lazily) positions from the
    ``saved_variables`` sub-directories written by Randolph. Handles adaptive
    temperature ladders by backfilling energy matrices for sub-simulations
    that ran with fewer replicas than the final ladder, using MBAR importance
    resampling to estimate missing state energies.

    Parameters
    ----------
    input_dir : str
        Path to the FultonMarket output directory containing the
        ``saved_variables`` sub-directory.
    pdb : str
        Path to the reference PDB file used to construct the MDTraj topology.
    skip : int
        Number of frames to discard from the start of each sub-simulation
        when loading energies and state indices. Default 10.
    scheduling : str
        Temperature scheduling scheme identifier. Currently informational
        only. Default ``'Temperature'``.
    resSeqs : list of int, optional
        Residue sequence numbers (PDB numbering) to use for PCA and
        convergence analysis. If None, all protein atoms are used.
    sele_str : str, optional
        MDTraj atom-selection string for a ligand or other non-protein
        selection, applied when writing resampled trajectories.
    upper_limit : int, optional
        Maximum frame index (inclusive) to retain from the concatenated
        energy matrix. Useful for truncating analysis to a sub-interval.
    remove_harmonic : bool
        If True, subtract harmonic restraint energies from the loaded
        energies. Requires ``spring_centers``. Default False.
    spring_centers : np.ndarray, optional
        Spring centres required when ``remove_harmonic=True``.
    """

    def __init__(self,
                 input_dir: str,
                 pdb: str,
                 skip: int = 10,
                 scheduling: str = 'Temperature',
                 resSeqs: List[int] = None,
                 sele_str: str = None,
                 upper_limit: int = None,
                 remove_harmonic: bool = False,
                 spring_centers: np.ndarray = None):

        # Resolve input directory
        self.input_dir = input_dir.rstrip('/')
        self.stor_dir = os.path.join(self.input_dir, 'saved_variables')
        assert os.path.isdir(self.stor_dir), self.stor_dir
        printf(f'Found storage directory at {self.stor_dir}')
        self.storage_dirs = sorted(glob.glob(self.stor_dir + '/*'),
                                   key=lambda x: int(x.split('/')[-1]))
        # Topology
        self.pdb = pdb
        self.top = md.load_pdb(self.pdb).topology
        if resSeqs is not None:
            all_resSeqs = [self.top.residue(i).resSeq for i in range(self.top.n_residues)]
            self.resSeqs = [all_resSeqs.index(r) for r in resSeqs]
        else:
            self.resSeqs = None
        self.sele_str = sele_str
        printf(f'Ligand selection string: {sele_str}')
        # Load saved variables (memory-mapped)
        self.skip = skip
        self.scheduling = scheduling
        self.temperatures_list = [np.round(np.load(os.path.join(d, 'temperatures.npy'), mmap_mode='r'), decimals=2)
                                  for d in self.storage_dirs]
        self.temperatures = self.temperatures_list[-1]
        printf(f'Temperature array shapes: {[(i, t.shape) for i, t in enumerate(self.temperatures_list)]}')
        
        self.state_inds = [np.load(os.path.join(d, 'states.npy'), mmap_mode='r')[skip:] for d in self.storage_dirs]
        self.unshaped_energies = [np.load(os.path.join(d, 'energies.npy'), mmap_mode='r')[skip:]
                                  for d in self.storage_dirs]
        # Reshape energies from replica order to state order
        self.energies = self._reshape_list(self.unshaped_energies)
        
        # Build (sim_no, iteration, state) → (sim_no, sim_iter, rep_ind) map
        self._get_postions_map()
        
        # Backfill energies for sub-simulations with fewer replicas
        self._backfill()
        
        # Optionally restrict to a time window
        if upper_limit is not None:
            self.energies = self.energies[:upper_limit + 1]
            self.map = self.map[:upper_limit + 1]
            self.upper_limit = upper_limit
        printf(f'Final energy matrix shape: {self.energies.shape}')
    
    # ------------------------------------------------------------------
    # Energy access and plotting
    # ------------------------------------------------------------------
    def get_state_energies(self, state_index: int = 0) -> np.ndarray:
        """
        Return the time series of reduced energies for a single thermodynamic
        state evaluated at its own Hamiltonian.

        Extracts the diagonal element ``energies[:, state_index, state_index]``
        from the full (n_frames, n_states, n_states) energy matrix.

        Parameters
        ----------
        state_index : int
            Index into the temperature ladder. Default 0 (lowest temperature).

        Returns
        -------
        state_energies : np.ndarray of shape (n_frames,)
            Reduced energies of ``state_index`` over simulation time.
        """
        return self.energies[:, state_index, state_index]


    def get_average_energy(self) -> np.ndarray:
        """
        Compute the mean reduced energy across all thermodynamic states at
        each frame.

        Returns
        -------
        average_energies : np.ndarray of shape (n_frames,)
            Per-frame mean of the diagonal of the energy matrix.
        """
        n_frames, n_states = self.energies.shape[:2]
        state_energies = np.array([self.energies[:, state, state] for state in range(n_states)]).T
        return state_energies.mean(axis=1)


    def plot_average_energy(self, figsize: tuple = (6, 6), equilibration_method: str = 'energy'):
        """
        Plot the mean reduced energy over simulation time with the detected
        equilibration point marked.

        Calls ``determine_equilibration`` if it has not already been run.

        Parameters
        ----------
        figsize : tuple of (float, float)
            Matplotlib figure size in inches. Default (6, 6).
        equilibration_method : str
            Method passed to ``determine_equilibration`` if ``t0`` is not yet
            set. One of ``'energy'``, ``'PCA'``, or ``'None'``. Default
            ``'energy'``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        self.determine_equilibration(equilibration_method=equilibration_method)
        data = self.get_average_energy()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(data.shape[0]), data, color='k')
        ax.vlines(self.t0, data.min(), data.max(), color='r', label='equilibration')
        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_title('Mean energy')
        ax.set_ylabel('Energy (kJ/mol)')
        ax.set_xlabel('Iterations')
        fig.tight_layout()
        return fig, ax


    def plot_energy_distributions(self, figsize: tuple = (8, 4), post_equil: bool = False):
        """
        Plot kernel density estimates of the energy distribution for each
        thermodynamic state.

        Parameters
        ----------
        figsize : tuple of (float, float)
            Matplotlib figure size in inches. Default (8, 4).
        post_equil : bool
            If True, only frames after the detected equilibration point
            (``t0``) are included. Calls ``determine_equilibration`` if
            ``t0`` is not yet set. Default False.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        if post_equil:
            if not hasattr(self, 't0'):
                self.determine_equilibration()
            state_energies = np.array([self.get_state_energies(state_index=s)[self.t0:] for s in range(self.energies.shape[1])]).T
        else:
            state_energies = np.array([self.get_state_energies(state_index=s) for s in range(self.energies.shape[1])]).T
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(state_energies, ax=ax, legend=False)
        ax.set_xlabel('Energy (kJ/mol)')
        fig.tight_layout()
        return fig, ax


    # ------------------------------------------------------------------
    # Equilibration detection
    # ------------------------------------------------------------------
    def determine_equilibration(self, equilibration_method: str = 'energy', stride: int = 10):
        """
        Detect the equilibration point of the simulation and store it as
        ``self.t0``.

        If ``t0`` is already set, returns immediately without re-running.
        Supported methods:

        - ``'energy'`` — detects the equilibration frame from the average
          energy time series using ``detect_energy_equil``.
        - ``'PCA'`` — projects the lowest-temperature trajectory onto its
          principal components and finds the earliest frame at which each PC
          is equilibrated, weighted by explained variance.
        - ``'None'`` — sets ``t0 = 0`` (no equilibration discarded).

        Parameters
        ----------
        equilibration_method : str
            Detection method. Default ``'energy'``.
        stride : int
            Frame stride used when ``equilibration_method='PCA'``. Default 10.

        Sets
        ----
        self.t0 : int
            Frame index (relative to the full concatenated energy matrix) at
            which the simulation is considered equilibrated.
        """
        if hasattr(self, 't0'):
            return
        
        if equilibration_method == 'PCA':
            self.get_PCA(state_no=0, stride=stride, explained_variance_threshold=0.9)
            equil_times = np.array([detect_PC_equil(pc, self.reduced_cartesian) for pc in range(self.n_components)])
            self.t0 = int(np.sum(equil_times * (self.explained_variance / self.explained_variance.sum()))) * stride
        
        elif equilibration_method == 'energy':
            self.t0 = detect_energy_equil(self.get_average_energy())
        
        elif equilibration_method == 'None':
            self.t0 = 0
        
        else:
            raise ValueError(f"equilibration_method must be 'energy', 'PCA', or 'None', got '{equilibration_method}'")
        
        #printf(f'Equilibration detected at {np.round(self.t0 * self.energies.shape[1] / 1000, 3)} ns '
        #       f'(method: {equilibration_method})')
    
    # ------------------------------------------------------------------
    # Importance resampling and weights
    # ------------------------------------------------------------------
    def importance_resampling(self, n_samples: int = -1, equilibration_method: str = 'energy',
                              specify_state: int = 0, replace: bool = True):
        """
        Compute MBAR weights over the post-equilibration portion of the
        trajectory and draw ``n_samples`` importance-resampled frames.

        Calls ``determine_equilibration`` if ``t0`` is not yet set.

        Parameters
        ----------
        n_samples : int
            Number of frames to resample. -1 returns one sample per
            post-equilibration frame. Default -1.
        equilibration_method : str
            Passed to ``determine_equilibration`` if needed. Default
            ``'energy'``.
        specify_state : int
            Thermodynamic state index for which MBAR weights are computed.
            Default 0 (lowest temperature / unbiased ensemble).
        replace : bool
            Whether to sample with replacement. Default True.

        Sets
        ----
        self.flat_inds : np.ndarray of shape (n_post_equil_frames, 2)
            ``[state, frame]`` index pairs covering the post-equilibration
            window.
        self.weights : np.ndarray
            Unnormalised MBAR weights for each flat index.
        self.resampled_inds : np.ndarray of shape (n_samples, 2)
            ``[state, frame]`` pairs for the resampled frames.
        self.resampled_weights : np.ndarray
            Normalised MBAR weights corresponding to ``resampled_inds``.
        """
        if not hasattr(self, 't0'):
            self.determine_equilibration(equilibration_method=equilibration_method)

        self.flat_inds = np.array([[state, ind] for ind in range(self.t0, self.energies.shape[0]) for state in range(self.energies.shape[1])])
        u_kln = np.array([self.energies[self.t0:, :, k].flatten() for k in range(self.energies.shape[2])])
        N_k = np.full(self.energies.shape[2], self.energies[self.t0:].shape[0])

        self.resampled_inds, self.weights, self.resampled_weights = resample_with_MBAR(objs=[self.flat_inds],
                                                                                       u_kln=u_kln,
                                                                                       N_k=N_k,
                                                                                       size=n_samples,
                                                                                       return_inds=False,
                                                                                       return_weights=True,
                                                                                       return_resampled_weights=True,
                                                                                       specify_state=specify_state,
                                                                                       replace=replace)
        
    def reshape_weights(self):
        """
        Distribute the flat MBAR weights into a 2-D array indexed by
        (temperature state, frame).

        Requires ``importance_resampling`` to have been called first.

        Sets
        ----
        self.reshaped_weights : np.ndarray of shape (n_states, n_frames)
            Weight matrix suitable for imshow-style visualisation.
        """
        self.reshaped_weights = np.zeros((len(self.temperatures), len(self.energies)))
        for (state, frame), weight in zip(self.flat_inds, self.weights[:, 0]):
            self.reshaped_weights[state, frame] = weight


    def plot_weights(self, figsize: tuple = (25, 10), savefig: str = None):
        """
        Visualise the MBAR weight matrix as a heatmap.

        Calls ``reshape_weights`` internally. A vertical red line marks the
        equilibration cutoff.

        Parameters
        ----------
        figsize : tuple of (float, float)
            Matplotlib figure size in inches. Default (25, 10).
        savefig : str, optional
            File path to save the figure. If None, the figure is only
            displayed. Default None.
        """
        self.reshape_weights()

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.reshaped_weights[:, ::10])
        ax.vlines(self.flat_inds[:, 1].min() / 10, 0, self.temperatures.shape[0],
                  color='red', label='equilibration')
        ax.set_yticks(np.arange(self.temperatures.shape[0])[::20], self.temperatures[::20])
        ax.set_xlabel('Replicate simulation time (ns)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Truncated MBAR weights')
        plt.legend(loc='upper left')
        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)
        return fig, ax


    # ------------------------------------------------------------------
    # Trajectory access and writing
    # ------------------------------------------------------------------

    def write_resampled_traj(self,
                              pdb_out: str,
                              dcd_out: str,
                              weights_out: str = None,
                              inds_out: str = None,
                              return_traj: bool = False,
                              correction: bool = True,
                              alt_load_positions_name: str = None):
        """
        Assemble and write the importance-resampled trajectory to disk.

        Calls ``importance_resampling`` and ``_load_positions_box_vecs`` if
        they have not already been run. Uses ``self.map`` to look up the
        correct (sim_no, sim_iter, rep_ind) triple for each resampled frame.

        Parameters
        ----------
        pdb_out : str
            Path for the output single-frame PDB (topology reference).
        dcd_out : str
            Path for the output DCD trajectory.
        weights_out : str, optional
            If provided, the resampled MBAR weights are saved as a .npy file
            at this path.
        inds_out : str, optional
            If provided, the resampled frame indices are saved as a .npy file
            at this path.
        return_traj : bool
            If True, the assembled MDTraj trajectory is returned. Default
            False.
        correction : bool
            Passed to ``write_traj_from_pos_boxvecs`` for periodic boundary
            correction. Default True.
        alt_load_positions_name : str, optional
            Alternative filename (without directory) for the positions numpy
            array, passed to ``_load_positions_box_vecs``.

        Returns
        -------
        traj : md.Trajectory or None
            The assembled trajectory if ``return_traj=True``, else None.
        """
        if not hasattr(self, 'resampled_inds'):
            self.importance_resampling()

        if not hasattr(self, 'positions'):
            self._load_positions_box_vecs(positions_alt_name=alt_load_positions_name)

        n = len(self.resampled_inds)
        pos     = np.empty((n, self.positions[0].shape[2], 3))
        box_vec = np.empty((n, 3, 3))

        for i, (state, frame) in enumerate(self.resampled_inds):
            if i % max(1, n // 10) == 0:
                printf(f'{100 * i / n:.1f}% assembled')
            sim_no, sim_iter, sim_rep_ind = self.map[frame, state].astype(int)
            pos[i]     = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])

        self.traj = write_traj_from_pos_boxvecs(pos, box_vec, self.top, self.sele_str, correction=correction)
        self.traj[0].save_pdb(pdb_out)
        self.traj.save_dcd(dcd_out)
        printf(f'{self.traj.n_frames} frames written to {pdb_out} and {dcd_out}')

        if weights_out is not None:
            np.save(weights_out, self.resampled_weights)
            printf(f'MBAR weights written to {weights_out}')

        if inds_out is not None:
            np.save(inds_out, self.resampled_inds)
            printf(f'Resampled indices written to {inds_out}')

        if return_traj:
            return self.traj


    def state_trajectory(self, state_no: int = 0, stride: int = 1) -> 'md.Trajectory':
        """
        Extract the continuous trajectory of a single thermodynamic state.

        Uses ``self.map`` to resolve the correct replica frame at every
        iteration, correctly accounting for replica permutations.

        Parameters
        ----------
        state_no : int
            Thermodynamic state index to extract. Default 0 (lowest
            temperature).
        stride : int
            Frame stride. Default 1 (every frame).

        Returns
        -------
        traj : md.Trajectory
            MDTraj trajectory for the requested state.
        """
        if not (hasattr(self, 'positions') or hasattr(self, 'box_vectors')):
            self._load_positions_box_vecs()

        inds = np.arange(0, self.energies.shape[0], stride)
        pos     = np.empty((len(inds), self.positions[0].shape[2], 3))
        box_vec = np.empty((len(inds), 3, 3))

        for i, ind in enumerate(inds):
            sim_no, sim_iter, sim_rep_ind = self.map[ind, state_no].astype(int)
            pos[i]     = np.array(self.positions[sim_no][sim_iter][sim_rep_ind])
            box_vec[i] = np.array(self.box_vectors[sim_no][sim_iter][sim_rep_ind])

        return write_traj_from_pos_boxvecs(pos, box_vec, self.top, self.sele_str)


    # ------------------------------------------------------------------
    # PCA and reduced coordinates
    # ------------------------------------------------------------------

    def get_PCA(self,
                state_no: int = None,
                stride: int = 1,
                explained_variance_threshold: float = 0.9):
        """
        Project the trajectory onto its principal components and retain
        enough components to explain ``explained_variance_threshold`` of the
        total variance.

        If ``state_no`` is None, uses ``self.traj`` (the resampled trajectory
        must therefore already exist). Otherwise calls ``state_trajectory`` to
        build the trajectory on-the-fly.

        Parameters
        ----------
        state_no : int, optional
            Thermodynamic state index to analyse. If None, ``self.traj`` is
            used. Default None.
        stride : int
            Frame stride passed to ``state_trajectory``. Default 1.
        explained_variance_threshold : float
            Fraction of total variance that must be explained by the retained
            components. Default 0.9.

        Sets
        ----
        self.reduced_cartesian : np.ndarray of shape (n_frames, n_components)
            PCA-projected coordinates.
        self.explained_variance : np.ndarray of shape (n_components,)
            Explained variance ratio for each retained PC.
        self.n_components : int
            Number of retained principal components.
        """
        if state_no is None:
            traj = deepcopy(self.traj)
        else:
            traj = self.state_trajectory(state_no, stride)
            if hasattr(self, 'upper_limit'):
                traj = traj[:self.upper_limit + 1]

        if self.resSeqs is not None:
            sele = traj.topology.select(
                f'resSeq {" ".join(str(r) for r in self.resSeqs)}'
            )
        else:
            sele = traj.topology.select('protein')
        traj = traj.atom_slice(sele)

        pca, self.reduced_cartesian, self.explained_variance, self.n_components = get_traj_PCA(
            traj, explained_variance_threshold=explained_variance_threshold
        )
        printf(f'Reduced Cartesian shape: {self.reduced_cartesian.shape}')


    def get_weighted_reduced_cartesian(self,
                                        rc_upper_limit: int = None,
                                        return_weighted_rc: bool = False,
                                        use_state: bool = False,
                                        stride: int = 1):
        """
        Compute the MBAR-weighted mean reduced Cartesian coordinate and its
        uncertainty.

        The reduced Cartesian is the variance-weighted Euclidean norm across
        all retained PCs, providing a scalar summary of the conformational
        ensemble centroid. Used to track convergence across sub-simulations.

        Parameters
        ----------
        rc_upper_limit : int, optional
            Upper frame index (exclusive) used to restrict the resampled
            indices considered. If None, all frames are used.
        return_weighted_rc : bool
            If True, the mean and error are returned as well as stored.
            Default False.
        use_state : bool
            If True, use uniform weights from the lowest-temperature state
            trajectory rather than MBAR weights. Useful as a diagnostic.
            Default False.
        stride : int
            Frame stride applied when ``use_state=True``. Default 1.

        Returns
        -------
        mean_weighted_reduced_cartesian : float
            Only returned when ``return_weighted_rc=True``.
        mean_weighted_reduced_cartesian_err : float
            Only returned when ``return_weighted_rc=True``.

        Sets
        ----
        self.mean_weighted_reduced_cartesian : float
        self.mean_weighted_reduced_cartesian_err : float
        """
        if rc_upper_limit is None:
            rc_upper_limit = np.inf

        if use_state:
            state_flat_inds = np.array([[0, ind] for ind in range(self.energies.shape[0])])[::stride]
            state_weights = np.ones(state_flat_inds.shape[0])
            self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err = (
                calculate_weighted_rc(self.reduced_cartesian, state_flat_inds, rc_upper_limit,
                                      self.explained_variance, state_weights)
            )
        else:
            self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err = (
                calculate_weighted_rc(self.reduced_cartesian, self.resampled_inds, rc_upper_limit,
                                      self.explained_variance, self.resampled_weights)
            )

        if return_weighted_rc:
            return self.mean_weighted_reduced_cartesian, self.mean_weighted_reduced_cartesian_err


    # ------------------------------------------------------------------
    # Trajectory truncation
    # ------------------------------------------------------------------

    def truncate(self):
        """
        Remove unused heavy atoms from saved position arrays to reduce disk
        usage.

        Iterates through all but the last two sub-simulation directories and
        rewrites the ``positions.npy`` file using a memory-mapped float32
        array containing only the atoms selected by
        ``get_truncation_atom_keep_inds``. A ``topology.pdb`` file is written
        alongside each truncated positions file so that subsequent loads can
        reconstruct the correct atom count.

        Sub-simulation directories that already contain a ``topology.pdb``
        are skipped, making this method safe to re-run after interruption.

        Sets
        ----
        self.top : md.Topology
            Updated to the truncated topology after processing.
        """
        self._load_positions_box_vecs(skip=0)

        keep_inds = get_truncation_atom_keep_inds(self.top)
        traj = md.load_pdb(self.pdb).atom_slice(keep_inds)
        self.top = deepcopy(traj.topology)

        for i, storage_dir in enumerate(self.storage_dirs[:-2]):
            top_fn = os.path.join(storage_dir, 'topology.pdb')
            if os.path.exists(top_fn):
                continue

            init_shape = self.positions[i].shape
            sim_pos = np.array(self.positions[i][:, :, keep_inds, :])
            printf(f'Truncated {storage_dir}: {init_shape} → {sim_pos.shape}')

            pos_fn = os.path.join(storage_dir, 'positions.npy')
            os.remove(pos_fn)
            mmap = np.memmap(pos_fn, mode='w+', dtype='float32', shape=sim_pos.shape)
            mmap[:] = sim_pos.copy()
            mmap.flush()
            traj[0].save_pdb(top_fn)

        self._load_positions_box_vecs()


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reshape_list(self, unshaped_list: List) -> List:
        """
        Reshape each energy array in ``unshaped_list`` from replica order to
        state order.

        Parameters
        ----------
        unshaped_list : list of np.ndarray
            One array per sub-simulation, each of shape
            (n_frames, n_replicates, n_states).

        Returns
        -------
        reshaped : list of np.ndarray
            Same structure as input, with axis-1 reordered by thermodynamic
            state index.
        """
        return [
            self._reshape_array(unshaped_list[i], self.state_inds[i])
            for i in range(len(self.storage_dirs))
        ]


    def _reshape_array(self, unshaped_arr: np.ndarray, state_arr: np.ndarray) -> np.ndarray:
        """
        Reorder a single energy array so that axis-1 corresponds to
        thermodynamic state index rather than replica index.

        At each frame, the replica that occupied state ``s`` is identified via
        ``state_arr`` and its energies are placed in position ``s`` along
        axis-1.

        Parameters
        ----------
        unshaped_arr : np.ndarray of shape (n_frames, n_replicates, n_states)
            Energy array in replica order.
        state_arr : np.ndarray of shape (n_frames, n_replicates)
            State index occupied by each replica at each frame.

        Returns
        -------
        reshaped_arr : np.ndarray of shape (n_frames, n_states, n_states)
            Energy array reindexed to state order.
        """
        reshaped_arr = np.empty(unshaped_arr.shape)
        for state in range(unshaped_arr.shape[1]):
            for iter_num in range(unshaped_arr.shape[0]):
                reshaped_arr[iter_num, state, :] = unshaped_arr[
                    iter_num, np.where(state_arr[iter_num] == state)[0], :
                ]
        return reshaped_arr


    def _get_postions_map(self):
        """
        Build a lookup map from (frame, state) to (sim_no, sim_iter, rep_ind).

        For every frame and thermodynamic state in each sub-simulation,
        records which sub-simulation, which iteration within that
        sub-simulation, and which replica index holds the configuration for
        that state. This map is used by ``write_resampled_traj`` and
        ``state_trajectory`` to retrieve positions without loading all data
        into memory.

        Sets
        ----
        self.map : list of np.ndarray
            One integer array per sub-simulation, each of shape
            (n_frames, n_states, 3). The last axis encodes
            ``[sim_no, sim_iter, rep_ind]``.
        """
        self.map = []
        for sim_no, sim_state_inds in enumerate(self.state_inds):
            n_iters, n_states = self.energies[sim_no].shape[:2]
            sim_map = np.empty((n_iters, n_states, 3), dtype=int)
            for sim_iter in range(n_iters):
                for sim_state in range(n_states):
                    rep_ind = np.where(sim_state_inds[sim_iter] == sim_state)[0][0]
                    sim_map[sim_iter, sim_state, :] = [sim_no, sim_iter, rep_ind]
            self.map.append(sim_map)


    def _determine_interpolation_inds(self):
        """
        Identify which thermodynamic state indices are missing from each
        sub-simulation's temperature ladder.

        Compares each sub-simulation's temperature array against the final
        (largest) ladder. Any temperature present in the final ladder but
        absent from a given sub-simulation is recorded as a missing index.

        Sets
        ----
        self.interpolation_inds : list of list of int
            One list per sub-simulation. Each inner list contains the state
            indices (in the final ladder) that were not sampled during that
            sub-simulation.
        """
        final_set = self.temperatures
        self.interpolation_inds = []

        for set_i in self.temperatures_list:
            missing = [i for i, t in enumerate(final_set) if t not in set_i]
            assert len(missing) + len(set_i) == len(final_set), (
                f'Interpolation index mismatch: {len(missing)} missing + '
                f'{len(set_i)} present ≠ {len(final_set)} final'
            )
            self.interpolation_inds.append(missing)


    def _backfill(self):
        """
        Expand all sub-simulation energy matrices to the shape of the final
        temperature ladder by importance-resampling missing state energies.

        Sub-simulations that sampled the full final ladder are used as the
        donor pool for MBAR resampling. For each missing state in an
        incomplete sub-simulation, energies and map entries are drawn from
        the donor pool weighted by the MBAR weights of that state. This
        produces a consistent (n_total_frames, n_final_states, n_final_states)
        energy matrix that can be passed directly to MBAR.

        Sets
        ----
        self.energies : np.ndarray of shape (n_total_frames, n_states, n_states)
            Concatenated, backfilled energy matrix.
        self.map : np.ndarray of shape (n_total_frames, n_states, 3)
            Concatenated position lookup map aligned with ``self.energies``.
        self.n_frames : int
            Total number of frames in the concatenated matrix.
        """
        self._determine_interpolation_inds()
        printf(f'Detected interpolation indices: {self.interpolation_inds}')

        # Identify sub-simulations that sampled the complete ladder
        filled_sim_inds = [
            i for i, inds in enumerate(self.interpolation_inds) if not inds
        ]

        # Build per-sub-simulation index map (final-state index → sub-sim state index)
        interpolation_map = [
            np.arange(self.temperatures.shape[0])
            for _ in range(len(self.temperatures_list))
        ]
        for i, missing_inds in enumerate(self.interpolation_inds):
            for ind in missing_inds:
                interpolation_map[i] = interpolation_map[i][interpolation_map[i] != ind]

        backfilled_energies = []
        backfilled_map = []

        for sim_no, sim_interpolate_inds in enumerate(self.interpolation_inds):
            n_final = self.temperatures.shape[0]
            sim_energies = np.zeros((self.energies[sim_no].shape[0], n_final, n_final))
            sim_map      = np.zeros((self.map[sim_no].shape[0], n_final, 3))

            # Copy energies that exist in this sub-simulation
            for i, ind in enumerate(interpolation_map[sim_no]):
                sim_energies[:, ind, interpolation_map[sim_no]] = self.energies[sim_no][:, i, :]
                sim_map[:, ind] = self.map[sim_no][:, i]

            # Resample missing states from the donor pool
            for state_no in sim_interpolate_inds:
                filled_energies = np.concatenate([self.energies[s] for s in filled_sim_inds])
                filled_map      = np.concatenate([self.map[s]      for s in filled_sim_inds])

                state_energies = filled_energies[:, state_no]
                state_map      = filled_map[:, state_no]
                N_k = np.array([state_energies.shape[0]])

                res_energies, res_mappings, res_inds = resample_with_MBAR(
                    objs=[state_energies, state_map],
                    u_kln=np.array([state_energies[:, state_no]]),
                    N_k=N_k,
                    reshape_weights=state_energies.shape[0],
                    return_inds=True,
                    size=len(sim_energies),
                )

                sim_energies[:, state_no]    = res_energies.copy()
                sim_map[:, state_no]         = res_mappings.copy()
                sim_energies[:, :, state_no] = [
                    filled_energies[idx, :, state_no] for idx in res_inds
                ]

            backfilled_energies.append(sim_energies)
            backfilled_map.append(sim_map)

        self.energies  = np.concatenate(backfilled_energies, axis=0)
        self.map       = np.concatenate(backfilled_map, axis=0).astype(int)
        self.n_frames  = self.energies.shape[0]


    def _load_positions_box_vecs(self, positions_alt_name: str = None, skip: int = None):
        """
        Load position and box-vector arrays for all sub-simulations into
        memory-mapped numpy arrays.

        Attempts a direct ``np.load`` first. If that fails (e.g. for large
        files stored as float32 memmaps), falls back to ``np.memmap`` with a
        shape inferred from the energy arrays. If a ``topology.pdb`` file
        exists in the sub-simulation directory, the positions are sliced to
        include only the truncated atom set.

        Parameters
        ----------
        positions_alt_name : str, optional
            Alternative filename for the positions array (without directory
            path). Useful when positions have been saved under a non-default
            name. Default None (uses ``'positions.npy'``).
        skip : int, optional
            Number of leading frames to discard. If None, ``self.skip`` is
            used.

        Sets
        ----
        self.positions : list of np.ndarray
            One memory-mapped array per sub-simulation, each of shape
            (n_frames, n_replicates, n_atoms, 3).
        self.box_vectors : list of np.ndarray
            One memory-mapped array per sub-simulation, each of shape
            (n_frames, n_replicates, 3, 3).
        self.top : md.Topology
            Updated to the truncated topology if truncation has been applied.
        """
        if skip is None:
            skip = self.skip

        positions_fn = positions_alt_name if positions_alt_name is not None else 'positions.npy'
        truncated = False

        self.positions   = []
        self.box_vectors = []

        for i, storage_dir in enumerate(self.storage_dirs):
            pos_path = os.path.join(storage_dir, positions_fn)

            try:
                pos_i = np.load(pos_path, mmap_mode='r')[skip:]

            except Exception:
                top_fn = os.path.join(storage_dir, 'topology.pdb')
                if os.path.exists(top_fn):
                    truncated = True
                    top = md.load_pdb(top_fn).topology
                    self.top = deepcopy(top)
                else:
                    top = md.load_pdb(self.pdb).topology

                shape = (
                    self.unshaped_energies[i].shape[0] + self.skip,
                    self.unshaped_energies[i].shape[1],
                    top.n_atoms,
                    3,
                )
                try:
                    raw = np.memmap(pos_path, mode='r', dtype='float32', shape=shape)[skip:]
                    if truncated:
                        keep_inds = get_truncation_atom_keep_inds(top)
                        pos_i = raw[:, :, keep_inds]
                    else:
                        pos_i = raw
                except Exception as exc:
                    raise RuntimeError(
                        f'Cannot open {pos_path} as float32 with shape {shape}'
                    ) from exc

            assert pos_i.shape[0] > 0, (
                f'{storage_dir} has no frames — delete the directory and resume.'
            )
            self.positions.append(pos_i)
            printf(f'Loaded positions from {storage_dir}: shape {pos_i.shape}')

            bv_path = os.path.join(storage_dir, 'box_vectors.npy')
            bv_i = np.load(bv_path, mmap_mode='r')[skip:]
            self.box_vectors.append(bv_i)
            printf(f'Loaded box vectors from {storage_dir}: shape {bv_i.shape}')



    # ==================================================================
    # Retro-analysis methods
    # ==================================================================

    def retro_analyze_all(
        self,
        n_resample: int = 100,
        sim_nos: List[int] = None,
        overwrite: bool = False,
        read_only: bool = False,
        output_cache_dir: str = None,
        getcontacts_script: str = None,
        conda_env: str = None,
        getcontacts_python: str = None,
    ) -> Dict[int, dict]:
        """
        Retroactively compute and save resampled distance matrices for all
        (or a specified subset of) sub-simulations.

        A single FultonMarketAnalysis object (``self``) is reused across all
        sub-simulations. For each ``sim_no``, the energy matrix and position
        map are temporarily restricted to frames up to and including that
        checkpoint, importance resampling is performed, the resampled
        trajectory is written, and the three distance matrices (torsional,
        alpha-carbon, contact) are computed and saved.

        Three operating modes are supported via ``read_only`` and
        ``output_cache_dir`` — see ``retro_convergence_utils`` for details.

        Parameters
        ----------
        n_resample : int
            Number of frames to importance-resample per sub-simulation.
            Default 100.
        sim_nos : list of int, optional
            Specific sub-simulation indices to process. If None, all
            discovered sub-simulations are processed.
        overwrite : bool
            If True, recompute matrices even if they already exist on disk.
            Default False.
        read_only : bool
            If True, compute everything in memory — nothing written to disk.
            Default False.
        output_cache_dir : str, optional
            Writable shadow directory for derived files. Reads check here
            first; all writes go here. Created automatically if absent.
        getcontacts_script : str, optional
            Path to ``get_dynamic_contacts.py``.
        conda_env : str, optional
            Conda environment name for running GetContacts.
        getcontacts_python : str, optional
            Full path to the Python interpreter for running GetContacts.

        Returns
        -------
        all_matrices : dict of int -> dict of str -> np.ndarray
            Nested dict keyed by ``sim_no`` then matrix name
            (``'torsion'``, ``'alpha_carbon'``, ``'contact'``).
        """
        from .retro_convergence_utils import (
            resolve_write_dir, resolve_cache_dir, resolve_traj_paths,
            load_matrices, save_matrices,
            compute_distance_matrices, log_mode,
        )
        import tempfile

        available = sorted(int(d.split('/')[-1]) for d in self.storage_dirs)
        targets   = sim_nos if sim_nos is not None else available
        skipped   = set(targets) - set(available)
        if skipped:
            printf(f'WARNING: sim_nos not found, skipping: {sorted(skipped)}')
        targets = [t for t in targets if t in available]

        log_mode(read_only, output_cache_dir)
        printf(f'retro_analyze_all: processing {len(targets)} sub-simulations: {targets}')

        # Snapshot of full energies/map — temporarily sliced per sim_no
        full_energies = self.energies
        full_map      = self.map
        frame_counts  = [e.shape[0] for e in self.unshaped_energies]

        all_matrices = {}

        for sim_no in targets:
            try:
                src_sim_dir   = self.storage_dirs[sim_no]
                write_sim_dir = resolve_write_dir(src_sim_dir, output_cache_dir, sim_no, read_only)

                # Check cache/original for existing matrices
                cache_sim_dir = resolve_cache_dir(output_cache_dir, sim_no)
                existing = load_matrices(src_sim_dir, cache_sim_dir)
                if len(existing) == 3 and not overwrite:
                    printf(f'sim_no={sim_no}: all matrices present, skipping')
                    all_matrices[sim_no] = existing
                    continue

                printf(f'sim_no={sim_no}: computing matrices')

                # Restrict energy matrix and map to frames up to this sim_no
                upper = int(np.sum(frame_counts[:sim_no + 1])) - 1
                self.energies = full_energies[:upper + 1]
                self.map      = full_map[:upper + 1]

                # Clear any cached resampling state
                for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                             'resampled_weights', 'positions', 'box_vectors', 'traj'):
                    if hasattr(self, attr):
                        delattr(self, attr)

                self.importance_resampling(n_samples=n_resample)

                # Resolve trajectory output paths
                if read_only:
                    pdb_out      = '/dev/null'
                    dcd_out      = '/dev/null'
                    weights_out  = None
                    inds_out     = None
                    contacts_tsv = os.path.join(tempfile.mkdtemp(), 'contacts.tsv')
                else:
                    pdb_out, dcd_out, weights_out, inds_out = resolve_traj_paths(
                        src_sim_dir, write_sim_dir
                    )
                    contacts_tsv = os.path.join(
                        write_sim_dir or src_sim_dir, 'resampled_contacts.tsv'
                    )

                traj = self.write_resampled_traj(
                    pdb_out=pdb_out,
                    dcd_out=dcd_out,
                    weights_out=weights_out,
                    inds_out=inds_out,
                    return_traj=True,
                )

                matrices = compute_distance_matrices(
                    traj=traj,
                    pdb_out=pdb_out,
                    dcd_out=dcd_out,
                    contacts_tsv=contacts_tsv,
                    getcontacts_script=getcontacts_script,
                    conda_env=conda_env,
                    getcontacts_python=getcontacts_python,
                )

                if not read_only:
                    save_matrices(matrices, write_sim_dir or src_sim_dir, sim_no)
                else:
                    printf(f'sim_no={sim_no}: read_only=True — matrices in memory only')

                all_matrices[sim_no] = matrices

            except Exception as exc:
                printf(f'ERROR: sim_no={sim_no} failed: {exc}')
                import traceback; traceback.print_exc()

        # Restore full state
        self.energies = full_energies
        self.map      = full_map
        for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                     'resampled_weights', 'positions', 'box_vectors', 'traj'):
            if hasattr(self, attr):
                delattr(self, attr)

        printf(f'retro_analyze_all complete: {len(all_matrices)} / {len(targets)} processed.')
        return all_matrices


    def retro_convergence_report(
        self,
        max_equil_fraction: float = 0.75,
        minimum_fraction: float = 0.25,
        frobenius_thresh: float = 0.05,
        jsd_thresh: float = 0.05,
        n_resample: int = 100,
        sim_nos: List[int] = None,
        read_only: bool = False,
        output_cache_dir: str = None,
        getcontacts_script: str = None,
        conda_env: str = None,
        getcontacts_python: str = None,
    ) -> Tuple[Dict[int, Dict[str, bool]], Dict[int, dict]]:
        """
        Evaluate convergence checks as a function of simulation progress,
        expressed entirely in percentages of the total number of completed
        sub-simulations.

        All sub-simulations are assumed to be of equal length, so
        ``len(self.storage_dirs)`` is used as ``total_n_sims`` — no
        ``total_sim_time`` or ``sim_length`` arguments are required.

        For each ``sim_no``, the energy matrix is sliced to the appropriate
        frame window, equilibration is detected, the effective
        post-equilibration comparison window is computed, and Frobenius norms
        and JSD values are computed against all checkpoints within that window.
        Results are printed per-checkpoint as percentages and as a summary
        table.

        Parameters
        ----------
        max_equil_fraction : float
            Maximum fraction of simulation that may be discarded to
            equilibration. Enforces a minimum post-equilibration comparison
            window of ``1 - max_equil_fraction``. Default 0.75.
        minimum_fraction : float
            Minimum fraction of total sub-simulations that must be complete
            before convergence can be declared. Default 0.25.
        frobenius_thresh : float
            Frobenius norm threshold for convergence. Default 0.05.
        jsd_thresh : float
            JSD threshold for convergence. Default 0.05.
        n_resample : int
            Frames to resample when matrices must be computed on-the-fly.
            Default 100.
        sim_nos : list of int, optional
            Subset of sub-simulation indices to evaluate. If None, all
            available are evaluated.
        read_only : bool
            If True, compute everything in memory — nothing written. Default
            False.
        output_cache_dir : str, optional
            Writable shadow directory. Reads check here first; writes go here.
        getcontacts_script : str, optional
            Path to ``get_dynamic_contacts.py``.
        conda_env : str, optional
            Conda environment for GetContacts.
        getcontacts_python : str, optional
            Python interpreter for GetContacts.

        Returns
        -------
        report : dict of int -> dict of str -> bool
            Per-sim_no check results. All display values are percentages.
            Final key ``'STOP'`` is True when all checks pass.
        metrics : dict of int -> dict
            Per-sim_no convergence metadata for downstream analysis/graphing.
            Each entry contains:
              - ``equil_fraction`` : float — fraction of frames discarded as
                equilibration at this checkpoint.
              - ``first_valid_sim_no`` : int — first sim_no included in the
                post-equilibration comparison window.
              - ``effective_post_equil`` : float — fraction of simulation
                covered by the post-equil comparison window.
              - ``comparison_sim_nos`` : list of int — previous sim_nos
                compared against (x-axis for graphing).
              - ``frobenius`` : dict of str -> list of float — normalised
                Frobenius norms vs each comparison checkpoint, keyed by matrix
                type (``'torsion'``, ``'alpha_carbon'``, ``'contact'``).
                List order matches ``comparison_sim_nos``.
              - ``jsd`` : dict of str -> list of float — JSD values vs each
                comparison checkpoint, same structure as ``frobenius``.
        """
        from .retro_convergence_utils import (
            resolve_write_dir, resolve_cache_dir, resolve_traj_paths,
            load_matrices, save_matrices,
            compute_distance_matrices,
            evaluate_matrix_convergence, build_checks,
            print_sim_report, print_summary_table, log_mode,
        )
        import tempfile

        # total_n_sims derived entirely from what actually exists on disk —
        # no time-based parameters needed
        total_n_sims = len(self.storage_dirs)
        available    = sorted(int(d.split('/')[-1]) for d in self.storage_dirs)
        targets      = sim_nos if sim_nos is not None else available
        targets      = [t for t in targets if t in available]

        log_mode(read_only, output_cache_dir)
        printf(f'retro_convergence_report: {len(targets)} checkpoints, '
               f'total_n_sims={total_n_sims} '
               f'({100.0 * len(targets) / total_n_sims:.1f}% of simulation covered), '
               f'frobenius_thresh={frobenius_thresh}, jsd_thresh={jsd_thresh}')

        # Snapshot full state — we slice energies/map per sim_no
        full_energies = self.energies
        full_map      = self.map
        frame_counts  = [e.shape[0] for e in self.unshaped_energies]

        # In-process matrix cache: avoids re-computing previous checkpoints.
        # resolve_cache_dir is used here (not resolve_write_dir) so we always
        # check the cache dir for reads, regardless of the read_only flag.
        matrix_cache: Dict[int, dict] = {}
        for sim_no in targets:
            src_sim_dir   = self.storage_dirs[sim_no]
            cache_sim_dir = resolve_cache_dir(output_cache_dir, sim_no)
            existing      = load_matrices(src_sim_dir, cache_sim_dir)
            if existing:
                matrix_cache[sim_no] = existing
                printf(f'sim_no={sim_no} ({100.0*(sim_no+1)/total_n_sims:.1f}%): '
                       f'pre-loaded {len(existing)} cached matrices from disk')

        report  = {}
        metrics = {}

        for sim_no in targets:
            progress_pct = 100.0 * (sim_no + 1) / total_n_sims
            printf(f'\n{"=" * 60}')
            printf(f'Evaluating checkpoint {progress_pct:.1f}% (sim_no={sim_no})')

            try:
                src_sim_dir   = self.storage_dirs[sim_no]
                write_sim_dir = resolve_write_dir(src_sim_dir, output_cache_dir, sim_no, read_only)

                # Restrict energy matrix to frames through this sim_no
                upper = int(np.sum(frame_counts[:sim_no + 1])) - 1
                self.energies = full_energies[:upper + 1]
                self.map      = full_map[:upper + 1]

                # Clear cached resampling state from previous iteration
                for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                             'resampled_weights', 'positions', 'box_vectors', 'traj'):
                    if hasattr(self, attr):
                        delattr(self, attr)

                # Equilibration detection on the sliced energy series
                self.determine_equilibration(equilibration_method='energy')
                equil_fraction = self.t0 / self.energies.shape[0]

                # Effective post-equilibration window — expressed as fraction
                # of sub-simulations, not of time
                min_post_equil       = 1.0 - max_equil_fraction
                effective_post_equil = max(1.0 - equil_fraction, min_post_equil)
                first_valid_sim_no   = int((1.0 - effective_post_equil) * sim_no)

                # Compute matrices if not already cached
                if sim_no not in matrix_cache:
                    printf(f'  computing distance matrices on-the-fly')

                    self.importance_resampling(n_samples=n_resample)

                    if read_only:
                        pdb_out      = '/dev/null'
                        dcd_out      = '/dev/null'
                        weights_out  = None
                        inds_out     = None
                        contacts_tsv = os.path.join(tempfile.mkdtemp(), 'contacts.tsv')
                    else:
                        pdb_out, dcd_out, weights_out, inds_out = resolve_traj_paths(
                            src_sim_dir, write_sim_dir
                        )
                        contacts_tsv = os.path.join(
                            write_sim_dir or src_sim_dir, 'resampled_contacts.tsv'
                        )

                    traj = self.write_resampled_traj(
                        pdb_out=pdb_out,
                        dcd_out=dcd_out,
                        weights_out=weights_out,
                        inds_out=inds_out,
                        return_traj=True,
                    )

                    matrices = compute_distance_matrices(
                        traj=traj,
                        pdb_out=pdb_out,
                        dcd_out=dcd_out,
                        contacts_tsv=contacts_tsv,
                        getcontacts_script=getcontacts_script,
                        conda_env=conda_env,
                        getcontacts_python=getcontacts_python,
                    )

                    if not read_only:
                        save_matrices(matrices, write_sim_dir or src_sim_dir, sim_no)

                    matrix_cache[sim_no] = matrices

                current_matrices = matrix_cache[sim_no]

                frob_results, jsd_results = evaluate_matrix_convergence(
                    current_matrices=current_matrices,
                    matrix_cache=matrix_cache,
                    first_valid_sim_no=first_valid_sim_no,
                    current_sim_no=sim_no,
                    frobenius_thresh=frobenius_thresh,
                    jsd_thresh=jsd_thresh,
                )

                checks = build_checks(
                    sim_no=sim_no,
                    total_n_sims=total_n_sims,
                    minimum_fraction=minimum_fraction,
                    equil_fraction=equil_fraction,
                    max_equil_fraction=max_equil_fraction,
                    frob_results=frob_results,
                    jsd_results=jsd_results,
                    frobenius_thresh=frobenius_thresh,
                    jsd_thresh=jsd_thresh,
                )

                print_sim_report(
                    sim_no=sim_no,
                    total_n_sims=total_n_sims,
                    checks=checks,
                    frob_results=frob_results,
                    jsd_results=jsd_results,
                    frobenius_thresh=frobenius_thresh,
                    jsd_thresh=jsd_thresh,
                    equil_fraction=equil_fraction,
                    effective_post_equil=effective_post_equil,
                    first_valid_sim_no=first_valid_sim_no,
                )

                report[sim_no] = checks

                # Build per-sim_no metrics for downstream graphing.
                # frob_results / jsd_results are Dict[str, Dict[int, float]]
                # keyed by matrix name then by previous sim_no.
                comparison_sim_nos = sorted(
                    set().union(*(d.keys() for d in frob_results.values()))
                )
                metrics[sim_no] = {
                    'equil_fraction':      equil_fraction,
                    'first_valid_sim_no':  first_valid_sim_no,
                    'effective_post_equil': effective_post_equil,
                    'comparison_sim_nos':  comparison_sim_nos,
                    'frobenius': {
                        name: [frob_results[name].get(s) for s in comparison_sim_nos]
                        for name in ('torsion', 'alpha_carbon', 'contact')
                    },
                    'jsd': {
                        name: [jsd_results[name].get(s) for s in comparison_sim_nos]
                        for name in ('torsion', 'alpha_carbon', 'contact')
                    },
                }

            except Exception as exc:
                printf(f'ERROR: sim_no={sim_no} failed: {exc}')
                import traceback; traceback.print_exc()

        # Restore full state
        self.energies = full_energies
        self.map      = full_map
        for attr in ('t0', 'flat_inds', 'weights', 'resampled_inds',
                     'resampled_weights', 'positions', 'box_vectors', 'traj'):
            if hasattr(self, attr):
                delattr(self, attr)

        print_summary_table(report, total_n_sims)
        return report, metrics