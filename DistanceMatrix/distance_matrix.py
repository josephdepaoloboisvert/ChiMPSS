# Package Imports
import os
import mdtraj as md
import numpy as np
from typing import List
#Custom Code Imports
from .DistanceMatrixUtils import _compute_component_matrices, _compute_matrix, printf


class DistanceMatrix():
    """
    Frame-by-frame pairwise structural distance matrix for an MD trajectory.

    Combines torsion, alpha-carbon, and hydrogen-bond component matrices
    with user-specified weights into a single composite distance matrix.

    Parameters
    ----------
    super_traj : md.Trajectory
        Trajectory with all sub-simulation data concatenated.
    ca_resSeqs : list of int, optional
        Residue sequence numbers to include in alpha-carbon distance
        calculations. If None, all residues are used.
    """

    def __init__(self, super_traj: md.Trajectory, ca_resSeqs: List[int]=None):
        """
        Initialize a DistanceMatrix from an aggregated trajectory.

        See class docstring for parameter descriptions.
        """

        # Attributes
        self.super_traj = super_traj
        self.ca_resSeqs = ca_resSeqs


    def compute_component_matrices(self, load_matrices: str=None, stride: int=1, save_matrices: str=os.getcwd()):
        """
        Compute torsion, alpha-carbon, and hydrogen-bond distance matrices.

        Results are stored as ``self.torsion_distances``, ``self.ca_distances``,
        and ``self.hbond_distances``.

        Parameters
        ----------
        load_matrices : str, optional
            Path to a directory containing pre-computed matrices named
            ``torsions_distances.txt``, ``ca_distances.txt``, and
            ``hbond_distances.txt``. If None, matrices are computed fresh.
        stride : int, optional
            Sub-sampling stride applied after loading saved matrices. Default 1.
        save_matrices : str, optional
            Path to a directory where computed matrices are saved. Defaults to
            the current working directory.
        """

        # Compute 
        self.torsion_distances, self.ca_distances, self.hbond_distances = _compute_component_matrices(self.super_traj, self.ca_resSeqs, load_matrices=load_matrices, save_matrices=save_matrices)

        # Adjust to stride
        if not len(self.torsion_distances) == self.super_traj.n_frames:
            print('Doing something here', flush=True)
            self.torsion_distances = self.torsion_distances[::stride, ::stride]
            self.ca_distances = self.ca_distances[::stride, ::stride]
            self.hbond_distances = self.hbond_distances[::stride, ::stride]


    def compute_matrix(self, weights: List[float]=[0.33, 0.33, 0.33], save_matrices=os.getcwd()):
        """
        Combine component matrices into a single weighted distance matrix.

        Parameters
        ----------
        weights : list of float, optional
            Weights for [torsion, alpha-carbon, hydrogen-bond] matrices.
            Default ``[0.33, 0.33, 0.33]``.
        save_matrices : str, optional
            Directory path for saving the composite matrix. Defaults to the
            current working directory.

        Returns
        -------
        matrix : np.ndarray
            Composite distance matrix of shape (n_frames, n_frames).
        """

        # Combine matrices
        self.weights = weights
        
        return _compute_matrix(self.torsion_distances, self.ca_distances, self.hbond_distances, weights=self.weights, save_matrices=save_matrices)

        
        