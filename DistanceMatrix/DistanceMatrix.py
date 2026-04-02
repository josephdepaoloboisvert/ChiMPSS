# Package Imports
import os
import mdtraj as md
import numpy as np
from typing import List
#Custom Code Imports
from .DistanceMatrixUtils import _compute_component_matrices, _compute_matrix, printf


class DistanceMatrix():
    """
    """

    def __init__(self, super_traj: md.Trajectory, ca_resSeqs: List[int]=None):
        """ 
        Initialize a DistanceMatrix object.

        Parameters:
        ----------
            super_traj (md.Trajectory): 
                Trajectory with all data aggregated into one trajectory.         
            
            ca_resSeqs (list of ints): 
                List of residue numbers that will be used for pairwise distance calculations. 

        Returns:
        --------
            DistanceMatrix object.
        """

        # Attributes
        self.super_traj = super_traj
        self.ca_resSeqs = ca_resSeqs


    def compute_component_matrices(self, load_matrices: str=None, stride: int=1, save_matrices: str=os.getcwd()):
        """
        Compute torsion, CA, and Hbond matrices. 

        Parameters:
        ----------        
            load_matrices (str): 
                Absolute string path to directory where matrices will be loaded and are name torions_distances.txt, ca_distances.txt, hbond_distances.txt. Default is None which will not load matrices.

            stride (int):
                If load_matrices is proveded, stride is for parsing distance matrix based on simulation timestep. Default is 1 to load everything. 
                
            save_matrices (str): 
                Absolute string path to directory where matrices will be saved as torions_distances.txt, ca_distances.txt, hbond_distances.txt. Default is None which will not save matrices.       
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
        Combine weighted self.torsion_distances, self.ca_distances, self.hbond_distances to return final distance matrix

        Parameters:
        -----------
            weights (List[float]):
                Weights for combining matrices into final distance matrix. Shape of weights must follow format [Torsions weight (float), CA weight (float), Hbond weight (float)]
        """

        # Combine matrices
        self.weights = weights
        
        return _compute_matrix(self.torsion_distances, self.ca_distances, self.hbond_distances, weights=self.weights, save_matrices=save_matrices)

        
        