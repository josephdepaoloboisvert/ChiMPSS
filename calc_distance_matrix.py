import numpy as np
import mdtraj as md
from DistanceMatrix.DistanceMatrix import DistanceMatrix

supertraj = md.load('AM630_comparison_10k_super.dcd', top='AM630_comparison_10k_super.pdb')
dist = DistanceMatrix(supertraj)
dist.compute_component_matrices()
matrix = dist.compute_matrix()
np.save('AM630_10k_dist_mat.npy', matrix)
