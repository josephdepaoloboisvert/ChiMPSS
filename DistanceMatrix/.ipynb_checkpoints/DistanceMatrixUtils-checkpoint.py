import os, sys
import mdtraj as md
import numpy as np
from typing import List
from datetime import datetime
import multiprocessing as mp

printf = lambda x : print(f"{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}//{x}", flush=True)


def _compute_component_matrices(super_traj: md.Trajectory, ca_resSeqs: List[int]=[], load_matrices: bool=False, save_matrices: str=os.getcwd(), _force_fail: bool=False):
    """
    Calculates pairwise, symmetric distance matrix for each frame through custom distances between torsions, specified CA distances, and local Hbond acceptor/donor distances

        Each distance between two frames is composed of three components:
            Torsion distances =  sqrt((sum((180 - ||a_i - a_j| - 180|)**2)/n_angles)
            Alpha carbon distances = sqrt(sum((dist_i - dist_j)**2)/n_atoms)
            Hydrogen bond donor/acceptor distances = sqrt(sum((dist_i - dist_j)**2)/n_atoms)
                Only calculated if the distance between two atoms is <= 8.0 Angstrom in initial frame of super_traj
            
        Components are normalized:
            Torsion distances /= max(Torsion distances)
            Alpha carbon distances /= max(Alpha carbon distances)
            Hydrogen bond distances /= max(Hydrogen bond distances)

    Parameters:
    ----------
        super_traj (md.Trajectory): 
            Trajectory with all data aggregated into one trajectory.         
        
        ca_resSeqs (list of ints): 
            List of residue numbers that will be used for pairwise distance calculations. 

        weights (list of floats): 
            List of floats that represent weights for each component in shape [Torsions weight (float), CA weight (float), Hbond weight (float)]. Default is [0.33, 0.33, 0.33].

        load_matrices (str, list, or None): 
            if string is provided, it is assumed it is a string pointing to the directory in which the distance matrices are saved
            if list is provided, it is assumed it is a list of filenames for the matrices in the order torsions, CAs, H_bonds
            if None is provided, the matrices will be computed and saved in default locations
    
    Returns:
    --------
        torsion_distances (np.array): 
            Distance matrix comparing the torsions of all frames in frame_labels. Length must match frame_labels. 

        CA_distances (np.array): 
            Distance matrix comparing the ca distances of all frames in frame_labels. Length must match frame_labels. 

        hbond_distances (np.array): 
            Distance matrix comparing the local hbond distances of all frames in frame_labels. Length must match frame_labels.
    """
    built_any = False

    try:
        # Load torsion matrix
        printf('Loading torsion matrix...')
        torsion_distances = np.memmap(os.path.join(load_matrices, 'torsion_distances.raw'),
                                      mode='r+', dtype='float32', shape=(super_traj.xyz.shape[0], super_traj.xyz.shape[0]))
        printf(f'Successfully loaded torsion matrix. Shape: {torsion_distances.shape}')
    
    except:    
        # Build torsion matrix
        printf('Building torsion matrix...')
        torsion_distances = _calc_torsion_matrix(super_traj, filename=os.path.join(save_matrices, 'torsion_distances.raw'))
        printf(f'Torsion matrix built and saved in {save_matrices}')
        del torsion_distances
        built_any=True

    try:
        # Load CA matrix
        assert os.path.exists(load_matrices)
        printf('Loading CA matrix...')
        ca_distances = np.memmap(os.path.join(load_matrices, 'ca_distances.raw'),
                                 mode='r+', dtype='float32', shape=(super_traj.xyz.shape[0], super_traj.xyz.shape[0]))
        printf(f'Successfully loaded CA matrix. Shape:{ca_distances.shape}')

    except:
        # Build CA matrix
        printf('Building CA matrix...')
        ca_distances = _calc_CA_matrix(super_traj, ca_resSeqs, filename=os.path.join(save_matrices, 'ca_distances.raw'))
        printf(f'CA matrix built and saved in {save_matrices}')
        del ca_distances
        built_any = True

    try:
        # Load hbond matrix
        assert os.path.exists(load_matrices)
        printf('Loading hbond matrix...')
        hbond_distances = np.memmap(os.path.join(load_matrices, 'hbond_distances.raw'),
                                    mode='r+', dtype='float32', shape=(super_traj.xyz.shape[0], super_traj.xyz.shape[0]))
        printf(f'Successfully loaded hbond matrix. Shape: {hbond_distances.shape}')

    except:
        # Build hbond matrix
        printf('Building hbond matrix...')
        hbond_distances = _calc_hbond_matrix(super_traj, filename=os.path.join(save_matrices, 'hbond_distances.raw'))
        hbond_max = hbond_distances.max()
        if hbond_max > 1:
            printf('Normalizing hbonds...')
            hbond_distances /= hbond_max
        else:
            printf('Hbonds already normalized.')
        printf(f'Hbond matrix built and saved in {save_matrices}')
        del hbond_distances
        built_any = True
    
    
    if built_any:
        if _force_fail:
            raise Exception()
        else:
            return _compute_component_matrices(super_traj, load_matrices=save_matrices, _force_fail=True)
    else:
        return torsion_distances, ca_distances, hbond_distances


    

def _compute_matrix(torsion_distances, ca_distances, hbond_distances, weights: List[float]=[0.333, 0.333, 0.333], save_matrices=os.getcwd()):
    """
    Combine weighted self.torsion_distances, self.ca_distances, self.hbond_distances to return final distance matrix

    Parameters:
    -----------
        torsion_distances (np.array): 
            Distance matrix comparing the torsions of all frames in frame_labels. Length must match frame_labels. 

        CA_distances (np.array): 
            Distance matrix comparing the ca distances of all frames in frame_labels. Length must match frame_labels. 

        hbond_distances (np.array): 
            Distance matrix comparing the local hbond distances of all frames in frame_labels. Length must match frame_labels. 

        weights (List[float]):
            Weights for combining matrices into final distance matrix. Shape of weights must follow format [Torsions weight (float), CA weight (float), Hbond weight (float)]

        base_filename (string):
            Will append the given weights and .raw extension to this and save the distance matrix here.

    Returns:
    --------
        dist_matrix (np.array);
            Weighted combined distance matrix from torsion_distances, ca_distances, hbond_distance 
    """
    assert len(weights) == 3, 'Shape of weights must follow format [Torsions weight (float), CA weight (float), Hbond weight (float)].'
    assert torsion_distances.shape == ca_distances.shape
    assert torsion_distances.shape == hbond_distances.shape

    #Define the insertion function
    global make_dist_mat_row
    def make_dist_mat_row(i):
        return weights[0]*torsion_distances[i] + weights[1]*ca_distances[i] + weights[2]*hbond_distances[i]

    #MultiProcess
    printf('Starting MP Run for weighted distance matrix')
    with mp.Pool(processes=120) as pool:
        dist_matrix = pool.map(make_dist_mat_row, range(torsion_distances.shape[0]))
        pool.close()
        pool.join()
    printf('Done with MP Run for weighted distance matrix')
    del make_dist_mat_row

    return np.array(dist_matrix)

#TORSIONS
def _calc_torsions(traj):
    _, chi1 = md.compute_chi1(traj)
    _, chi2 = md.compute_chi2(traj)
    _, psi = md.compute_psi(traj)
    _, phi = md.compute_phi(traj)
    angles = np.concatenate((chi1, chi2, psi, phi), axis=1)
    return angles*180/np.pi

def _calc_torsion_matrix(traj, filename='torsional_matrix.raw'):
    #Obtain Feature Vectors
    angles = _calc_torsions(traj)
    
    #Open a Memory Map
    torsional_matrix = np.memmap(filename, dtype='float32', mode='w+',
                                 shape=(traj.n_frames, traj.n_frames))
    #Define a row function
    def torsional_distance_row(angles_i):
        return np.round(np.sqrt((1/len(angles_i))*np.sum((180 - np.abs((np.abs(angles_i - angles) - 180)))**2, axis=1)), 2)
        
    #Define an insertion function
    global add_tors_matrix_row
    def add_tors_matrix_row(row_index):
        torsional_matrix[row_index] = torsional_distance_row(angles[row_index])
        
    #MultiProcess
    printf('Starting MP Run for torsional distance matrix')
    with mp.Pool(processes=100) as pool:
        pool.map(add_tors_matrix_row, range(traj.n_frames))
        pool.close()
        pool.join()
    printf('Done with MP Run for torsional distance matrix')
    del add_tors_matrix_row

    # Normalize
    torsional_matrix /= torsional_matrix.max()
    torsional_matrix.flush()
    
    return torsional_matrix



#ALPHA CARBONS
def _calc_CA_dist(traj, resSeqs=None):
    
    #Determine pairs of CAs
    if resSeqs is None: #Then use all of them
        atoms = traj.top.select('name CA')
    else:
        top = traj.topology
        atoms = [top.select(f'protein and resSeq {resSeq} and name CA')[0] for resSeq in resSeqs]
    
    #Bin these into pairs
    pairs = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)): 
            pairs.append([atoms[i], atoms[j]])
    pairs = np.array(pairs)
    
    #Get the distances (in angstrom)
    return md.compute_distances(traj, pairs)*10


    
def _calc_CA_matrix(traj, resSeqs, filename='matrices/CA_matrix.raw'):
    #Obtain Feature Vectors
    ca = _calc_CA_dist(traj, resSeqs)
    
    #Open a Memory Map
    global CA_matrix
    CA_matrix = np.memmap(filename, dtype='float32', mode='w+',
                          shape=(ca.shape[0], ca.shape[0]))
    
    #Define a row function
    def CA_distance_row(ca_i):
        return np.round(np.sqrt((1/len(ca_i))*np.sum(np.abs(ca_i - ca)**2, axis=1)), 2)
        
    #Define an insertion function
    global add_CA_matrix_row
    def add_CA_matrix_row(row_index):
        CA_matrix[row_index] = CA_distance_row(ca[row_index])
        
    #MultiProcess?
    printf('Starting MP Run for CA distance matrix')
    with mp.Pool(processes=75) as pool:
        pool.map(add_CA_matrix_row, range(ca.shape[0]))
    printf('Done with MP Run for CA distance matrix')    
    del add_CA_matrix_row

    # Normalize 
    CA_matrix /= CA_matrix.max()
    CA_matrix.flush()
    
    return CA_matrix


    
#HYDROGEN BONDS
def _calc_hbond_dist(traj, threshold=12):
    #Find Pairs
    indices = traj.topology.select('type O N S and not name O N')
    pairs = []
    init_pos = traj.xyz[0]
    for i in indices:
        for j in indices:
            if j > i and np.linalg.norm(init_pos[i] - init_pos[j])*10 <= threshold:
                pairs.append([i, j])
    pairs = np.array(pairs)

    return md.compute_distances(traj, pairs)*10



def _calc_hbond_matrix(traj, threshold=12, filename='HB_matrix.raw'):
    #Obtain Feature Vectors
    hbond = _calc_hbond_dist(traj, threshold=threshold)
    
    #Open a Memory Map
    HB_matrix = np.memmap(filename, dtype='float32', mode='w+',
                          shape=(hbond.shape[0], hbond.shape[0]))
    #Define a row function
    def HB_distance_row(hbond_i):
        return np.round(np.sqrt((1/len(hbond_i))*np.sum(np.abs(hbond_i - hbond)**2, axis=1)), 2)
        
    #Define an insertion function
    global add_HB_matrix_row
    def add_HB_matrix_row(row_index):
        HB_matrix[row_index] = HB_distance_row(hbond[row_index])
    
    #MultiProcess
    printf('Starting MP Run for HB distance matrix')
    with mp.Pool(processes=70) as pool:
        pool.map(add_HB_matrix_row, range(hbond.shape[0]))
    printf('Done with MP Run for HB distance matrix')
    del add_HB_matrix_row

    # Normalize
    HB_matrix /= HB_matrix.max()
    HB_matrix.flush() 
    
    return HB_matrix
    
