# Imports
import os, sys, math, glob, itertools, jax
import jax.numpy as jnp
from datetime import datetime
import netCDF4 as nc
import numpy as np
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
import warnings
warnings.filterwarnings('ignore')
from typing import List

printf = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
perms = jnp.array([x for x in itertools.product([-1, 0, 1], repeat=3)])
jaxrmsd = lambda a, b: jnp.sqrt(jnp.mean(jnp.sum((b-a)**2, axis=-1), axis=-1))
fprint = lambda x: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + x, flush=True)
jax_add = lambda a, b: a+b
jax_add = jax.vmap(jax_add, in_axes=(0, None))
rmsd_j = jax.vmap(jaxrmsd, in_axes=(0, None))

def rmsd(a, b):
    if len(a.shape) == 1:
        return np.sqrt(((a - b)**2).sum(-1).mean())
    else:
        return np.array([np.sqrt(((a[i] - b[i])**2).sum(-1).mean()) for i in range(a.shape[0])])

def plot_MRC(domains, mean_weighted_rc, mean_weighted_rc_err, savefig: str=None):

    fig, ax = plt.subplots()
    
    ax.plot(domains, mean_weighted_rc, color='k')
    ax.errorbar(domains, np.abs( mean_weighted_rc - mean_weighted_rc[-1]), yerr=mean_weighted_rc_err, color='k', capsize=3)
    ax.set_ylabel('| MRC(t) - MRC(T) |')
    ax.set_xlabel('Total sim time (ns')
    plt.show()

    if savefig is not None:
        fig.savefig(savefig)


def PCA_convergence_detection(rc, rc_err):

    converged = np.array([False for i in range(len(rc)-1)])
    for i, (rc_i, rc_err_i) in enumerate(zip(rc[:-1], rc_err[:-1])):
        if (rc_i > rc[-1] and rc_i - rc_err_i <= rc[-1]) or (rc_i <= rc[-1] and rc_i + rc_err_i >= rc[-1]):
            converged[i] = True
        else:
            converged[:i+1] = False

    return converged


def write_traj_from_pos_boxvecs(pos, box_vec, top, sele_str, receptor_sele_str='chainid 0', correction: bool=True):        
    
    # Create traj obj
    traj = md.Trajectory(xyz=pos.copy(), 
                         topology=top, 
                         time=np.arange(pos.shape[0]), 
                         unitcell_lengths=box_vec.sum(axis=1), 
                         unitcell_angles=np.repeat([90,90,90], pos.shape[0]).astype(np.float32).reshape(pos.shape[0], 3))

    # Correct periodic issues
    prot_sele = traj.topology.select(receptor_sele_str) # receptor should always be chainid 0
    if sele_str is not None and correction:
        if type(sele_str) == str:
            sele_str = [sele_str]
        lig_seles = []
        lig_coms = []
        for s in sele_str:
            try:
                lig_sele = traj.topology.select(s)
            except:
                raise Exception(f'failed to parse {s}')
            lig_seles.append(lig_sele)
            lig_coms.append(md.compute_center_of_mass(traj.atom_slice(lig_sele)))
        prot_com = md.compute_center_of_mass(traj.atom_slice(prot_sele))
        for frame in range(traj.n_frames):
            for (s, lig_sele, lig_com) in zip(sele_str, lig_seles, lig_coms):
                best_trans, _ = best_translation_by_unitcell(traj.unitcell_lengths[frame], lig_com[frame], prot_com[frame])
                traj.xyz[frame, lig_sele, :] += best_trans


    # Align frames for veiwing purposes
    if correction:
        traj = traj.superpose(traj, atom_indices=prot_sele, ref_atom_indices=prot_sele)

    return traj


def get_traj_PCA(traj, explained_variance_threshold: float=None):
    """
    """
    # PCA
    pca = PCA()
    reduced_cartesian = pca.fit_transform(traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3))
    explained_variance = np.array([np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(pca.n_components_)])

    if explained_variance_threshold is not None:
        n_components = int(np.where(explained_variance >= explained_variance_threshold)[0][0])

    return pca, reduced_cartesian[:,:n_components], explained_variance[:n_components], n_components    


def calculate_weighted_rc(reduced_cartesian, resampled_inds, upper_limit, pca_weights, mbar_weights):
    assert reduced_cartesian.shape[0] == len(resampled_inds), f'{reduced_cartesian.shape}, {resampled_inds.shape}'
    assert reduced_cartesian.shape[0] == len(mbar_weights)
    assert reduced_cartesian.shape[1] == len(pca_weights)
    
    mean_weighted_rcs = []
    for (rc, frame_no, mbar_weight) in zip(reduced_cartesian, resampled_inds[:,1], mbar_weights):
        if frame_no <= upper_limit:
            mean_weighted_rcs.append(np.mean(rc * pca_weights) * pca_weights)
            
    return np.mean(mean_weighted_rcs), np.std(mean_weighted_rcs) / np.sqrt(reduced_cartesian.shape[0])
    

@staticmethod
def resample_with_MBAR(objs: List, u_kln: np.array, N_k: np.array, size: int, reshape_weights: tuple=None, specify_state: int=0, return_inds: bool=False, return_weights: bool=False, return_resampled_weights: bool=False, replace: bool=True):

    # Get MBAR weights
    weights = compute_MBAR_weights(u_kln, N_k)

    # Reshape weights if specified
    if reshape_weights is not None:
        weights = weights.reshape(reshape_weights)
        
    # Get probabilities
    if len(weights.shape) == 1:
        probs = weights.copy()
    else:
        probs = weights[:, specify_state]


    # Resample
    if size == -1:
        resampled_inds = np.where(probs >= probs.max()*0.001)[0]
        printf(f'Top 99.9% of probability includes {len(resampled_inds)} no. of frames')
    else:
        probs /= np.nan_to_num(probs).sum() # Renormalize to avoid errors
        probs = np.nan_to_num(probs)
        try:
            resampled_inds = np.random.choice(range(len(probs)), size=size, replace=replace, p=probs)
        except: 
            resampled_inds = np.random.choice(range(len(probs)), size=len(np.where(probs > 0)[0]), replace=replace, p=probs)
        
    resampled_objs = []
    for obj in objs:
        resampled_objs.append(np.array([obj[resampled_ind] for resampled_ind in resampled_inds]))
    # Return resampled objects
    return_list = []
    if len(objs) == 1:
        return_list.append(resampled_objs[0])
    elif len(objs) > 1:
        for resampled_obj in resampled_objs:
            return_list.append(resampled_obj)

    # Optional returns
    if return_inds:
        return_list.append(resampled_inds)
    if return_weights:
        return_list.append(weights)
    if return_resampled_weights:
        resampled_weights = weights[resampled_inds, specify_state]
        return_list.append(resampled_weights)

    return return_list


@staticmethod
def compute_MBAR_weights(u_kln, N_k):
    """
    """
    mbar = MBAR(u_kln, N_k, initialize='BAR')

    return mbar.weights()


@staticmethod
# PCA
def detect_PC_equil(pc, reduced_cartesian):
    t0, _, _ = detect_equilibration(reduced_cartesian[:,pc])

    return t0


def detect_energy_equil(avg_energies):
    t0, _, _ = detect_equilibration(avg_energies)

    return t0


def get_restraint_energy_kT(pos, trans, centers, T, spring_constant):
    pos += trans # Translate, if necessary, to avoid wrapping issues
    pos *= 10 #Convert to Angstrom
    centers *= 10 #Convert to Angstrom
    x_dis = np.sum((centers[:,0] - pos[:,0])**2, axis=0)
    y_dis = np.sum((centers[:,1] - pos[:,1])**2, axis=0)
    z_dis = np.sum((centers[:,2] - pos[:,2])**2, axis=0)
    displacement_sq = np.sum((x_dis, y_dis, z_dis), axis=0)

    restraint_energy = (1 / (2 * 8.3145 * T)) * spring_constant * displacement_sq # E(kT) = (1/2RT) * k_spring * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)  This expression converts restraint energies in (J/mol) to kT to match openmmtools energies

    return restraint_energy


def get_truncation_atom_keep_inds(top):
    remove_atom_inds = top.select('resname HOH NA CL POP')
    return np.array([i for i in range(top.n_atoms) if i not in remove_atom_inds])
    

def best_translation_by_unitcell(cell_lengths, mobile_coords, target_coords):
    perms = np.array([x for x in itertools.product([-1, 0, 1], repeat=3)])
    
    translations = cell_lengths * perms
    permuted_positions = np.array([np.sum((translations[i], mobile_coords), axis=0) for i in range(translations.shape[0])])
    rmsds_of_permutations = np.array([rmsd(permuted_positions[i], target_coords) for i in range(permuted_positions.shape[0])])
    return translations[np.argmin(rmsds_of_permutations)], rmsds_of_permutations[np.argmin(rmsds_of_permutations)]

    
def best_translation_by_unitcell_jax(cell_lengths, mobile_coords, target_coords):
    translations = cell_lengths * perms
    permuted_positions = jax_add(translations, mobile_coords)
    rmsds_of_permutations = rmsd_j(permuted_positions, target_coords)
    return translations[jnp.argmin(rmsds_of_permutations)], rmsds_of_permutations[jnp.argmin(rmsds_of_permutations)]

# Jax to speed up some functions
best_translation_by_unitcell_jax = jax.vmap(best_translation_by_unitcell_jax, in_axes=(0, 0, None))


