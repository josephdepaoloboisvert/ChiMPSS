"""
Consolidated distance matrix computation (torsion + CA + H-bond components).

Migrated from DistanceMatrix/DistanceMatrix.py and DistanceMatrix/DistanceMatrixUtils.py.
"""

import os
import multiprocessing as mp
from datetime import datetime
from typing import List

import numpy as np
import mdtraj as md

printf = lambda x: print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True)


class DistanceMatrix:

    def __init__(self, super_traj: md.Trajectory, ca_resSeqs: List[int] = None):
        """
        Parameters
        ----------
        super_traj : aggregated trajectory
        ca_resSeqs : residue numbers for pairwise CA distance calculations
        """
        self.super_traj = super_traj
        self.ca_resSeqs = ca_resSeqs

    def compute_component_matrices(self, load_matrices: str = None,
                                   stride: int = 1,
                                   save_matrices: str = os.getcwd()):
        self.torsion_distances, self.ca_distances, self.hbond_distances = \
            _compute_component_matrices(self.super_traj, self.ca_resSeqs,
                                        load_matrices=load_matrices,
                                        save_matrices=save_matrices)
        if not len(self.torsion_distances) == self.super_traj.n_frames:
            self.torsion_distances = self.torsion_distances[::stride, ::stride]
            self.ca_distances      = self.ca_distances[::stride, ::stride]
            self.hbond_distances   = self.hbond_distances[::stride, ::stride]

    def compute_matrix(self, weights: List[float] = None, save_matrices=os.getcwd()):
        if weights is None:
            weights = [0.33, 0.33, 0.33]
        self.weights = weights
        return _compute_matrix(self.torsion_distances, self.ca_distances,
                               self.hbond_distances, weights=self.weights,
                               save_matrices=save_matrices)


# ── component-matrix helpers ──────────────────────────────────────────────────

def _compute_component_matrices(super_traj: md.Trajectory,
                                ca_resSeqs: List[int] = None,
                                load_matrices=None,
                                save_matrices: str = os.getcwd(),
                                _force_fail: bool = False):
    if ca_resSeqs is None:
        ca_resSeqs = []
    built_any = False

    try:
        printf('Loading torsion matrix...')
        torsion_distances = np.memmap(
            os.path.join(load_matrices, 'torsion_distances.raw'),
            mode='r+', dtype='float32',
            shape=(super_traj.xyz.shape[0], super_traj.xyz.shape[0]))
        printf(f'Successfully loaded torsion matrix. Shape: {torsion_distances.shape}')
    except Exception:
        printf('Building torsion matrix...')
        torsion_distances = _calc_torsion_matrix(
            super_traj, filename=os.path.join(save_matrices, 'torsion_distances.raw'))
        printf(f'Torsion matrix built and saved in {save_matrices}')
        del torsion_distances
        built_any = True

    try:
        assert os.path.exists(load_matrices)
        printf('Loading CA matrix...')
        ca_distances = np.memmap(
            os.path.join(load_matrices, 'ca_distances.raw'),
            mode='r+', dtype='float32',
            shape=(super_traj.xyz.shape[0], super_traj.xyz.shape[0]))
        printf(f'Successfully loaded CA matrix. Shape:{ca_distances.shape}')
    except Exception:
        printf('Building CA matrix...')
        ca_distances = _calc_CA_matrix(
            super_traj, ca_resSeqs,
            filename=os.path.join(save_matrices, 'ca_distances.raw'))
        printf(f'CA matrix built and saved in {save_matrices}')
        del ca_distances
        built_any = True

    try:
        assert os.path.exists(load_matrices)
        printf('Loading hbond matrix...')
        hbond_distances = np.memmap(
            os.path.join(load_matrices, 'hbond_distances.raw'),
            mode='r+', dtype='float32',
            shape=(super_traj.xyz.shape[0], super_traj.xyz.shape[0]))
        printf(f'Successfully loaded hbond matrix. Shape: {hbond_distances.shape}')
    except Exception:
        printf('Building hbond matrix...')
        hbond_distances = _calc_hbond_matrix(
            super_traj, filename=os.path.join(save_matrices, 'hbond_distances.raw'))
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
        return _compute_component_matrices(super_traj, ca_resSeqs,
                                           load_matrices=save_matrices,
                                           _force_fail=True)
    return torsion_distances, ca_distances, hbond_distances


def _compute_matrix(torsion_distances, ca_distances, hbond_distances,
                    weights: List[float] = None, save_matrices=os.getcwd()):
    if weights is None:
        weights = [0.333, 0.333, 0.333]
    assert len(weights) == 3
    assert torsion_distances.shape == ca_distances.shape
    assert torsion_distances.shape == hbond_distances.shape

    global make_dist_mat_row

    def make_dist_mat_row(i):
        return (weights[0] * torsion_distances[i]
                + weights[1] * ca_distances[i]
                + weights[2] * hbond_distances[i])

    printf('Starting MP Run for weighted distance matrix')
    with mp.Pool(processes=120) as pool:
        dist_matrix = pool.map(make_dist_mat_row, range(torsion_distances.shape[0]))
        pool.close()
        pool.join()
    printf('Done with MP Run for weighted distance matrix')
    del make_dist_mat_row

    return np.array(dist_matrix)


# ── torsion component ─────────────────────────────────────────────────────────

def _calc_torsions(traj):
    _, chi1 = md.compute_chi1(traj)
    _, chi2 = md.compute_chi2(traj)
    _, psi  = md.compute_psi(traj)
    _, phi  = md.compute_phi(traj)
    angles  = np.concatenate((chi1, chi2, psi, phi), axis=1)
    return angles * 180 / np.pi


def _calc_torsion_matrix(traj, filename='torsional_matrix.raw'):
    angles = _calc_torsions(traj)
    torsional_matrix = np.memmap(filename, dtype='float32', mode='w+',
                                 shape=(traj.n_frames, traj.n_frames))

    def torsional_distance_row(angles_i):
        return np.round(np.sqrt(
            (1 / len(angles_i)) * np.sum(
                (180 - np.abs((np.abs(angles_i - angles) - 180))) ** 2, axis=1)), 2)

    global add_tors_matrix_row

    def add_tors_matrix_row(row_index):
        torsional_matrix[row_index] = torsional_distance_row(angles[row_index])

    printf('Starting MP Run for torsional distance matrix')
    with mp.Pool(processes=100) as pool:
        pool.map(add_tors_matrix_row, range(traj.n_frames))
        pool.close()
        pool.join()
    printf('Done with MP Run for torsional distance matrix')
    del add_tors_matrix_row

    torsional_matrix /= torsional_matrix.max()
    torsional_matrix.flush()
    return torsional_matrix


# ── alpha-carbon component ─────────────────────────────────────────────────────

def _calc_CA_dist(traj, resSeqs=None):
    if resSeqs is None:
        atoms = traj.top.select('name CA')
    else:
        top   = traj.topology
        atoms = [top.select(f'protein and resSeq {resSeq} and name CA')[0]
                 for resSeq in resSeqs]
    pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pairs.append([atoms[i], atoms[j]])
    return md.compute_distances(traj, np.array(pairs)) * 10


def _calc_CA_matrix(traj, resSeqs, filename='matrices/CA_matrix.raw'):
    ca = _calc_CA_dist(traj, resSeqs)
    global CA_matrix
    CA_matrix = np.memmap(filename, dtype='float32', mode='w+',
                          shape=(ca.shape[0], ca.shape[0]))

    def CA_distance_row(ca_i):
        return np.round(np.sqrt(
            (1 / len(ca_i)) * np.sum(np.abs(ca_i - ca) ** 2, axis=1)), 2)

    global add_CA_matrix_row

    def add_CA_matrix_row(row_index):
        CA_matrix[row_index] = CA_distance_row(ca[row_index])

    printf('Starting MP Run for CA distance matrix')
    with mp.Pool(processes=75) as pool:
        pool.map(add_CA_matrix_row, range(ca.shape[0]))
    printf('Done with MP Run for CA distance matrix')
    del add_CA_matrix_row

    CA_matrix /= CA_matrix.max()
    CA_matrix.flush()
    return CA_matrix


# ── hydrogen-bond component ────────────────────────────────────────────────────

def _calc_hbond_dist(traj, threshold=12):
    indices  = traj.topology.select('type O N S and not name O N')
    pairs    = []
    init_pos = traj.xyz[0]
    for i in indices:
        for j in indices:
            if j > i and np.linalg.norm(init_pos[i] - init_pos[j]) * 10 <= threshold:
                pairs.append([i, j])
    return md.compute_distances(traj, np.array(pairs)) * 10


def _calc_hbond_matrix(traj, threshold=12, filename='HB_matrix.raw'):
    hbond   = _calc_hbond_dist(traj, threshold=threshold)
    HB_matrix = np.memmap(filename, dtype='float32', mode='w+',
                           shape=(hbond.shape[0], hbond.shape[0]))

    def HB_distance_row(hbond_i):
        return np.round(np.sqrt(
            (1 / len(hbond_i)) * np.sum(np.abs(hbond_i - hbond) ** 2, axis=1)), 2)

    global add_HB_matrix_row

    def add_HB_matrix_row(row_index):
        HB_matrix[row_index] = HB_distance_row(hbond[row_index])

    printf('Starting MP Run for HB distance matrix')
    with mp.Pool(processes=70) as pool:
        pool.map(add_HB_matrix_row, range(hbond.shape[0]))
    printf('Done with MP Run for HB distance matrix')
    del add_HB_matrix_row

    HB_matrix /= HB_matrix.max()
    HB_matrix.flush()
    return HB_matrix
