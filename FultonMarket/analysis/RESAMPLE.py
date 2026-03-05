import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('pdb_dir', help="path to directory with .pdb files for each simulation in 'repexchange_dir'")
parser.add_argument('repexchange_dir', help="path to directory with the replica exchange output directories")
parser.add_argument('output_dir', help="path to outputdir where resampled trajectories will be stored.")
parser.add_argument('--resSeqs-npy', default=None, help="path to .npy file with the resSeqs to include for principal component analysis and equilibration detection")
parser.add_argument('--nframes', default=-1, type=int, help="number of frames to resample, default is -1 meaning to resample frames from the top 99.9%% of probability.")
parser.add_argument('--no-replace', action='store_true', help="choose not to use resampling with replacement. this is only recommended when n_frames is -1 or default")
parser.add_argument('--upper-limit', default=None, type=int, help="upper limit (number of frames) for resampling. Default is None, meaning all of the frames from replica exchange will be included in resampling.")
parser.add_argument('--sim-names', default=None, help="Comma delimited string of replica exchange names to analyze. EX: drug_1,drug_2,drug_3")
parser.add_argument('--skip', default=[], help="Comma delimited string of replica exchaange names to skip analysis")
parser.add_argument('--sele-str', default=None, type=str, help="mdtraj selection string for the ligand. Default is None")
parser.add_argument('--force', action='store_true', default=False, help='Force the resample, even if output files already exist.')
parser.add_argument('--no-correction', action='store_true', default=False, help='Do not perform any corrections (translations, alignment, wrapping).')
args = parser.parse_args()

from FultonMarketAnalysis import FultonMarketAnalysis
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymbar.timeseries import detect_equilibration
import os, sys
import netCDF4 as nc 
import multiprocessing as mp
import jax


# Multiprocessing method
def resample(dir, pdb, upper_limit, resSeqs, pdb_out, dcd_out, weights_out, inds_out, mrc_out, n_samples, replace, sele_str, correction):

   
    # Initialize
    analysis = FultonMarketAnalysis(dir, pdb, skip=10, upper_limit=upper_limit, resSeqs=resSeqs, sele_str=sele_str)

    # Importance Resampling
    analysis.determine_equilibration()
    analysis.importance_resampling(n_samples=n_samples, replace=replace)

    # Write out
    analysis.write_resampled_traj(pdb_out, dcd_out, weights_out, correction=correction)
    
    del analysis    


if __name__ == '__main__':
    
    # Input
    if args.sim_names is not None:
        sims = sorted(args.sim_names.split(','))
    else:
        sims = sorted(os.listdir(args.repexchange_dir))
    repexchange_dir = args.repexchange_dir
    output_dir = args.output_dir
    pdb_dir = args.pdb_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'pdb')):
        os.mkdir(os.path.join(output_dir, 'pdb'))
    if not os.path.exists(os.path.join(output_dir, 'dcd')):
        os.mkdir(os.path.join(output_dir, 'dcd'))
    if not os.path.exists(os.path.join(output_dir, 'mbar_weights')):
        os.mkdir(os.path.join(output_dir, 'mbar_weights'))
    if not os.path.exists(os.path.join(output_dir, 'mrc')):
        os.mkdir(os.path.join(output_dir, 'mrc'))
    if not os.path.exists(os.path.join(output_dir, 'inds')):
        os.mkdir(os.path.join(output_dir, 'inds'))

    # Set up arguments
    mpargs = []
    for sim in sims:

        if sim in args.skip:
            continue
        
        # Define outputs
        pdb_out = os.path.join(output_dir, 'pdb', "_".join(sim.split('_')[:-1]) + '.pdb')
        dcd_out = os.path.join(output_dir, 'dcd', "_".join(sim.split('_')[:-1]) + '.dcd')
        weights_out = os.path.join(output_dir, 'mbar_weights', "_".join(sim.split('_')[:-1]) + '.npy')
        inds_out = os.path.join(output_dir, 'inds', "_".join(sim.split('_')[:-1]) + '.npy')
        mrc_out = os.path.join(output_dir, 'mrc', "_".join(sim.split('_')[:-1]) + '.npy')
        if not os.path.exists(dcd_out) or args.force:
            
            # Sim input
            dir = os.path.join(repexchange_dir, sim)
            pdb = os.path.join(pdb_dir, "_".join(sim.split('_')[:-1]) + '.pdb')
            upper_limit = args.upper_limit
            if args.resSeqs_npy is not None:
                resSeqs = np.load(args.resSeqs_npy)
            else:
                resSeqs = None
            n_samples = args.nframes
            if args.no_replace:
                replace = False
            else:
                replace = True
            if not sim.startswith('apo') and args.sele_str is not None:
               sele_str = args.sele_str
            else:
               sele_str = None

            # Resample
            resample(dir, pdb, upper_limit, resSeqs, pdb_out, dcd_out, weights_out, inds_out, mrc_out, n_samples, replace, sele_str, correction=not args.no_correction)

    

