"""
USAGE: python REPLICA_EXCHANGE.py $INPUT_DIR $NAME $OUTPUT_DIR $REPLICATE $SIM_TIME $NUM_OF_REPLICA 

PARAMETERS:
-----------
    INPUT_DIR: absolute path to the directory with input xml and pdb
    NAME: pdb file before the extension
    OUTPUT_DIR: absolute path to the directory where a subdirectory with output will be stored
    REPLICATE: Replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting
    SIM_TIME: Total simulation aggregate time. Default is 500 ns. 
    NUM_OF_REPLICA: number of replica to start with between T_min (300 K) and T_max (360 K)
"""


import os, sys, argparse
import numpy as np

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_dir', help="absolute path to the directory with input xml and pdb")
parser.add_argument('name', help="pdb file before the extension")
parser.add_argument('output_dir', help="absolute path to the directory where a subdirectory with output will be stored")
parser.add_argument('replicate', help="replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting")
parser.add_argument('--no-context', action='store_true', help="if this option is chosen, do not use and openMM state to contextualize the simulation. recommended to continuing simulations only.")
parser.add_argument('-c', '--convergence-thresh', default=None, type=float, help='amount of time (ns) the simulation needs to be converged according to the mean weighted reduced cartesians of resampled frames. Default is None, but 350 is recommended. If this is not None, then this criterion will be used over total simulation time (see below).')
parser.add_argument('-r', '--resSeqs-npy', default=None, type=str, help='path to numpy array of resSeqs to use to compute the PCA and evaluate the mean weighted reduced cartesians. If convergence_thresh is not None, this option should be specified. Default is None.')
parser.add_argument('-t', '--total-sim-time', default=None, type=int, help="aggregate simulation time from all replicates in nanoseconds. Default is None. If this option is specified and convergence_thresh is None, then this criterion will be used to evaluate when the simulation is complete.")
parser.add_argument('-s', '--sub-sim-length', default=50, type=int, help="   Amount of time for each sub simulation in nanoseconds. This value dictates how often .ncdf objects are truncated, data is store, resampling occures, PCA analysis occurs, andconvergence criterion is evaluated. Default is 50, but 25 is recommended.")
parser.add_argument('-n', '--n-replica', default=100, help="number of replica to start with between T_min (300 K) and T_max (360 K)", type=int)
parser.add_argument('-x', '--sele-str', nargs='+', default=None, type=str, help='ligand selection string for mdtraj')
parser.add_argument('-i', '--iter-length', default=0.001, type=float, help='Length of iteration (ns)')
args = parser.parse_args()

sys.path.append('FultonMarket')
from FultonMarket import FultonMarket

if __name__ == '__main__':

    # Inputs
    input_dir = args.input_dir
    name = args.name
    input_sys = os.path.join(input_dir, name+'_sys.xml')
    if args.resSeqs_npy is not None:
        resSeqs = np.load(args.resSeqs_npy)
    else:
        resSeqs = None
    if args.no_context:
        input_state = None
    else:
        input_state = os.path.join(input_dir, name+'_state.xml')
    
    input_pdb = os.path.join(input_dir, name+'.pdb')
    if args.sele_str is not None:
        sele_str = ' '.join(args.sele_str)
    else:
        sele_str = None    
    # Outputs
    output_dir = os.path.join(sys.argv[3], name + '_' + str(args.replicate))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert os.path.exists(output_dir)

    
    # Run rep exchange
    market = FultonMarket(input_pdb=input_pdb, 
                          input_system=input_sys, 
                          input_state=input_state, 
                          n_replicates=args.n_replica,
                          sele_str=sele_str)
    
    market.run(iter_length=args.iter_length,
              dt=2.0,
              sim_length=args.sub_sim_length,
              convergence_thresh=args.convergence_thresh,
              resSeqs=resSeqs,
              total_sim_time=args.total_sim_time,
              output_dir=output_dir)
    
