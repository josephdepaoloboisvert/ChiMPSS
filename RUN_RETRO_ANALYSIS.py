import argparse, os

parser = argparse.ArgumentParser(
    description='Retroactively compute distance matrices and convergence report '
                'for a completed FultonMarket simulation.'
)

parser.add_argument('input_dir',  type=str,
                    help='Path to the FultonMarket output directory '
                         '(contains netcdfs and saved_variables dir).')
parser.add_argument('pdb',        type=str,
                    help='Path to the topology PDB used for the simulation.')

parser.add_argument('--output_cache_dir', default=None, type=str,
                    help='Directory to read/write cached distance matrices. '
                         'Default: <input_dir>/retro_cache/.')
parser.add_argument('--n_resample', default=1000, type=int,
                    help='Frames to importance-resample per sub-simulation. '
                         'Default 1000.')

args = parser.parse_args()

# --- Resolve cache directory ---
output_cache_dir = args.output_cache_dir
if not os.path.isdir(output_cache_dir):
    print(f"Creating cache directory at: {output_cache_dir}")
    os.makedirs(output_cache_dir, exist_ok=True)
    
# --- Run ---
from FultonMarket.FultonMarketAnalysis import FultonMarketAnalysis

#Dirty harcoded block so I can run on expanse fast
analyzer = FultonMarketAnalysis(input_dir=args.input_dir, pdb=args.pdb, sele_str='resname UNK')    
matrices = analyzer.retro_analyze_all(n_resample=args.n_resample, output_cache_dir=output_cache_dir, getcontacts_script='/expanse/lustre/projects/uil133/josephdb/getcontacts/get_dynamic_contacts.py', conda_env='pyinteraph2', getcontacts_python=os.path.join(os.environ['CONDA_PREFIX'].replace('replica2','pyinteraph2'),'bin','python'))
