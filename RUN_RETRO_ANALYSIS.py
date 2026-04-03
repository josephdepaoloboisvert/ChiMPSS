import argparse, os

parser = argparse.ArgumentParser(
    description='Retroactively compute distance matrices and convergence report '
                'for a completed FultonMarket simulation.'
)

# --- Required inputs ---
parser.add_argument('input_dir',  type=str,
                    help='Path to the FultonMarket output directory '
                         '(contains netcdfs and saved_variables dir).')
parser.add_argument('pdb',        type=str,
                    help='Path to the topology PDB used for the simulation.')

# --- Analysis scope ---
parser.add_argument('--sele_str', default=None, type=str,
                    help='MDTraj selection string for the ligand / residues of '
                         'interest (e.g. "resname UNK"). Default: None.')
parser.add_argument('--sim_nos', default=None, type=int, nargs='+',
                    help='Space-separated list of sub-simulation indices to '
                         'process. Default: all available.')

# --- Resampling ---
parser.add_argument('--n_resample', default=1000, type=int,
                    help='Frames to importance-resample per sub-simulation. '
                         'Default 1000.')
parser.add_argument('--output_cache_dir', default=None, type=str,
                    help='Directory to read/write cached distance matrices. '
                         'Default: <input_dir>/retro_cache/.')
parser.add_argument('--overwrite', action='store_true',
                    help='Recompute and overwrite cached matrices even if they '
                         'already exist on disk.')
parser.add_argument('--read_only', action='store_true',
                    help='Load matrices from cache but do not write new files.')

# --- Convergence thresholds ---
parser.add_argument('--frobenius_thresh', default=0.05, type=float,
                    help='Normalised Frobenius norm threshold. Default 0.05.')
parser.add_argument('--jsd_thresh', default=0.10, type=float,
                    help='Jensen-Shannon divergence threshold. Default 0.10.')
parser.add_argument('--max_equil_fraction', default=0.75, type=float,
                    help='Max fraction of simulation discardable as '
                         'equilibration before the check is considered '
                         'unreliable. Default 0.75.')
parser.add_argument('--minimum_fraction', default=0.25, type=float,
                    help='Minimum fraction of total sub-simulations required '
                         'before reporting convergence. Default 0.25.')

# --- getContacts ---
parser.add_argument('--getcontacts_script', default=None, type=str,
                    help='Path to get_dynamic_contacts.py. Required for '
                         'contact distance matrix computation.')
parser.add_argument('--conda_env', default=None, type=str,
                    help='Name of the conda environment containing getContacts. '
                         'The current environment is auto-discovered from '
                         'CONDA_PREFIX and its name is replaced with this value '
                         'to locate the interpreter. Required when '
                         '--getcontacts_python is not given.')
parser.add_argument('--getcontacts_python', default=None, type=str,
                    help='Explicit path to the Python interpreter for '
                         'getContacts. When provided, --conda_env is ignored.')

args = parser.parse_args()

# --- Resolve cache directory ---
output_cache_dir = args.output_cache_dir
if output_cache_dir is None:
    output_cache_dir = os.path.join(args.input_dir, 'retro_cache')

if not os.path.isdir(output_cache_dir):
    print(f"Creating cache directory at: {output_cache_dir}")
    os.makedirs(output_cache_dir, exist_ok=True)

# --- Build getContacts kwargs (shared by both methods) ---
getcontacts_kwargs = {}
if args.getcontacts_script is not None:
    getcontacts_kwargs = {
        'getcontacts_script': args.getcontacts_script,
        'conda_env':          args.conda_env,
        'getcontacts_python': args.getcontacts_python,
    }

# --- Run ---
from FultonMarket.FultonMarketAnalysis import FultonMarketAnalysis

analysis = FultonMarketAnalysis(input_dir=args.input_dir,
                                pdb=args.pdb,
                                sele_str=args.sele_str)

analysis.retro_analyze_all(
    n_resample=args.n_resample,
    sim_nos=args.sim_nos,
    overwrite=args.overwrite,
    read_only=args.read_only,
    output_cache_dir=output_cache_dir,
    **getcontacts_kwargs,
)

report, metrics = analysis.retro_convergence_report(
    n_resample=args.n_resample,
    sim_nos=args.sim_nos,
    read_only=args.read_only,
    output_cache_dir=output_cache_dir,
    frobenius_thresh=args.frobenius_thresh,
    jsd_thresh=args.jsd_thresh,
    max_equil_fraction=args.max_equil_fraction,
    minimum_fraction=args.minimum_fraction,
    **getcontacts_kwargs,
)
