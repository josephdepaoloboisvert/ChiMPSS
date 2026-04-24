"""
CLI entry point for retroactive FultonMarket distance-matrix analysis.

Console script: chimpss-retro-analysis
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description='Retroactively compute distance matrices and convergence report '
                    'for a completed FultonMarket simulation.')
    p.add_argument('input_dir', type=str,
                   help='Path to the FultonMarket output directory '
                        '(contains netcdfs and saved_variables dir).')
    p.add_argument('pdb', type=str,
                   help='Path to the topology PDB used for the simulation.')
    p.add_argument('--output_cache_dir', default=None, type=str,
                   help='Directory to read/write cached distance matrices. '
                        'Default: <input_dir>/retro_cache/.')
    p.add_argument('--n_resample', default=1000, type=int,
                   help='Frames to importance-resample per sub-simulation. Default 1000.')
    p.add_argument('--sele_str', default='resname UNK', type=str,
                   help='MDAnalysis selection string for the ligand. Default: "resname UNK".')
    p.add_argument('--getcontacts_script', default=None, type=str,
                   help='Path to get_dynamic_contacts.py.')
    p.add_argument('--conda_env', default=None, type=str,
                   help='Name of the conda environment containing getContacts.')
    p.add_argument('--getcontacts_python', default=None, type=str,
                   help='Explicit path to the Python interpreter for getContacts. '
                        'When provided, --conda_env is ignored.')
    return p.parse_args()


def main():
    args = parse_args()

    output_cache_dir = args.output_cache_dir
    if output_cache_dir is None:
        output_cache_dir = os.path.join(args.input_dir, 'retro_cache')

    os.makedirs(output_cache_dir, exist_ok=True)

    from chimpss.fultonmarket import FultonMarketAnalysis

    analyzer = FultonMarketAnalysis(
        input_dir=args.input_dir,
        pdb=args.pdb,
        sele_str=args.sele_str)

    getcontacts_python = args.getcontacts_python
    if getcontacts_python is None and args.conda_env is not None:
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            getcontacts_python = os.path.join(
                conda_prefix.replace(
                    os.path.basename(conda_prefix), args.conda_env),
                'bin', 'python')

    analyzer.retro_analyze_all(
        n_resample=args.n_resample,
        output_cache_dir=output_cache_dir,
        getcontacts_script=args.getcontacts_script,
        conda_env=args.conda_env,
        getcontacts_python=getcontacts_python)


if __name__ == '__main__':
    main()
