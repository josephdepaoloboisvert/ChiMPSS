"""
CLI entry point for recovering a crashed FultonMarket simulation.

Run this when a simulation is interrupted mid-block; it truncates the output
NCDF to the last complete checkpoint and writes the block's state arrays so
that chimpss-fultonmarket can resume.

Console script: chimpss-recovery
"""

import argparse
import glob
import os
import random
import string

import numpy as np
from openmmtools.multistate import MultiStateReporter

from chimpss.fultonmarket.fulton_market import REQUIRED_BLOCK_ARRAYS


def parse_args():
    p = argparse.ArgumentParser(
        description='Recover a crashed FultonMarket simulation by truncating '
                    'the output NCDF to the last complete checkpoint.')
    p.add_argument('output_dir',
                   help='Path to the replica exchange directory of the '
                        'simulation to recover '
                        '(e.g. /path/to/runs/NTSR1_ML-301.hmr.rep0)')
    p.add_argument('--ncdf', default=None,
                   help='Trajectory NCDF to truncate. Defaults to the single '
                        '*.trajectory.ncdf in output_dir. FultonMarket names it '
                        'PROTEIN_LIGAND.trajectory.ncdf from --protein-name and '
                        '--ligand-name, so there is no fixed filename to assume.')
    return p.parse_args()


def find_trajectory_ncdf(output_dir, explicit=None):
    """Locate the production trajectory NCDF.

    The filename is derived from --protein-name/--ligand-name via
    ``build_output_path``, so it is PROTEIN_LIGAND.trajectory.ncdf and differs
    per run.  The checkpoint companion is excluded deliberately: truncating it
    instead of the trajectory silently produces an unusable block.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f'--ncdf {explicit} does not exist')
        return explicit

    candidates = [p for p in glob.glob(os.path.join(output_dir, '*.trajectory.ncdf'))
                  if not p.endswith('trajectory_checkpoint.ncdf')]
    if not candidates:
        raise FileNotFoundError(
            f'no *.trajectory.ncdf found in {output_dir}. FultonMarket derives the '
            f'name from --protein-name and --ligand-name; pass --ncdf explicitly.')
    if len(candidates) > 1:
        raise RuntimeError(
            f'several trajectory NCDFs in {output_dir}: '
            f'{[os.path.basename(c) for c in candidates]}. Pass --ncdf to choose.')
    return candidates[0]


def latest_block_dir(save_dir):
    """Highest-numbered block directory, counting directories only.

    ``len(os.listdir(...))`` would count stray files as blocks and point the
    recovery at the wrong index.
    """
    blocks = sorted((d for d in os.listdir(save_dir)
                     if d.isdigit() and os.path.isdir(os.path.join(save_dir, d))),
                    key=int)
    if not blocks:
        raise RuntimeError(f'no numbered block directories in {save_dir}')
    return os.path.join(save_dir, blocks[-1])


def main():
    args = parse_args()
    from chimpss.fultonmarket.utils import truncate_ncdf

    output_dir = args.output_dir
    save_dir = os.path.join(output_dir, 'saved_variables')
    sub_sim_save_dir = latest_block_dir(save_dir)

    ncdf_fn = find_trajectory_ncdf(output_dir, args.ncdf)
    print(f'Truncating {os.path.basename(ncdf_fn)} into {sub_sim_save_dir}', flush=True)

    random_string = ''.join(random.choice(string.ascii_letters + string.digits)
                            for _ in range(6))
    tmp = f'{random_string}.ncdf'

    reporter = MultiStateReporter(ncdf_fn)
    reporter.open()

    pos, velos, box_vecs, states, energies, temps = truncate_ncdf(
        ncdf_fn, tmp, sub_sim_save_dir, reporter)

    if energies.shape[0] <= 10:
        raise RuntimeError(
            f'{os.path.basename(ncdf_fn)} holds too few frames to recover '
            f'({energies.shape[0]}). Delete it and resume the simulation instead.')

    # positions.npy is one of the six arrays _configure_experiment_parameters
    # requires; omitting it leaves a block that aborts the next resume with
    # "Incomplete save directories detected".
    for name, array in (('positions.npy', pos),
                        ('velocities.npy', velos),
                        ('box_vectors.npy', box_vecs),
                        ('states.npy', states),
                        ('energies.npy', energies),
                        ('temperatures.npy', temps)):
        np.save(os.path.join(sub_sim_save_dir, name), getattr(array, 'data', array))
        print(f'Saved {name}', flush=True)

    missing = [f for f in REQUIRED_BLOCK_ARRAYS
               if not os.path.exists(os.path.join(sub_sim_save_dir, f))]
    if missing:
        raise RuntimeError(
            f'{sub_sim_save_dir} is still missing {missing} after recovery; '
            f'the run will not resume from it.')
    print(f'Block {os.path.basename(sub_sim_save_dir)} is complete and resumable.',
          flush=True)

    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == '__main__':
    main()
