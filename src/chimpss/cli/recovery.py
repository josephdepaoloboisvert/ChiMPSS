"""
CLI entry point for recovering a crashed FultonMarket simulation.

Run this when an OOM error interrupts a simulation; it truncates the output
NCDF to the last complete checkpoint so that chimpss-fultonmarket can resume.

Console script: chimpss-recovery
"""

import os
import argparse
import random
import string

import numpy as np
import netCDF4 as nc
from openmmtools.multistate import MultiStateReporter


def parse_args():
    p = argparse.ArgumentParser(
        description='Recover a crashed FultonMarket simulation by truncating '
                    'the output NCDF to the last complete checkpoint.')
    p.add_argument('output_dir',
                   help='Path to the replica exchange directory of the '
                        'simulation to recover '
                        '(e.g. /path/to/replica_exchange/SIMNAME_REP)')
    return p.parse_args()


def main():
    args = parse_args()
    from chimpss.fultonmarket.utils import truncate_ncdf

    output_dir       = args.output_dir
    save_dir         = os.path.join(output_dir, 'saved_variables')
    sub_sim_save_dir = os.path.join(save_dir, str(len(os.listdir(save_dir)) - 1))

    random_string = ''.join(random.choice(string.ascii_letters + string.digits)
                            for _ in range(6))

    ncdf_fn  = os.path.join(output_dir, 'output.ncdf')
    reporter = MultiStateReporter(ncdf_fn)
    reporter.open()

    pos, velos, box_vecs, states, energies, temps = truncate_ncdf(
        ncdf_fn, f'{random_string}.ncdf', sub_sim_save_dir, reporter)

    assert energies.shape[0] > 10, (
        'output.ncdf does not have data — please delete it and resume the simulation.')

    print(f'Saving to {sub_sim_save_dir}', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'velocities.npy'),   velos.data)
    print('Saved velocities', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'box_vectors.npy'),  box_vecs.data)
    print('Saved box_vectors', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'states.npy'),       states.data)
    print('Saved states', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'energies.npy'),     energies.data)
    print('Saved energies', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'temperatures.npy'), temps.data)
    print('Saved temperatures', flush=True)

    tmp = f'{random_string}.ncdf'
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == '__main__':
    main()
