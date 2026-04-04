#!/usr/bin/env python
"""
Recover a FultonMarket simulation after an out-of-memory error.

Truncates the output netCDF and saves numpy checkpoints so that
run_fulton_market.py can resume from the last good frame.
"""
import argparse
import os
import random
import string

import numpy as np
import netCDF4 as nc
from openmmtools.multistate import MultiStateReporter
from FultonMarket.FultonMarketUtils import truncate_ncdf


def generate_random_string(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recover a FultonMarket simulation after an out-of-memory error.'
    )
    parser.add_argument('output_dir', type=str,
                        help='Path to the replica exchange directory '
                             '(e.g. /path/to/replica_exchange/SIMNAME_REP)')
    args = parser.parse_args()

    random_string = generate_random_string()

    # Set output directory
    output_dir = args.output_dir
    save_dir = os.path.join(output_dir, 'saved_variables')
    sub_sim_save_dir = os.path.join(save_dir, str(len(os.listdir(save_dir))-1))

    # Create reporter
    ncdf_fn = os.path.join(output_dir, 'output.ncdf')
    reporter = MultiStateReporter(ncdf_fn)
    reporter.open()

    # Truncate
    pos, velos, box_vecs, states, energies, temps = truncate_ncdf(ncdf_fn, f'{random_string}.ncdf', sub_sim_save_dir, reporter)

    # Save
    assert energies.shape[0] > 10, 'this output.ncdf does not have data, please delete and resume simulation.'
    print('Trying to save to', sub_sim_save_dir, flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'velocities.npy'), velos.data)
    print('Saved velos', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'box_vectors.npy'), box_vecs.data)
    print('Saved box_vectors', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'states.npy'), states.data)
    print('Saved states', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'energies.npy'), energies.data)
    print('Saved energies', flush=True)
    np.save(os.path.join(sub_sim_save_dir, 'temperatures.npy'), temps.data)
    print('Saved temps', flush=True)

    if os.path.exists(f'{random_string}.ncdf'):
        os.remove(f'{random_string}.ncdf')
