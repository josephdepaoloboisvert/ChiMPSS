#!/usr/bin/env python3
"""
MotorRow stability test:
  Run A -- default system.xml, dt=2.0 fs
  Runs B1-B5 -- FG_HMR_system.xml, dt = 3.0 / 3.25 / 3.5 / 3.75 / 4.0 fs

Usage:
  python run_motorrow.py --systems <dir> --out <dir>

Results are appended to <out>/motorrow_results.txt
"""
import argparse
import os
import sys
import traceback
from datetime import datetime

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--systems',
                    default='/tmp/pytest-of-exouser/pytest-12/test_bridgeport_full_run0/systems',
                    help='Directory containing NTSR1_ML-301 topology/system XML files')
parser.add_argument('--out', default='/media/volume/Josephs-Volume/ChiMPSS_Testing',
                    help='Root directory for all run output (default: %(default)s)')
args = parser.parse_args()

SYSTEMS_DIR = args.systems
PDB = os.path.join(SYSTEMS_DIR, 'NTSR1_ML-301.topology.pdb')
XML = os.path.join(SYSTEMS_DIR, 'NTSR1_ML-301.system.xml')
HMR = os.path.join(SYSTEMS_DIR, 'NTSR1_ML-301.FG_HMR_system.xml')

OUT_BASE = args.out
RESULTS  = os.path.join(OUT_BASE, 'motorrow_results.txt')
os.makedirs(OUT_BASE, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(RESULTS, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_motorrow(xml, work_dir, dt, label):
    from chimpss.motorrow import MotorRow
    os.makedirs(work_dir, exist_ok=True)
    log(f'START {label}  dt={dt} fs  dir={work_dir}')
    mr = MotorRow(
        PDB, xml, work_dir,
        lig_resname='UNK',
        protein_name='NTSR1',
        ligand_name='ML-301',
    )
    state_fn, pdb_fn = mr.main(PDB, dt=dt)
    log(f'DONE  {label}  -> {pdb_fn}')
    return True


# Run A: default system
log('=' * 70)
log('RUN A: default system.xml, dt=2.0 fs')
log('=' * 70)
try:
    run_motorrow(XML, os.path.join(OUT_BASE, 'motorrow_default'), dt=2.0, label='Run-A')
except Exception:
    log('Run-A FAILED')
    tb = traceback.format_exc()
    print(tb, flush=True)
    with open(RESULTS, 'a', encoding='utf-8') as f:
        f.write(tb + '\n')

# Runs B: FG+HMR system, increasing dt
log('=' * 70)
log('RUNS B: FG_HMR_system.xml, dt = 3.0 / 3.25 / 3.5 / 3.75 / 4.0 fs')
log('=' * 70)

last_good_dt = None
for dt in [3.0, 3.25, 3.5, 3.75, 4.0]:
    label = f'Run-B dt={dt}'
    work  = os.path.join(OUT_BASE, f'motorrow_hmr_{str(dt).replace(".", "p")}')
    try:
        run_motorrow(HMR, work, dt=dt, label=label)
        last_good_dt = dt
    except Exception:
        log(f'{label} FAILED - stopping HMR sweep')
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open(RESULTS, 'a', encoding='utf-8') as f:
            f.write(tb + '\n')
        break

log('=' * 70)
if last_good_dt is not None:
    log(f'RESULT: highest stable HMR timestep = {last_good_dt} fs')
else:
    log('RESULT: no HMR timestep succeeded')
log('=' * 70)
