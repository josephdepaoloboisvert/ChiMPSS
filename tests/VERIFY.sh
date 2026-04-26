#!/usr/bin/env bash
# Verification commands for chimpss phases.
# Run this on a Linux machine with the full conda env installed:
#   conda activate chimpss
#   pip install -e .[dev]
#   bash tests/VERIFY.sh
#
# Each section is labelled by phase.  Run all, or paste a single section.
set -e

# ── Phase 1: Package skeleton ──────────────────────────────────────────────
echo "=== Phase 1: package skeleton ==="
python -c "import chimpss; print('chimpss version:', chimpss.__version__)"
pip show chimpss | grep -E "^(Name|Version|Location)"
ruff check src/ tests/
echo "Phase 1 OK"

# ── Phase 2: Bridgeport migration ─────────────────────────────────────────
echo "=== Phase 2: Bridgeport ==="
python -c "from chimpss.bridgeport import Bridgeport; print('Bridgeport OK:', Bridgeport)"
python -c "from chimpss.bridgeport.protein_preparer import ProteinPreparer; print('ProteinPreparer OK')"
python -c "from chimpss.bridgeport.repair_protein import RepairProtein; print('RepairProtein OK')"
python -c "from chimpss.bridgeport.ligand import Ligand; print('Ligand OK')"
python -c "from chimpss.bridgeport.analogue import Analogue; print('Analogue OK')"
python -c "from chimpss.bridgeport.mutated_peptide import MutatedPeptide; print('MutatedPeptide OK')"
python -c "from chimpss.bridgeport.forcefield import ForceFieldHandler; print('ForceFieldHandler OK')"
python -c "from chimpss.bridgeport.openmm_joiner import Joiner; print('Joiner OK')"
python -c "from chimpss.bridgeport.minimizer import Minimizer; print('Minimizer OK')"
pytest tests/unit/test_bridgeport/ -v
# Backward-compat shim check
python -c "from Bridgeport import Bridgeport; print('shim OK')"
echo "Phase 2 OK"

# ── Phase 3: MotorRow migration ────────────────────────────────────────────
echo "=== Phase 3: MotorRow ==="
python -c "from chimpss.motorrow import MotorRow; print('MotorRow OK:', MotorRow)"
python -c "from chimpss.motorrow.utils import get_positions_from_pdb, restrain_atoms, unpack_infiles; print('motorrow.utils OK')"
pytest tests/unit/test_motorrow/ -v
# Backward-compat shim check
python -c "from MotorRow import MotorRow; print('shim OK')"
echo "Phase 3 OK"

# ── Phase 4: FultonMarket migration ───────────────────────────────────────
echo "=== Phase 4: FultonMarket ==="
python -c "from chimpss.fultonmarket import FultonMarket; print('FultonMarket OK:', FultonMarket)"
python -c "from chimpss.fultonmarket import Randolph; print('Randolph OK:', Randolph)"
python -c "from chimpss.fultonmarket import FultonMarketAnalysis; print('FultonMarketAnalysis OK:', FultonMarketAnalysis)"
python -c "from chimpss.fultonmarket import printf, geometric_distribution, build_sampler_states, truncate_ncdf, frobenius_norm, jsd_distance_matrices; print('fultonmarket.utils OK')"
python -c "from chimpss.fultonmarket.contact_network import ContactNetworkBuilder; print('ContactNetworkBuilder OK')"
python -c "from chimpss.fultonmarket import retro_convergence; print('retro_convergence OK')"
pytest tests/unit/test_fultonmarket/ -v
# Backward-compat shim checks
python -c "from FultonMarket.FultonMarketwithAnalyzer import FultonMarket; print('shim FultonMarketwithAnalyzer OK')"
python -c "from FultonMarket.RandolphwithAnalyzer import Randolph; print('shim RandolphwithAnalyzer OK')"
python -c "from FultonMarket.FultonMarketAnalysis import FultonMarketAnalysis; print('shim FultonMarketAnalysis OK')"
python -c "from FultonMarket.FultonMarketUtils import truncate_ncdf, frobenius_norm; print('shim FultonMarketUtils OK')"
python -c "from FultonMarket import FultonMarket, Randolph, FultonMarketAnalysis; print('shim package __init__ OK')"
echo "Phase 4 OK"

# ── Phase 5: Shared utilities ──────────────────────────────────────────────
echo "=== Phase 5: shared utilities ==="
python -c "import chimpss.shared; print('chimpss.shared importable')"
python -c "from chimpss.shared.io import ensure_exists, write_FASTA, change_resname, describe_system, describe_state, cif2pdb, remove_dummy_atoms, isolate_chains, slice_select; print('shared.io OK')"
python -c "from chimpss.shared.logging import timestamp, printf, unique_residues, report_chain_information; print('shared.logging OK')"
# Confirm migrated bridgeport.ligand uses only internal chimpss imports
python -c "from chimpss.bridgeport.ligand import Ligand; print('bridgeport.ligand OK')"
# Backward-compat shim check
python -c "from utils.ProteinPreparer import ProteinPreparer; print('shim utils.ProteinPreparer OK')"
echo "Phase 5 OK"

# ── Phase 6: Analysis module + CLI entry points ────────────────────────────
echo "=== Phase 6: analysis module + CLI ==="

# analysis module imports
python -c "import chimpss.analysis; print('chimpss.analysis importable')"
python -c "from chimpss.analysis.contacts import ContactNetworkBuilder; print('analysis.contacts OK')"
python -c "from chimpss.analysis.pdb_fetch import get, fetch_family_proteins, list_families, classify_method; print('analysis.pdb_fetch OK')"
python -c "from chimpss.analysis.distance_matrix import DistanceMatrix; print('analysis.distance_matrix OK')"
python -c "from chimpss.analysis.grid_potentials import IO_Grid, select_whole_residues; print('analysis.grid_potentials OK')"
python -c "from chimpss.analysis.gpcr_pca import Structure_Analyzer, fetch_all_parallel, build_bw_assignments, conservation_filter; print('analysis.gpcr_pca OK')"

# CLI modules importable
python -c "from chimpss.cli.motorrow import main; print('cli.motorrow OK')"
python -c "from chimpss.cli.fultonmarket import main; print('cli.fultonmarket OK')"
python -c "from chimpss.cli.fultonmarket_mpi import main; print('cli.fultonmarket_mpi OK')"
python -c "from chimpss.cli.retro_analysis import main; print('cli.retro_analysis OK')"
python -c "from chimpss.cli.recovery import main; print('cli.recovery OK')"
python -c "from chimpss.cli.fetch_pdbs import main; print('cli.fetch_pdbs OK')"
python -c "from chimpss.cli.generate_pca import main; print('cli.generate_pca OK')"
python -c "from chimpss.cli.project_pca import main; print('cli.project_pca OK')"

# Console scripts registered (requires pip install -e .)
chimpss-motorrow --help         > /dev/null && echo "chimpss-motorrow --help OK"
chimpss-fultonmarket --help     > /dev/null && echo "chimpss-fultonmarket --help OK"
chimpss-retro-analysis --help   > /dev/null && echo "chimpss-retro-analysis --help OK"
chimpss-fetch-pdbs --help       > /dev/null && echo "chimpss-fetch-pdbs --help OK"
chimpss-generate-pca --help     > /dev/null && echo "chimpss-generate-pca --help OK"
chimpss-project-pca --help      > /dev/null && echo "chimpss-project-pca --help OK"
chimpss-recovery --help         > /dev/null && echo "chimpss-recovery --help OK"

# Backward-compat shims
python -c "from gpcr_pca_utils import Structure_Analyzer; print('shim gpcr_pca_utils OK')"
python -c "from GetContactsHelper import ContactNetworkBuilder; print('shim GetContactsHelper OK')"
python -c "from DistanceMatrix.DistanceMatrix import DistanceMatrix; print('shim DistanceMatrix OK')"

echo "Phase 6 OK"

# ── Phase 7: Notebook audit, extraction, and reorganization ──────────────────
echo "=== Phase 7: notebook reorganization ==="

# 7a: plotting helpers importable from package
python -c "
from chimpss.fultonmarket import plot_convergence_metric, add_equil_metric_to_plot
from chimpss.fultonmarket.retro_convergence import _RETRO_COLOR_MAP
assert callable(plot_convergence_metric)
assert callable(add_equil_metric_to_plot)
assert {'torsion','alpha_carbon','contact'} == set(_RETRO_COLOR_MAP)
print('Phase 7a: retro plotting helpers OK')
"

# 7b: notebook directory layout
python -c "
import os, sys
checks = {
    'DIR notebooks/tutorials':    os.path.isdir('notebooks/tutorials'),
    'DIR notebooks/examples':     os.path.isdir('notebooks/examples'),
    'DIR notebooks/analysis':     os.path.isdir('notebooks/analysis'),
    'DIR notebooks/exploratory':  os.path.isdir('notebooks/exploratory'),
    'tutorials/07-System_Preparation.ipynb': os.path.isfile('notebooks/tutorials/07-System_Preparation.ipynb'),
    'examples/interactive.ipynb':            os.path.isfile('notebooks/examples/interactive.ipynb'),
    'examples/ANALOGUE_EXAMPLE.ipynb':       os.path.isfile('notebooks/examples/ANALOGUE_EXAMPLE.ipynb'),
    'examples/Build(BP).ipynb':              os.path.isfile('notebooks/examples/Build(BP).ipynb'),
    'analysis/Retro_Analysis.ipynb':         os.path.isfile('notebooks/analysis/Retro_Analysis.ipynb'),
    'analysis/PCA_of_GPCRs.ipynb':           os.path.isfile('notebooks/analysis/PCA_of_GPCRs.ipynb'),
    'analysis/NewDistanceMatrix.ipynb':      os.path.isfile('notebooks/analysis/NewDistanceMatrix.ipynb'),
    'exploratory/Minimize.ipynb':            os.path.isfile('notebooks/exploratory/Minimize.ipynb'),
    'exploratory/Docking_Test.ipynb':        os.path.isfile('notebooks/exploratory/Docking_Test.ipynb'),
    'DELETED DistanceMatrix/Untitled.ipynb': not os.path.isfile('DistanceMatrix/Untitled.ipynb'),
    'DELETED root Minimize.ipynb':           not os.path.isfile('Minimize.ipynb'),
}
failures = [k for k, v in checks.items() if not v]
if failures:
    print('LAYOUT FAILURES:', failures); sys.exit(1)
print('Phase 7b: notebook layout OK')
"

# 7a/D: no old import patterns remain in reorganized notebooks
python -c "
import json, glob, sys
OLD = [
    'from Bridgeport.Bridgeport import',
    'from MotorRow import MotorRow',
    'from FultonMarket.FultonMarket import',
    'from FultonMarket.FultonMarketAnalysis import',
    'from FultonMarket.analysis.',
    'from DistanceMatrix.DistanceMatrix import',
    'from utility.Reporting import',
    'from utility.General import',
    'from utility.FileManipulations import',
    'from RepairProtein.RepairProtein import',
]
failures = []
for path in glob.glob('notebooks/**/*.ipynb', recursive=True):
    try:
        nb = json.load(open(path, encoding='utf-8', errors='replace'))
        for i, cell in enumerate(nb.get('cells',[])):
            if cell.get('cell_type') == 'code':
                src = ''.join(cell.get('source', []))
                for pat in OLD:
                    if pat in src:
                        failures.append(f'{path} cell {i}: \"{pat}\"')
    except Exception:
        pass
if failures:
    print('OLD IMPORTS FOUND:'); [print(' ', f) for f in failures]; sys.exit(1)
print('Phase 7a/D: import cleanup OK')
"

pytest tests/unit/test_fultonmarket/test_retro_plotting.py -v

echo "Phase 7 OK"

# ── Phase 8: Test suite + CI ──────────────────────────────────────────────────
echo "=== Phase 8: test suite ==="

# 8a: pure-logic unit tests (no MD stack needed — numpy only)
pytest tests/unit/test_shared/test_logging.py -v
pytest tests/unit/test_fultonmarket/test_retro_logic.py -v

# 8b: shared.io tests (requires mdtraj)
pytest tests/unit/test_shared/test_io.py -v

# 8c: fultonmarket math tests (requires openmm)
pytest tests/unit/test_fultonmarket/test_math.py -v

# 8d: fultonmarket plotting tests (requires openmm + matplotlib)
pytest tests/unit/test_fultonmarket/test_retro_plotting.py -v

# 8e: analysis pdb_fetch tests (requires requests)
pytest tests/unit/test_analysis/test_pdb_fetch.py -v

# 8f: analysis gpcr_pca tests (requires MDAnalysis)
pytest tests/unit/test_analysis/test_gpcr_pca.py -v

# 8g: full unit test suite, all markers except slow/gpu
pytest tests/unit/ -v -m "not slow and not gpu"

echo "Phase 8 OK"

# ── Phase 9: File naming, path handling, and existence checks ─────────────────
echo "=== Phase 9: file naming + path utilities ==="

# 9a: shared.io new utilities
python -c "
from chimpss.shared.io import validate_name, build_output_path, file_exists_skip

# validate_name: accept clean names
assert validate_name('5HT2B') == '5HT2B'
assert validate_name('LSD') == 'LSD'

# validate_name: reject names with _ or .
for bad in ['bad_name', 'bad.name', 'a_b.c']:
    try:
        validate_name(bad)
        assert False, f'should have raised for {bad!r}'
    except ValueError:
        pass

# build_output_path: correct filename format
import os
p = build_output_path('/tmp', '5HT2B', 'LSD', 'topology', 'pdb')
assert p == '/tmp/5HT2B_LSD.topology.pdb', p
p2 = build_output_path('/tmp', 'ADRB2', 'isoproterenol', 'trajectory', 'ncdf')
assert p2 == '/tmp/ADRB2_isoproterenol.trajectory.ncdf', p2

# file_exists_skip: returns False for non-existent, True for existing
import tempfile, os
assert file_exists_skip('/nonexistent/path/xyz.pdb', 'test') is False
with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
    tmp = f.name
try:
    assert file_exists_skip(tmp, 'test') is True
finally:
    os.unlink(tmp)

print('Phase 9a: shared.io utilities OK')
"

# 9b: import smoke tests for all three stage classes
python -c "from chimpss.bridgeport import Bridgeport; print('Bridgeport import OK')"
python -c "from chimpss.motorrow import MotorRow; print('MotorRow import OK')"
python -c "from chimpss.fultonmarket import FultonMarket; print('FultonMarket import OK')"

# 9c: CLI help includes new flags
chimpss-motorrow --help | grep -q "protein-name" && echo "chimpss-motorrow --protein-name flag OK"
chimpss-motorrow --help | grep -q "ligand-name"  && echo "chimpss-motorrow --ligand-name flag OK"
chimpss-fultonmarket --help | grep -q "protein-name" && echo "chimpss-fultonmarket --protein-name flag OK"
chimpss-fultonmarket --help | grep -q "ligand-name"  && echo "chimpss-fultonmarket --ligand-name flag OK"

# 9d: no notebook still uses the old two-key Protein dict
python -c "
import json, glob, sys
failures = []
for path in glob.glob('notebooks/**/*.ipynb', recursive=True):
    try:
        nb = json.load(open(path, encoding='utf-8', errors='replace'))
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                src = ''.join(cell.get('source', []))
                if 'input_pdb_dir' in src and \"json_dict['Protein']\" in src:
                    failures.append(f'{path} cell {i}')
    except Exception:
        pass
if failures:
    print('OLD Protein dict pattern found:'); [print(' ', f) for f in failures]; sys.exit(1)
print('Phase 9d: no old Protein dict schema in notebooks OK')
"

echo "Phase 9 OK"
