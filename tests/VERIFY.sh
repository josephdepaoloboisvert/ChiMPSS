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
