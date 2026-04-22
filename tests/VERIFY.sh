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
