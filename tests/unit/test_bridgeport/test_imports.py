"""Smoke tests: verify that chimpss.bridgeport submodules are importable.

These tests require the full MD stack (openmm, mdtraj, MDAnalysis, etc.).
They are skipped automatically when that stack is not installed.

Run with:  pytest tests/unit/test_bridgeport/
"""
import importlib
import importlib.util
import pytest

# Skip entire module if MD stack is absent (CI runs with --no-deps)
pytest.importorskip("mdtraj", reason="MD stack not installed — skipping bridgeport import tests")


BRIDGEPORT_MODULES = [
    "chimpss.bridgeport",
    "chimpss.bridgeport.bridgeport",
    "chimpss.bridgeport.protein_preparer",
    "chimpss.bridgeport.repair_protein",
    "chimpss.bridgeport.ligand",
    "chimpss.bridgeport.analogue",
    "chimpss.bridgeport.mutated_peptide",
    "chimpss.bridgeport.forcefield",
    "chimpss.bridgeport.openmm_joiner",
    "chimpss.bridgeport.minimizer",
    "chimpss.bridgeport.minimizer_utils",
    "chimpss.bridgeport.ligand_utils",
    "chimpss.bridgeport._utils",
]


@pytest.mark.parametrize("module", BRIDGEPORT_MODULES)
def test_module_importable(module):
    """Each bridgeport submodule must import without raising."""
    importlib.import_module(module)


def test_bridgeport_class_exported():
    from chimpss.bridgeport import Bridgeport
    assert callable(Bridgeport)
