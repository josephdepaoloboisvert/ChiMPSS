"""Smoke tests: verify that chimpss.motorrow submodules are importable.

These tests require the full MD stack (openmm, mdtraj, MDAnalysis, etc.).
They are skipped automatically when that stack is not installed.

Run with:  pytest tests/unit/test_motorrow/
"""
import importlib
import pytest

pytest.importorskip("openmm", reason="MD stack not installed — skipping motorrow import tests")


MOTORROW_MODULES = [
    "chimpss.motorrow",
    "chimpss.motorrow.motorrow",
    "chimpss.motorrow.utils",
]


@pytest.mark.parametrize("module", MOTORROW_MODULES)
def test_module_importable(module):
    """Each motorrow submodule must import without raising."""
    importlib.import_module(module)


def test_motorrow_class_exported():
    from chimpss.motorrow import MotorRow
    assert callable(MotorRow)


def test_shim_importable():
    """Root-level MotorRow.py shim must re-export MotorRow."""
    from MotorRow import MotorRow
    assert callable(MotorRow)
