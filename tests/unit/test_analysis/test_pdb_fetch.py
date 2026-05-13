"""
Unit tests for chimpss.analysis.pdb_fetch.

classify_method is pure logic (no network calls). The rest of the module
makes live HTTP requests and is not tested here.

pdb_fetch.py imports requests at module level, so the whole file is gated.
"""
import pytest

pytest.importorskip("requests", reason="requests not installed — skipping pdb_fetch tests")

from chimpss.analysis.pdb_fetch import GPCRDB, classify_method

# ── classify_method ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ({"structure_type": "X-ray Crystallography"}, "xray"),
    ({"structure_type": "X-RAY DIFFRACTION"}, "xray"),
    ({"method": "x-ray"}, "xray"),
    ({"structure_type": "Electron Microscopy"}, "em"),
    ({"structure_type": "cryo-EM"}, "em"),
    ({"method": "Cryo-EM"}, "em"),
    ({"structure_type": "electron"}, "em"),
    ({"structure_type": "Solution NMR"}, "other"),
    ({"structure_type": ""}, "other"),
    ({}, "other"),
])
def test_classify_method(raw, expected):
    assert classify_method(raw) == expected


def test_classify_method_prefers_structure_type_over_method():
    structure = {"structure_type": "X-ray Crystallography", "method": "electron"}
    assert classify_method(structure) == "xray"


def test_gpcrdb_constant():
    assert GPCRDB.startswith("https://")
