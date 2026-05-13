"""
Integration tests for the Bridgeport pipeline using test_case_1 (NTSR1 / ML-301).

These tests verify that the full Bridgeport pipeline still works with the real
test inputs that were used before the repository reorganization.

  Fast tests  (no markers)    — run on every `pytest` invocation
  @pytest.mark.slow            — full pipeline run, use `pytest -m slow`

All tests share a fixture that writes a properly-patched JSON config pointing
to the local test data, so paths are never hardcoded.
"""
import json
import os

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_DATA = os.path.join(REPO_ROOT, "test_data", "test_case_1")


def _abspath(filename):
    return os.path.join(TEST_DATA, filename)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bridgeport_config(tmp_path):
    """Write a Bridgeport JSON config that points at local test data."""
    modeller_dir = str(tmp_path / "modeller_intermediates")
    os.makedirs(modeller_dir, exist_ok=True)

    config = {
        "working_dir": str(tmp_path),
        "protein_name": "NTSR1",
        "Protein": {
            "input_pdb": _abspath("6za8-pdb-bundle1.pdb"),
            "chain": "A",
        },
        "Ligand": {
            "name": "RTI-3a",
            "resname": "SR5",
            "chainid": False,
            "smiles": "COC1=CC=CC(OC)=C1C1=CC(=NN1C1=CC=NC2=C1C=CC(Cl)=C2)C(=O)N[C@@H](CC(C)C)C(O)=O",
            "Analogue": {
                "name": "ML-301",
                "smiles": "COC1=CC=CC(OC)=C1C1=NC(=CN1C1=C2C=CC(Cl)=CC2=NC=C1)C(=O)N[C@H](CC(C)C)C([O-])=O",
                "add_atoms": [
                    [10, 13], [6, 15], [26, 26], [27, 27], [28, 28],
                    [29, 29], [30, 33], [34, 30], [36, 32], [35, 31],
                ],
                "remove_atoms": False,
                "align_all": True,
                "rmsd_thresh": 1,
                "n_conformers": 1,
            },
            "small_molecule_params": True,
            "sanitize": True,
            "removeHs": True,
            "proximityBonding": True,
            "nstd_resids": None,
            "pH": 7.0,
            "neutral_Cterm": False,
        },
        "Environment": {
            "alignment_ref": _abspath("6z66_OPM.pdb"),
            "reference_chain": ["A"],
            "membrane": True,
            "pH": 7.2,
            "ion_strength": 0.15,
        },
        "RepairProtein": {
            "fasta_path": _abspath("NTSR1.fasta"),
            "working_dir": modeller_dir,
            "tails": [48, 390],
            "loops": [],
            "secondary_template": _abspath("AF-P30989-F1-model_v4.pdb"),
            "engineered_resids": None,
        },
    }

    config_path = str(tmp_path / "ML-301_local.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ---------------------------------------------------------------------------
# Fast smoke tests — always run
# ---------------------------------------------------------------------------

def test_test_data_files_present():
    """All expected input files exist in test_data/test_case_1."""
    required = [
        "6za8-pdb-bundle1.pdb",
        "6z66_OPM.pdb",
        "AF-P30989-F1-model_v4.pdb",
        "NTSR1.fasta",
        "ML-301.json",
    ]
    for fname in required:
        assert os.path.exists(_abspath(fname)), f"Missing test input: {fname}"


def test_bridgeport_init(bridgeport_config):
    """Bridgeport initialises correctly from the local config (no heavy work)."""
    from chimpss.bridgeport import Bridgeport

    bp = Bridgeport(bridgeport_config)

    assert bp.protein_name == "NTSR1"
    assert bp.ligand_name == "ML-301"
    assert bp.type == "small_molecule"
    assert bp.chain == "A"
    assert bp.resname == "SR5"
    assert os.path.exists(bp.input_pdb)


def test_bridgeport_alignment(bridgeport_config):
    """Alignment step completes and produces an aligned PDB."""
    from chimpss.bridgeport import Bridgeport

    bp = Bridgeport(bridgeport_config)
    bp.align_to_reference()

    assert hasattr(bp, "aligned_pdb"), "align_to_reference should set bp.aligned_pdb"
    assert os.path.exists(bp.aligned_pdb), f"Expected aligned PDB at {bp.aligned_pdb}"


# ---------------------------------------------------------------------------
# Full pipeline — slow, requires MODELLER + OpenFF + pdbfixer
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bridgeport_full_run(bridgeport_config, tmp_path):
    """
    End-to-end Bridgeport run: align → MCS → repair → solvate → parameterise → build system.

    Asserts that the final topology PDB and system XML are produced.
    Requires: MODELLER, OpenFF, pdbfixer, openmm.
    Runtime: ~30–60 minutes on a workstation.
    """
    from chimpss.bridgeport import Bridgeport

    bp = Bridgeport(bridgeport_config)
    bp.run()

    assert os.path.exists(bp.final_pdb),     f"Expected topology PDB at {bp.final_pdb}"
    assert os.path.exists(bp.final_xml),     f"Expected system XML at {bp.final_xml}"
    assert os.path.exists(bp.final_xml_fg),  f"Expected FG system XML at {bp.final_xml_fg}"
    assert os.path.exists(bp.final_xml_hmr), f"Expected FG+HMR system XML at {bp.final_xml_hmr}"
    for path in (bp.final_pdb, bp.final_xml, bp.final_xml_fg, bp.final_xml_hmr):
        assert os.path.getsize(path) > 0, f"Output file is empty: {path}"
