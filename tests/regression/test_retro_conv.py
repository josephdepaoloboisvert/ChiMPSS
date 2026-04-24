"""
Regression tests for FultonMarketAnalysis.retro_analyze_all and
retro_convergence_report (replaces the root-level test_retro_conv.py).

These tests are slow and require:
  - openmm, openmmtools, mdtraj, pymbar
  - The getcontacts script (or a compatible conda env)
  - Real simulation output directories

Configure via environment variables (see parametrize below), or supply a
custom pytest.ini section. Tests are skipped when the expected paths are absent.

Run with:
    pytest tests/regression/test_retro_conv.py -m slow -v \
        --sim-dir /path/to/sim/output \
        --pdb /path/to/receptor.pdb
"""
import os

import pytest

openmm = pytest.importorskip("openmm",  reason="openmm not installed")
mdtraj = pytest.importorskip("mdtraj",  reason="mdtraj not installed")
pymbar = pytest.importorskip("pymbar",  reason="pymbar not installed")


# ── CLI options injected by conftest ─────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--sim-dir",  default=None, help="Path to FultonMarket output directory")
    parser.addoption("--pdb",      default=None, help="Path to input PDB for retro analysis")
    parser.addoption("--sele-str", default="resname UNK", help="MDTraj selection string for ligand")
    parser.addoption(
        "--cache-dir", default=None,
        help="Optional cache directory for retro_analyze_all intermediate files",
    )
    parser.addoption(
        "--getcontacts-script", default=None,
        help="Path to get_dynamic_contacts.py",
    )
    parser.addoption("--conda-env",         default=None)
    parser.addoption("--getcontacts-python", default=None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_path(request, option, label):
    p = request.config.getoption(option)
    if p is None or not os.path.exists(p):
        pytest.skip(f"--{option} not set or path does not exist ({label})")
    return p


# ── retro_analyze_all ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_retro_analyze_all(request, tmp_path):
    """
    Run retro_analyze_all on a real simulation output directory.

    Writes intermediate files to a temp cache dir unless --cache-dir is given.
    """
    from chimpss.fultonmarket import FultonMarketAnalysis

    sim_dir  = _require_path(request, "sim-dir", "simulation output directory")
    pdb      = _require_path(request, "pdb",     "input PDB file")
    sele_str = request.config.getoption("--sele-str")
    cache_dir = request.config.getoption("--cache-dir") or str(tmp_path / "retro_cache")

    extra = {}
    for opt, key in [
        ("--getcontacts-script", "getcontacts_script"),
        ("--conda-env",          "conda_env"),
        ("--getcontacts-python", "getcontacts_python"),
    ]:
        val = request.config.getoption(opt)
        if val:
            extra[key] = val

    analysis = FultonMarketAnalysis(
        input_dir=sim_dir,
        pdb=pdb,
        sele_str=sele_str,
    )
    matrices = analysis.retro_analyze_all(
        n_resample=500,
        output_cache_dir=cache_dir,
        **extra,
    )
    assert isinstance(matrices, dict)
    assert len(matrices) > 0


# ── retro_convergence_report ──────────────────────────────────────────────────

@pytest.mark.slow
def test_retro_convergence_report(request, tmp_path):
    """
    Run retro_convergence_report in read-only mode on a pre-built cache.
    Requires --cache-dir pointing to an existing retro cache.
    """
    from chimpss.fultonmarket import FultonMarketAnalysis

    sim_dir  = _require_path(request, "sim-dir", "simulation output directory")
    pdb      = _require_path(request, "pdb",     "input PDB file")
    sele_str = request.config.getoption("--sele-str")
    cache_dir = request.config.getoption("--cache-dir")
    if cache_dir is None:
        pytest.skip("--cache-dir required for retro_convergence_report regression test")

    analysis = FultonMarketAnalysis(
        input_dir=sim_dir,
        pdb=pdb,
        sele_str=sele_str,
    )
    report, metrics = analysis.retro_convergence_report(
        total_sim_time=1000,
        sim_length=50,
        read_only=True,
        n_resample=500,
        output_cache_dir=cache_dir,
    )
    assert isinstance(report, dict)
    assert isinstance(metrics, dict)
