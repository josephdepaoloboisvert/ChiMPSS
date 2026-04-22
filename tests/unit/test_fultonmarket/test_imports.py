"""
Smoke tests for chimpss.fultonmarket — verifies importability of all
submodules and the shim backward-compat layer.

Run on Linux with the full conda env:
    pytest tests/unit/test_fultonmarket/ -v
"""
import pytest

openmm = pytest.importorskip("openmm", reason="openmm not installed")
mdtraj = pytest.importorskip("mdtraj", reason="mdtraj not installed")


def test_fultonmarket_package_importable():
    import chimpss.fultonmarket


def test_fulton_market_class_importable():
    from chimpss.fultonmarket import FultonMarket
    assert callable(FultonMarket)


def test_randolph_class_importable():
    from chimpss.fultonmarket import Randolph
    assert callable(Randolph)


def test_analysis_class_importable():
    from chimpss.fultonmarket import FultonMarketAnalysis
    assert callable(FultonMarketAnalysis)


def test_utils_importable():
    from chimpss.fultonmarket import (
        printf,
        geometric_distribution,
        build_sampler_states,
        truncate_ncdf,
        frobenius_norm,
        jsd_distance_matrices,
    )
    assert callable(printf)
    assert callable(geometric_distribution)


def test_contact_network_importable():
    from chimpss.fultonmarket.contact_network import ContactNetworkBuilder
    assert callable(ContactNetworkBuilder)


def test_retro_convergence_importable():
    from chimpss.fultonmarket import retro_convergence
    assert hasattr(retro_convergence, 'resolve_write_dir')


# ---------------------------------------------------------------------------
# Shim backward-compat
# ---------------------------------------------------------------------------

def test_shim_fultonmarketwithanalyzer():
    from FultonMarket.FultonMarketwithAnalyzer import FultonMarket
    assert callable(FultonMarket)


def test_shim_randolphwithanalyzer():
    from FultonMarket.RandolphwithAnalyzer import Randolph
    assert callable(Randolph)


def test_shim_fultonmarketanalysis():
    from FultonMarket.FultonMarketAnalysis import FultonMarketAnalysis
    assert callable(FultonMarketAnalysis)


def test_shim_fultonmarketutils():
    from FultonMarket.FultonMarketUtils import truncate_ncdf, frobenius_norm
    assert callable(truncate_ncdf)
    assert callable(frobenius_norm)


def test_shim_package_init():
    from FultonMarket import FultonMarket, Randolph, FultonMarketAnalysis
    assert callable(FultonMarket)
    assert callable(Randolph)
    assert callable(FultonMarketAnalysis)
