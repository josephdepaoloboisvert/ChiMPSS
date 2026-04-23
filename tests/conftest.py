"""
Shared pytest fixtures for all chimpss tests.
"""
import os
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def small_pdb_content():
    return (
        "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.541   1.000   1.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.020   2.430   1.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.249   3.371   1.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       2.023   0.144   2.147  1.00  0.00           C\n"
        "END\n"
    )


@pytest.fixture
def small_pdb(tmp_path, small_pdb_content):
    p = tmp_path / "test.pdb"
    p.write_text(small_pdb_content)
    return str(p)


@pytest.fixture
def pdb_with_dummy(tmp_path):
    content = (
        "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\n"
        "HETATM    2 DUM  DUM A   2       5.000   5.000   5.000  1.00  0.00          Du\n"
        "ATOM      3  CA  ALA A   1       1.541   1.000   1.000  1.00  0.00           C\n"
        "END\n"
    )
    p = tmp_path / "with_dummy.pdb"
    p.write_text(content)
    return str(p)


@pytest.fixture
def opm_pdb_path():
    here = os.path.dirname(__file__)
    return os.path.join(here, "..", "test_data", "raw_OPM", "5xr8.pdb")
