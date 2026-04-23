"""
Unit tests for chimpss.shared.io.

All functions that call mdtraj are in the same module, so the whole module
is gated. Tests for ensure_exists, write_FASTA, and remove_dummy_atoms use
only stdlib + tmpfiles and are fast.
"""
import os
import pytest

pytest.importorskip("mdtraj", reason="mdtraj not installed — skipping shared.io tests")

from chimpss.shared.io import ensure_exists, write_FASTA, remove_dummy_atoms


# ── ensure_exists ─────────────────────────────────────────────────────────────

def test_ensure_exists_creates_directory(tmp_path):
    new_dir = str(tmp_path / "new_subdir")
    assert not os.path.isdir(new_dir)
    result = ensure_exists(new_dir)
    assert result is True
    assert os.path.isdir(new_dir)


def test_ensure_exists_idempotent(tmp_path):
    d = str(tmp_path / "existing")
    os.makedirs(d)
    result = ensure_exists(d)
    assert result is True
    assert os.path.isdir(d)


def test_ensure_exists_nested(tmp_path):
    nested = str(tmp_path / "a" / "b" / "c")
    ensure_exists(nested)
    assert os.path.isdir(nested)


# ── write_FASTA ───────────────────────────────────────────────────────────────

def test_write_fasta_creates_file(tmp_path):
    out = str(tmp_path / "test.fasta")
    returned = write_FASTA("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTOKENIZER")
    assert returned == out


def test_write_fasta_returns_path(tmp_path):
    out = str(tmp_path / "out.fasta")
    result = write_FASTA("ACDEFGHIKLMNPQRSTVWY", "test_protein", out)
    assert result == out


def test_write_fasta_content(tmp_path):
    out = str(tmp_path / "prot.fasta")
    write_FASTA("ACDEFG", "myprotein", out)
    with open(out) as f:
        content = f.read()
    assert "myprotein" in content
    assert "ACDEFG" in content
    assert content.startswith(">P1;myprotein")


def test_write_fasta_ends_with_star(tmp_path):
    out = str(tmp_path / "prot.fasta")
    write_FASTA("AAAA", "seq", out)
    with open(out) as f:
        content = f.read()
    assert content.rstrip().endswith("*")


# ── remove_dummy_atoms ────────────────────────────────────────────────────────

def test_remove_dummy_atoms_creates_file(tmp_path):
    pdb_content = (
        "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\n"
        "HETATM    2 DUM  DUM A   2       5.000   5.000   5.000  1.00  0.00          Du\n"
        "ATOM      3  CA  ALA A   1       1.541   1.000   1.000  1.00  0.00           C\n"
        "END\n"
    )
    pdb = tmp_path / "test.pdb"
    pdb.write_text(pdb_content)
    result = remove_dummy_atoms(str(pdb))
    assert result.endswith("_no_dummy.pdb")
    assert os.path.exists(result)


def test_remove_dummy_atoms_strips_dummy_lines(tmp_path):
    pdb_content = (
        "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\n"
        "HETATM    2 DUM  DUM A   2       5.000   5.000   5.000  1.00  0.00          Du\n"
        "ATOM      3  CA  ALA A   1       1.541   1.000   1.000  1.00  0.00           C\n"
        "END\n"
    )
    pdb = tmp_path / "test.pdb"
    pdb.write_text(pdb_content)
    result = remove_dummy_atoms(str(pdb))
    with open(result) as f:
        out_content = f.read()
    assert "DUM" not in out_content
    assert "ATOM" in out_content


def test_remove_dummy_atoms_no_dummies(tmp_path):
    pdb_content = (
        "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.541   1.000   1.000  1.00  0.00           C\n"
        "END\n"
    )
    pdb = tmp_path / "clean.pdb"
    pdb.write_text(pdb_content)
    result = remove_dummy_atoms(str(pdb))
    with open(result) as f:
        out_content = f.read()
    assert "ATOM" in out_content
