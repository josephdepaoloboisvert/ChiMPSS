"""
Unit tests for pure-logic functions in chimpss.analysis.gpcr_pca.

gpcr_pca.py imports MDAnalysis at module level, so all tests are gated.
Functions tested here contain no MDAnalysis calls — they work on plain
dicts and lists and represent the BW-numbering and sequence-matching logic.
"""
import pytest

pytest.importorskip("MDAnalysis", reason="MDAnalysis not installed — skipping gpcr_pca tests")

from chimpss.analysis.gpcr_pca import (
    _THREE2ONE,
    _MD_RESNAME_NORM,
    _NON_PROTEIN,
    three_to_one,
    _normalize_resname,
    build_gpcrdb_sequence,
    sliding_window_align,
    naming_from_convention,
)


# ── _THREE2ONE / three_to_one ─────────────────────────────────────────────────

def test_three2one_standard_residues():
    assert _THREE2ONE["ALA"] == "A"
    assert _THREE2ONE["GLY"] == "G"
    assert _THREE2ONE["TRP"] == "W"
    assert _THREE2ONE["CYS"] == "C"


def test_three_to_one_known():
    assert three_to_one("ALA") == "A"
    assert three_to_one("LYS") == "K"
    assert three_to_one("PHE") == "F"


def test_three_to_one_unknown():
    assert three_to_one("UNK") == "X"
    assert three_to_one("FOO") == "X"


def test_three2one_has_20_canonical():
    assert len(_THREE2ONE) == 20


# ── _normalize_resname ────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant,canonical", [
    ("HID", "HIS"), ("HIE", "HIS"), ("HIP", "HIS"),
    ("HSD", "HIS"), ("HSE", "HIS"),
    ("CYX", "CYS"), ("CYM", "CYS"),
    ("ASH", "ASP"), ("GLH", "GLU"), ("LYN", "LYS"),
])
def test_normalize_resname_variants(variant, canonical):
    assert _normalize_resname(variant) == canonical


def test_normalize_resname_canonical_passthrough():
    assert _normalize_resname("ALA") == "ALA"
    assert _normalize_resname("GLY") == "GLY"


def test_normalize_resname_case_insensitive():
    assert _normalize_resname("hid") == "HIS"
    assert _normalize_resname("Cyx") == "CYS"


# ── _NON_PROTEIN ──────────────────────────────────────────────────────────────

def test_non_protein_contains_water():
    assert "HOH" in _NON_PROTEIN
    assert "WAT" in _NON_PROTEIN


def test_non_protein_contains_ions():
    assert "NA" in _NON_PROTEIN
    assert "CL" in _NON_PROTEIN


# ── build_gpcrdb_sequence ─────────────────────────────────────────────────────

def test_build_gpcrdb_sequence_basic():
    naming_info = [
        {"sequence_number": 1, "amino_acid": "M"},
        {"sequence_number": 2, "amino_acid": "K"},
        {"sequence_number": 3, "amino_acid": "T"},
    ]
    result = build_gpcrdb_sequence(naming_info)
    assert result == {1: "M", 2: "K", 3: "T"}


def test_build_gpcrdb_sequence_skips_X():
    naming_info = [
        {"sequence_number": 1, "amino_acid": "M"},
        {"sequence_number": 2, "amino_acid": "X"},
        {"sequence_number": 3, "amino_acid": "K"},
    ]
    result = build_gpcrdb_sequence(naming_info)
    assert 2 not in result
    assert result[1] == "M"
    assert result[3] == "K"


def test_build_gpcrdb_sequence_skips_none_seqnum():
    naming_info = [
        {"sequence_number": None, "amino_acid": "A"},
        {"sequence_number": 5, "amino_acid": "G"},
    ]
    result = build_gpcrdb_sequence(naming_info)
    assert None not in result
    assert result[5] == "G"


def test_build_gpcrdb_sequence_uppercases():
    naming_info = [{"sequence_number": 1, "amino_acid": "m"}]
    result = build_gpcrdb_sequence(naming_info)
    assert result[1] == "M"


def test_build_gpcrdb_sequence_empty():
    assert build_gpcrdb_sequence([]) == {}


# ── sliding_window_align ──────────────────────────────────────────────────────

def test_sliding_window_align_empty_inputs():
    offset, conf, matched, compared = sliding_window_align([], {})
    assert offset == 0
    assert conf == 0.0
    assert matched == 0
    assert compared == 0


def test_sliding_window_align_perfect_match():
    # GPCRdb: {1: 'A', 2: 'C', 3: 'D'}
    # Traj starts at resid 1, same sequence
    gpcrdb = {1: "A", 2: "C", 3: "D"}
    traj = [(1, "A"), (2, "C"), (3, "D")]
    offset, conf, matched, compared = sliding_window_align(traj, gpcrdb)
    assert matched == 3
    assert conf == pytest.approx(1.0)
    assert compared == 3


def test_sliding_window_align_offset():
    gpcrdb = {100: "A", 101: "C", 102: "D", 103: "E"}
    traj = [(200, "A"), (201, "C"), (202, "D"), (203, "E")]
    offset, conf, matched, compared = sliding_window_align(traj, gpcrdb)
    assert conf == pytest.approx(1.0)
    assert matched == 4


# ── naming_from_convention ────────────────────────────────────────────────────

def _make_seq_info(scheme="BW"):
    return [
        {
            "sequence_number": 1,
            "amino_acid": "M",
            "alternative_generic_numbers": [
                {"scheme": scheme, "label": "1.50"},
                {"scheme": "other", "label": "X.XX"},
            ],
        },
        {
            "sequence_number": 2,
            "amino_acid": "K",
            "alternative_generic_numbers": [],
        },
    ]


def test_naming_from_convention_bw():
    result = naming_from_convention(_make_seq_info("BW"), scheme="BW")
    assert result[1] == "1.50"
    assert result[2] == "-1"


def test_naming_from_convention_no_match_returns_minus1():
    seq_info = [{"sequence_number": 5, "amino_acid": "A", "alternative_generic_numbers": []}]
    result = naming_from_convention(seq_info, scheme="BW")
    assert result[5] == "-1"


def test_naming_from_convention_invalid_scheme():
    with pytest.raises(AssertionError):
        naming_from_convention([], scheme="INVALID")
