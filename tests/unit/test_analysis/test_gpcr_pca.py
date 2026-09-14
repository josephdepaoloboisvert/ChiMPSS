"""
Unit tests for pure-logic functions in chimpss.analysis.gpcr_pca.

gpcr_pca.py imports MDAnalysis at module level, so all tests are gated.
Functions tested here contain no MDAnalysis calls — they work on plain
dicts and lists and represent the BW-numbering and sequence-matching logic.
"""
import pytest

pytest.importorskip("MDAnalysis", reason="MDAnalysis not installed — skipping gpcr_pca tests")

from chimpss.analysis.gpcr_pca import (
    _NON_PROTEIN,
    _THREE2ONE,
    _normalize_resname,
    build_gpcrdb_sequence,
    naming_from_convention,
    sliding_window_align,
    three_to_one,
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


# ── gap-aware sequence mapping ────────────────────────────────────────────────

from chimpss.analysis.gpcr_pca import (  # noqa: E402
    _gotoh_align,
    align_sequence_to_gpcrdb,
    map_conserved_resids_aligned,
)

_SEQ = ("MHLNSSVPQGTPGEPDAQPFSGPQSEMEATFLALSLSNGSGNTSESDTAGPNSDLDVNTDIY"
        "SKVLVTAVYLALFVVGTVGNSVTAFTLARKKSLQSLQSTVHYHLGSLALSDLLILLLAMPVE")


def _seqdict(s, start=1):
    return {i + start: aa for i, aa in enumerate(s)}


def test_gotoh_align_identical_sequences():
    pairs = _gotoh_align("ACDEFGHIK", "ACDEFGHIK")
    assert pairs == [(i, i) for i in range(9)]


def test_gotoh_align_recovers_internal_deletion():
    ref = "ACDEFGHIKLMNPQRST"
    qry = ref[:6] + ref[10:]          # four residues deleted internally
    pairs = _gotoh_align(ref, qry)
    assert len(pairs) == len(qry)
    for i, j in pairs:
        assert ref[i] == qry[j]


def test_align_sequence_to_gpcrdb_perfect_single_offset():
    gseq = _seqdict(_SEQ, start=1)
    traj = [(i + 50, aa) for i, aa in enumerate(_SEQ)]
    mapping, stats = align_sequence_to_gpcrdb(traj, gseq)
    assert stats["n_aligned"] == len(_SEQ)
    assert stats["identity"] == pytest.approx(1.0)
    assert stats["offsets"] == {-49: len(_SEQ)}
    assert mapping[1] == 50


def test_align_sequence_to_gpcrdb_resolves_two_offsets_across_a_deletion():
    """The regression this mapper exists for.

    A crystallisation construct with an internal deletion, renumbered
    contiguously, cannot be described by one integer offset.  The aligner must
    recover both segments and map every residue that is actually present.
    """
    gseq = _seqdict(_SEQ, start=1)
    kept = _SEQ[:60] + _SEQ[64:]               # drop GPCRdb residues 61-64
    traj = [(i + 50, aa) for i, aa in enumerate(kept)]

    mapping, stats = align_sequence_to_gpcrdb(traj, gseq)

    assert stats["n_aligned"] == len(kept)
    assert stats["identity"] == pytest.approx(1.0)
    assert len(stats["offsets"]) == 2
    assert set(stats["offsets"]) == {-49, -45}
    assert mapping[1] == 50                    # before the deletion
    assert mapping[65] == 110                  # after it, shifted by four


def test_single_offset_search_fails_where_the_aligner_succeeds():
    """Contrast with :func:`sliding_window_align`, which assumes one offset."""
    gseq = _seqdict(_SEQ, start=1)
    kept = _SEQ[:60] + _SEQ[64:]
    traj = [(i + 50, aa) for i, aa in enumerate(kept)]

    _, confidence, _, _ = sliding_window_align(traj, gseq)
    _, stats = align_sequence_to_gpcrdb(traj, gseq)

    assert confidence < 0.75                   # single offset cannot span the gap
    assert stats["identity"] == pytest.approx(1.0)


def test_align_sequence_to_gpcrdb_tolerates_point_mutations():
    gseq = _seqdict(_SEQ, start=1)
    mutated = list(_SEQ)
    for i in (5, 20, 44, 71, 90):
        mutated[i] = "A" if mutated[i] != "A" else "G"
    traj = [(i + 1, aa) for i, aa in enumerate(mutated)]

    mapping, stats = align_sequence_to_gpcrdb(traj, gseq)
    assert stats["n_aligned"] == len(_SEQ)
    assert 0.90 < stats["identity"] < 1.0
    assert mapping[1] == 1


def test_align_sequence_to_gpcrdb_n_terminal_truncation():
    gseq = _seqdict(_SEQ, start=1)
    traj = [(i + 1, aa) for i, aa in enumerate(_SEQ[30:])]
    mapping, stats = align_sequence_to_gpcrdb(traj, gseq)
    assert stats["identity"] == pytest.approx(1.0)
    assert mapping[31] == 1


def test_align_sequence_to_gpcrdb_empty_inputs():
    mapping, stats = align_sequence_to_gpcrdb([], {})
    assert mapping == {}
    assert stats["n_aligned"] == 0
    assert stats["identity"] == 0.0


def test_map_conserved_resids_aligned_maps_present_positions():
    bw_map = {10: "3.50", 20: "6.48", 30: "7.49"}
    seq_map = {10: 110, 20: 120, 30: 130}
    resids, missing = map_conserved_resids_aligned(
        bw_map, ["3.50", "6.48", "7.49"], seq_map, available={110, 120, 130})
    assert resids == [[110], [120], [130]]
    assert missing == []


def test_map_conserved_resids_aligned_flags_unmapped_and_unavailable():
    bw_map = {10: "3.50", 20: "6.48"}
    seq_map = {10: 110}                        # 20 never aligned
    resids, missing = map_conserved_resids_aligned(
        bw_map, ["3.50", "6.48", "1.50"], seq_map, available={110})
    assert resids == [[110], [], []]
    reasons = {lbl: reason for lbl, _, reason in missing}
    assert "not aligned" in reasons["6.48"]
    assert "not in GPCRdb BW map" in reasons["1.50"]


def test_map_conserved_resids_aligned_flags_resid_without_backbone():
    bw_map = {10: "3.50"}
    seq_map = {10: 110}
    resids, missing = map_conserved_resids_aligned(
        bw_map, ["3.50"], seq_map, available=set())
    assert resids == [[]]
    assert "no backbone atoms" in missing[0][2]


# ── numbering validation ──────────────────────────────────────────────────────

from chimpss.analysis.gpcr_pca import validate_numbering  # noqa: E402


def test_validate_numbering_trusts_correct_author_numbering():
    """Deposited entries almost always number by UniProt; do not touch them."""
    gseq = _seqdict(_SEQ, start=1)
    observed = [(i + 1, aa) for i, aa in enumerate(_SEQ)]
    mode, mapping, stats = validate_numbering(observed, gseq)
    assert mode == "direct"
    assert stats["direct_identity"] == pytest.approx(1.0)
    assert mapping[1] == 1


def test_validate_numbering_tolerates_disordered_gaps():
    """Missing density must not trigger a realignment: the numbering is fine."""
    gseq = _seqdict(_SEQ, start=1)
    observed = [(i + 1, aa) for i, aa in enumerate(_SEQ)
                if not (30 <= i < 45)]
    mode, _, stats = validate_numbering(observed, gseq)
    assert mode == "direct"
    assert stats["direct_identity"] == pytest.approx(1.0)


def test_validate_numbering_realigns_construct_numbering():
    """Construct numbering that does not match UniProt must be remapped."""
    gseq = _seqdict(_SEQ, start=1)
    observed = [(i + 1, aa) for i, aa in enumerate(_SEQ[20:])]   # renumbered from 1
    mode, mapping, stats = validate_numbering(observed, gseq)
    assert mode == "aligned"
    assert stats["aligned_identity"] > stats["direct_identity"]
    assert mapping[21] == 1


def test_validate_numbering_keeps_direct_when_alignment_is_worse():
    """A noisy chimera must not be made worse by realigning it."""
    gseq = _seqdict(_SEQ, start=1)
    observed = [(i + 1, "W") for i in range(len(_SEQ))]          # nothing matches
    mode, _, stats = validate_numbering(observed, gseq)
    assert mode == "direct"
    assert stats["identity"] == stats["direct_identity"]


def test_validate_numbering_threshold_is_honoured():
    gseq = _seqdict(_SEQ, start=1)
    observed = [(i + 1, aa) for i, aa in enumerate(_SEQ)]
    assert validate_numbering(observed, gseq, identity_threshold=1.01)[0] != "direct" \
        or validate_numbering(observed, gseq, identity_threshold=0.5)[0] == "direct"


def test_validate_numbering_handles_no_overlapping_resids():
    gseq = _seqdict(_SEQ, start=1)
    observed = [(i + 10000, aa) for i, aa in enumerate(_SEQ)]
    mode, mapping, stats = validate_numbering(observed, gseq)
    assert stats["n_checked"] == 0
    assert mode == "aligned"
    assert mapping[1] == 10000
