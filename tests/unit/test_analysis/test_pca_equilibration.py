"""
Unit tests for chimpss.analysis.pca_equilibration.

The module projects MD frames onto a PCA basis trained on crystal structures,
so the tests that matter are the ones guarding the silent failure modes:
a unit mismatch (nanometres against angstroms), an atom-ordering mismatch
between training and projection, and variance weighting that uses the
cumulative curve where per-component fractions are meant.  None of those
raise on their own -- they just produce wrong numbers.
"""
import json

import numpy as np
import pytest

pytest.importorskip("MDAnalysis", reason="MDAnalysis not installed")
pytest.importorskip("mdtraj", reason="mdtraj not installed")
pytest.importorskip("sklearn", reason="scikit-learn not installed")

import mdtraj as md  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from chimpss.analysis.pca_equilibration import (  # noqa: E402
    NM_TO_ANGSTROM,
    PanGPCRModel,
    _selection_atom_names,
    detect_pangpcr,
    kabsch_superpose,
    per_pc_table,
    project_coordinates,
    resolve_atoms,
    self_basis_projection,
    topology_sequence,
    training_spread,
)

#: A varied 70-residue reference sequence, so the alignment is unambiguous.
SEQ70 = "MHLNSSVPQGTPGEPDAQPFSGPQSEMEATFLALSLSNGSGNTSESDTAGPNSDLDVNTDIYSKVLVTA"

#: The trajectory covers GPCRdb residues 1-65 only, renumbered from resid 100.
N_TRAJ = 65
FIRST_RESID = 100

#: Conserved positions, five inside the construct and one past its C-terminus.
BW_SEQNUMS = [11, 21, 31, 41, 51]
BW_LABELS = ["1.50", "2.50", "3.50", "4.50", "5.50"]
BW_MISSING_SEQNUM, BW_MISSING_LABEL = 68, "7.50"

_ONE2THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    """A five-position, alpha-carbon PCA model with a known basis."""
    rng = np.random.default_rng(0)
    n = len(BW_LABELS)
    ref = rng.normal(scale=5.0, size=(n, 3))          # angstroms
    train = ref.ravel()[None] + rng.normal(scale=1.0, size=(40, n * 3))
    pca = PCA().fit(train)
    meta = {
        "selection": "name CA",
        "conserved_bw": list(BW_LABELS),
        "n_atoms": n,
        "codes_retained": [],
        "structure_meta": {},
    }
    return PanGPCRModel(pca=pca, meta=meta,
                        ref_positions=ref,
                        ref_masses=np.full(n, 12.011))


@pytest.fixture
def topology():
    """One protein chain renumbered from 100, plus water that must be ignored."""
    top = md.Topology()
    chain = top.add_chain()
    elems = {"N": md.element.nitrogen, "CA": md.element.carbon,
             "C": md.element.carbon, "O": md.element.oxygen}
    for i, letter in enumerate(SEQ70[:N_TRAJ]):
        res = top.add_residue(_ONE2THREE[letter], chain, resSeq=FIRST_RESID + i)
        for an in ("N", "CA", "C", "O"):
            top.add_atom(an, elems[an], res)
    wat = top.add_chain()
    for k in range(3):
        r = top.add_residue("HOH", wat, resSeq=900 + k)
        top.add_atom("O", md.element.oxygen, r)
    return top


@pytest.fixture
def naming_info():
    """A GPCRdb residues/extended payload for the full 70-residue sequence."""
    labelled = dict(zip(BW_SEQNUMS, BW_LABELS))
    labelled[BW_MISSING_SEQNUM] = BW_MISSING_LABEL
    out = []
    for i, letter in enumerate(SEQ70):
        seqnum = i + 1
        alts = ([{"scheme": "BW", "label": labelled[seqnum]}]
                if seqnum in labelled else [])
        out.append({"sequence_number": seqnum, "amino_acid": letter,
                    "alternative_generic_numbers": alts})
    return out


# ── selection parsing ─────────────────────────────────────────────────────────

def test_selection_atom_names_ca():
    assert _selection_atom_names("name CA") == ["CA"]
    assert _selection_atom_names("protein and name CA") == ["CA"]


def test_selection_atom_names_backbone():
    assert _selection_atom_names("backbone") == ["N", "CA", "C", "O"]


def test_selection_atom_names_explicit_list():
    assert _selection_atom_names("name CA CB") == ["CA", "CB"]


def test_selection_atom_names_rejects_unsupported():
    with pytest.raises(ValueError, match="unsupported training selection"):
        _selection_atom_names("resid 1-50 and around 5 protein")


# ── superposition ─────────────────────────────────────────────────────────────

def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_kabsch_recovers_a_known_rigid_transform():
    rng = np.random.default_rng(1)
    ref = rng.normal(scale=4.0, size=(12, 3))
    rot = _random_rotation(rng)
    moved = ref @ rot.T + np.array([13.0, -4.0, 7.5])
    back = kabsch_superpose(moved, ref)
    assert np.allclose(back, ref, atol=1e-8)


def test_kabsch_is_identity_for_already_aligned_coordinates():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(8, 3))
    assert np.allclose(kabsch_superpose(ref, ref), ref, atol=1e-10)


def test_kabsch_never_reflects():
    """A mirrored structure must not be superposed onto its reflection."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(10, 3))
    mirrored = ref * np.array([1.0, 1.0, -1.0])
    back = kabsch_superpose(mirrored, ref)
    assert not np.allclose(back, ref, atol=1e-6)


def test_kabsch_mass_weighting_changes_the_result():
    rng = np.random.default_rng(4)
    ref = rng.normal(scale=3.0, size=(10, 3))
    mobile = ref + rng.normal(scale=0.8, size=(10, 3))
    w = np.array([14.0, 12.0, 12.0, 16.0] * 2 + [14.0, 12.0])
    assert not np.allclose(kabsch_superpose(mobile, ref),
                           kabsch_superpose(mobile, ref, w), atol=1e-6)


def test_kabsch_matches_mdanalysis_alignto():
    """The training frame was built with alignto(weights='mass')."""
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.analysis import align

    rng = np.random.default_rng(5)
    n = 8
    ref_xyz = rng.normal(scale=5.0, size=(n, 3)).astype(np.float32)
    mob_xyz = (ref_xyz @ _random_rotation(rng).T
               + np.array([2.0, -3.0, 1.0])).astype(np.float32)
    masses = np.array([14.007, 12.011, 12.011, 15.999] * 2)

    def universe(xyz):
        u = mda.Universe.empty(n, n_residues=n, atom_resindex=np.arange(n),
                               trajectory=True)
        u.add_TopologyAttr("masses", masses)
        u.atoms.positions = xyz
        return u

    u_mob, u_ref = universe(mob_xyz), universe(ref_xyz)
    align.alignto(u_mob.atoms, u_ref.atoms, select="all", weights="mass")

    ours = kabsch_superpose(mob_xyz, ref_xyz, masses)
    assert np.allclose(ours, u_mob.atoms.positions, atol=1e-4)


def test_kabsch_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        kabsch_superpose(np.zeros((5, 3)), np.zeros((6, 3)))


# ── topology and residue resolution ───────────────────────────────────────────

def test_topology_sequence_skips_water(topology):
    seq = topology_sequence(topology, chain_index=0)
    assert [aa for _, aa in seq] == list(SEQ70[:N_TRAJ])
    assert [rid for rid, _ in seq] == list(range(FIRST_RESID, FIRST_RESID + N_TRAJ))


def test_topology_sequence_normalises_protonation_variants():
    top = md.Topology()
    ch = top.add_chain()
    for name in ("HID", "HIE", "HIP", "CYX"):
        r = top.add_residue(name, ch, resSeq=1)
        top.add_atom("CA", md.element.carbon, r)
    assert [aa for _, aa in topology_sequence(top)] == ["H", "H", "H", "C"]


def test_resolve_atoms_maps_every_conserved_position(model, topology, naming_info):
    idx, missing, stats = resolve_atoms(topology, naming_info, model)
    assert missing == []
    assert idx.size == len(BW_LABELS)
    assert stats["n_resolved"] == len(BW_LABELS)
    assert stats["atoms_per_residue"] == 1
    assert stats["identity"] == pytest.approx(1.0)
    # every index must be an alpha carbon of the protein chain
    assert all(topology.atom(int(i)).name == "CA" for i in idx)


def test_resolve_atoms_indices_are_in_bw_order(model, topology, naming_info):
    idx, _, _ = resolve_atoms(topology, naming_info, model)
    resids = [topology.atom(int(i)).residue.resSeq for i in idx]
    want = [FIRST_RESID + (n - 1) for n in BW_SEQNUMS]
    assert resids == want


def test_resolve_atoms_flags_a_position_absent_from_the_topology(
        model, topology, naming_info):
    """A BW label past the construct's C-terminus must be reported, not dropped."""
    model.meta["conserved_bw"] = BW_LABELS + [BW_MISSING_LABEL]
    idx, missing, stats = resolve_atoms(topology, naming_info, model)
    assert idx.size == len(BW_LABELS)
    assert [m[0] for m in missing] == [BW_MISSING_LABEL]
    assert stats["n_positions"] == len(BW_LABELS) + 1
    assert stats["n_resolved"] == len(BW_LABELS)


def test_resolve_atoms_backbone_selection_picks_four_atoms_per_residue(
        model, topology, naming_info):
    model.meta["selection"] = "backbone"
    idx, missing, stats = resolve_atoms(topology, naming_info, model)
    assert missing == []
    assert idx.size == 4 * len(BW_LABELS)
    assert stats["atoms_per_residue"] == 4
    assert [topology.atom(int(i)).name for i in idx[:4]] == ["N", "CA", "C", "O"]


# ── projection ────────────────────────────────────────────────────────────────

def test_project_coordinates_applies_the_nanometre_conversion(model):
    """Frames arrive in nm; the basis is in angstroms.

    A frame that is exactly the reference expressed in nanometres must land on
    the projection of the reference itself.  Without the factor of ten it lands
    an order of magnitude away, and nothing raises.
    """
    frame_nm = model.ref_positions / NM_TO_ANGSTROM
    got = project_coordinates(frame_nm[None], model)
    want = model.pca.transform(model.ref_positions.ravel()[None])
    assert np.allclose(got, want, atol=1e-4)


def test_project_coordinates_is_invariant_to_rigid_motion(model):
    rng = np.random.default_rng(6)
    frame_nm = model.ref_positions / NM_TO_ANGSTROM
    moved = frame_nm @ _random_rotation(rng).T + np.array([4.0, 1.0, -2.0])
    assert np.allclose(project_coordinates(frame_nm[None], model),
                       project_coordinates(moved[None], model), atol=1e-4)


def test_project_coordinates_accepts_a_single_frame(model):
    frame_nm = model.ref_positions / NM_TO_ANGSTROM
    assert project_coordinates(frame_nm, model).shape == (1, model.pca.n_components_)


def test_project_coordinates_rejects_wrong_atom_count(model):
    with pytest.raises(ValueError, match="expected 5"):
        project_coordinates(np.zeros((2, 9, 3)), model)


def test_self_basis_projection_shapes_and_variance(model):
    rng = np.random.default_rng(7)
    coords = (model.ref_positions / NM_TO_ANGSTROM)[None] + \
        rng.normal(scale=0.02, size=(30, len(BW_LABELS), 3))
    proj, evr = self_basis_projection(coords, model)
    assert proj.shape[0] == 30
    assert evr.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(evr) <= 1e-12)          # components are ordered


# ── detection and weighting ───────────────────────────────────────────────────

class _FixedDetector:
    """Returns a pre-set t0 per component, so weighting can be checked exactly."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, series, **_):
        v = self.values[self.calls]
        self.calls += 1
        if v is None:
            raise RuntimeError("detector failed")
        return v


def test_per_pc_table_records_variance_and_discard():
    proj = np.zeros((100, 3))
    rows = per_pc_table(proj, np.array([0.7, 0.2, 0.1]),
                        _FixedDetector([10, 50, 90]), n_pcs=3)
    assert [r["t0"] for r in rows] == [10, 50, 90]
    assert [r["discard"] for r in rows] == [0.1, 0.5, 0.9]
    assert [r["evr"] for r in rows] == [0.7, 0.2, 0.1]


def test_per_pc_table_captures_a_failing_detector_without_raising():
    proj = np.zeros((10, 2))
    rows = per_pc_table(proj, np.array([0.9, 0.1]),
                        _FixedDetector([5, None]), n_pcs=2)
    assert rows[0]["t0"] == 5
    assert rows[1]["t0"] is None
    assert "RuntimeError" in rows[1]["error"]


def test_detect_pangpcr_weights_by_per_component_variance():
    """PC1 carries 90% of the variance, so it must dominate the consensus."""
    proj = np.zeros((1000, 2))
    evr = np.array([0.9, 0.1])
    out = detect_pangpcr(proj, evr, detector=_FixedDetector([100, 0]), n_pcs=2)
    assert out["t0"] == 90


def test_detect_pangpcr_rejects_cumulative_weights_regression():
    """Guards the weighting bug fixed in ``determine_equilibration``.

    Passing the cumulative curve where per-component fractions are meant gives
    the tail components more weight than PC1.  With PC1 at t0=100 and a flat
    tail at t0=0, correct weighting lands near 100 and the cumulative curve
    drags the answer far below it.
    """
    proj = np.zeros((1000, 4))
    evr = np.array([0.7, 0.15, 0.10, 0.05])
    cumulative = np.cumsum(evr)

    correct = detect_pangpcr(proj, evr,
                             detector=_FixedDetector([100, 0, 0, 0]), n_pcs=4)
    wrong = detect_pangpcr(proj, cumulative,
                           detector=_FixedDetector([100, 0, 0, 0]), n_pcs=4)
    assert correct["t0"] == 70
    assert wrong["t0"] < correct["t0"]


def test_detect_pangpcr_chooses_n_pcs_from_cumulative_variance():
    proj = np.zeros((100, 5))
    evr = np.array([0.5, 0.3, 0.12, 0.05, 0.03])
    out = detect_pangpcr(proj, evr, detector=_FixedDetector([0] * 5),
                         cumulative_variance=0.90)
    assert out["n_pcs"] == 3


def test_detect_pangpcr_falls_back_to_uniform_weights_on_zero_variance():
    proj = np.zeros((100, 2))
    out = detect_pangpcr(proj, np.array([0.0, 0.0]),
                         detector=_FixedDetector([40, 60]), n_pcs=2)
    assert out["t0"] == 50


def test_detect_pangpcr_raises_when_every_component_fails():
    proj = np.zeros((10, 2))
    with pytest.raises(RuntimeError, match="failed on every component"):
        detect_pangpcr(proj, np.array([0.9, 0.1]),
                       detector=_FixedDetector([None, None]), n_pcs=2)


def test_detect_pangpcr_rejects_empty_projection():
    with pytest.raises(ValueError, match="empty projection"):
        detect_pangpcr(np.empty((0, 0)), np.array([1.0]))


def test_detect_pangpcr_recovers_a_planted_transient():
    """A decaying transient on PC1 must yield a later t0 than a flat series."""
    rng = np.random.default_rng(8)
    n, t_true = 600, 150
    transient = np.concatenate([
        np.exp(-np.arange(t_true) / 30.0) * 8.0,
        np.zeros(n - t_true)]) + rng.normal(scale=0.2, size=n)
    flat = rng.normal(scale=0.2, size=n)

    evr = np.array([1.0])
    hot = detect_pangpcr(transient[:, None], evr, detector="mser", n_pcs=1)
    cold = detect_pangpcr(flat[:, None], evr, detector="mser", n_pcs=1)

    assert hot["t0"] > cold["t0"]
    assert 0.5 * t_true <= hot["t0"] <= 2.0 * t_true


# ── training spread ───────────────────────────────────────────────────────────

def test_training_spread_returns_none_without_projections(model, tmp_path):
    assert training_spread(model, str(tmp_path / "absent")) is None


def test_training_spread_separates_active_from_inactive(model, tmp_path):
    rng = np.random.default_rng(9)
    active = rng.normal(loc=[8.0, 0.0], scale=0.5, size=(20, 2))
    inactive = rng.normal(loc=[-8.0, 0.0], scale=0.5, size=(20, 2))
    prefix = str(tmp_path / "m")
    np.save(f"{prefix}_train_projections.npy", np.vstack([active, inactive]))

    codes = [f"A{i}" for i in range(20)] + [f"I{i}" for i in range(20)]
    model.meta["codes_retained"] = codes
    model.meta["structure_meta"] = {
        c: {"state": "Active" if c.startswith("A") else "Inactive"} for c in codes}

    out = training_spread(model, prefix, n_pcs=2)
    assert out["n_structures"] == 40
    assert out["by_state"]["Active"]["n"] == 20
    assert out["active_inactive_cohens_d"][0] > 10.0      # PC1 separates
    assert out["active_inactive_cohens_d"][1] < 1.0       # PC2 does not
