"""Unit tests pinning the hydrogen mass repartitioning arithmetic.

``chimpss.bridgeport._utils.apply_hmr`` is the only production HMR path.  It is
applied unconditionally by ``Bridgeport._write_adjusted_systems`` to produce
``<PROTEIN>_<LIGAND>.FG_HMR_system.xml``, and nothing downstream re-checks it.

The contract these tests fix in place:

* a heavy atom carrying exactly one hydrogen scales that hydrogen to **2.0x**
  the standard hydrogen mass (1.007947 -> 2.015894 Da),
* a heavy atom carrying two or more hydrogens scales each of them to **2.5x**
  (1.007947 -> 2.519868 Da),
* the heavy atom is debited exactly what its hydrogens are credited, so total
  system mass is conserved,
* water is untouched.

This matters because the ChiMPSS manuscript Methods describe a 3.0x/3.5x
scheme (3.02-3.53 Da), which corresponds to transferring ``s * h_mass``
instead of ``(s - 1) * h_mass`` -- one whole hydrogen mass too much.  The
production code does the latter.  If anyone ever "restores" the manuscript
numbers by editing the transfer, these tests fail.
"""
import pytest

openmm = pytest.importorskip("openmm")
app = pytest.importorskip("openmm.app")

from openmm import System, XmlSerializer  # noqa: E402
from openmm.app import Element, Topology  # noqa: E402
from openmm.unit import dalton  # noqa: E402

from chimpss.bridgeport._utils import apply_hmr  # noqa: E402

H_MASS = Element.getBySymbol("H").mass.value_in_unit(dalton)
C_MASS = Element.getBySymbol("C").mass.value_in_unit(dalton)
O_MASS = Element.getBySymbol("O").mass.value_in_unit(dalton)


def _build(residues):
    """Build a (System, Topology) pair from a compact residue spec.

    ``residues`` is a list of ``(resname, [(heavy_symbol, n_hydrogens), ...])``.
    Every hydrogen is bonded to its heavy atom; heavy atoms within a residue are
    chained together so the topology is connected but carries no extra X-H bonds.
    """
    system = System()
    top = Topology()
    chain = top.addChain()

    for resname, heavies in residues:
        res = top.addResidue(resname, chain)
        prev_heavy = None
        for symbol, n_h in heavies:
            element = Element.getBySymbol(symbol)
            heavy = top.addAtom(symbol, element, res)
            system.addParticle(element.mass)
            if prev_heavy is not None:
                top.addBond(prev_heavy, heavy)
            prev_heavy = heavy
            for i in range(n_h):
                h = top.addAtom(f"H{i}", Element.getBySymbol("H"), res)
                system.addParticle(Element.getBySymbol("H").mass)
                top.addBond(heavy, h)

    return system, top


def _masses(system):
    return [system.getParticleMass(i).value_in_unit(dalton)
            for i in range(system.getNumParticles())]


# --------------------------------------------------------------------------
# Scale factors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_h,expected_scale", [
    (1, 2.0),
    (2, 2.5),
    (3, 2.5),
    (4, 2.5),
])
def test_hydrogen_final_mass_by_h_count(n_h, expected_scale):
    """1 H on the heavy atom -> 2.0x; 2 or more -> 2.5x."""
    system, top = _build([("LIG", [("C", n_h)])])
    apply_hmr(system, top)

    masses = _masses(system)
    h_masses = masses[1:]

    assert len(h_masses) == n_h
    for m in h_masses:
        assert m == pytest.approx(expected_scale * H_MASS, rel=1e-9)


def test_manuscript_scheme_is_not_what_the_code_does():
    """Guard against reintroducing the 3.0x/3.5x (3.02/3.53 Da) transfer.

    The published Methods say hydrogens were scaled to 3.02-3.53 Da.  That is
    ``dH = s * h_mass`` rather than ``dH = (s - 1) * h_mass``.  Pin the
    difference explicitly so the regression is unmissable.
    """
    system, top = _build([("LIG", [("C", 1)]), ("LIG", [("C", 2)])])
    apply_hmr(system, top)
    masses = _masses(system)

    ch_hydrogen = masses[1]
    ch2_hydrogen = masses[3]

    assert ch_hydrogen == pytest.approx(2.015894, abs=1e-5)
    assert ch2_hydrogen == pytest.approx(2.519868, abs=1e-5)

    # Explicitly NOT the manuscript values.
    assert ch_hydrogen != pytest.approx(3.023841, abs=1e-3)
    assert ch2_hydrogen != pytest.approx(3.527815, abs=1e-3)


# --------------------------------------------------------------------------
# Mass conservation
# --------------------------------------------------------------------------

def test_total_mass_is_conserved():
    system, top = _build([
        ("LIG", [("C", 3), ("C", 2), ("C", 1), ("O", 1), ("N", 2)]),
        ("ALA", [("C", 1), ("C", 3)]),
    ])
    before = sum(_masses(system))
    apply_hmr(system, top)
    after = sum(_masses(system))

    assert after == pytest.approx(before, abs=1e-9)


def test_heavy_atom_debit_equals_hydrogen_credit():
    """A methyl carbon loses exactly 3 x 1.5119 Da; a methine carbon 1 x 1.0079."""
    system, top = _build([("LIG", [("C", 3)]), ("LIG", [("C", 1)])])
    apply_hmr(system, top)
    masses = _masses(system)

    methyl_c, h1, h2, h3, methine_c, h4 = masses

    assert methyl_c == pytest.approx(C_MASS - 3 * (1.5 * H_MASS), rel=1e-9)
    assert methine_c == pytest.approx(C_MASS - 1 * (1.0 * H_MASS), rel=1e-9)
    assert methyl_c + h1 + h2 + h3 == pytest.approx(C_MASS + 3 * H_MASS, rel=1e-9)
    assert methine_c + h4 == pytest.approx(C_MASS + H_MASS, rel=1e-9)


def test_no_particle_reaches_nonpositive_mass():
    """Even CH4 / NH4+ stay comfortably positive at these scale factors."""
    system, top = _build([("LIG", [("C", 4)]), ("LIG", [("N", 4)])])
    apply_hmr(system, top)

    for m in _masses(system):
        assert m > 0.5


# --------------------------------------------------------------------------
# Water exclusion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("resname", ["HOH", "WAT", "TIP3", "SOL", "TIP"])
def test_water_is_untouched(resname):
    system, top = _build([(resname, [("O", 2)])])
    before = _masses(system)
    apply_hmr(system, top)
    after = _masses(system)

    assert after == pytest.approx(before, abs=1e-12)
    assert after[0] == pytest.approx(O_MASS, rel=1e-9)


def test_solute_repartitioned_while_water_alongside_is_not():
    system, top = _build([
        ("LIG", [("C", 2)]),
        ("HOH", [("O", 2)]),
    ])
    apply_hmr(system, top)
    masses = _masses(system)

    lig_c, lig_h1, lig_h2, wat_o, wat_h1, wat_h2 = masses

    assert lig_h1 == pytest.approx(2.5 * H_MASS, rel=1e-9)
    assert lig_h2 == pytest.approx(2.5 * H_MASS, rel=1e-9)
    assert wat_o == pytest.approx(O_MASS, rel=1e-9)
    assert wat_h1 == pytest.approx(H_MASS, rel=1e-9)
    assert wat_h2 == pytest.approx(H_MASS, rel=1e-9)


# --------------------------------------------------------------------------
# Lipids and ions are deliberately in scope
# --------------------------------------------------------------------------

def test_popc_hydrogens_are_repartitioned():
    """POPC is not on the water exclusion list -- lipid H are repartitioned."""
    system, top = _build([("POP", [("C", 3), ("C", 2)])])
    apply_hmr(system, top)
    masses = _masses(system)

    assert masses[1] == pytest.approx(2.5 * H_MASS, rel=1e-9)
    assert masses[5] == pytest.approx(2.5 * H_MASS, rel=1e-9)


# --------------------------------------------------------------------------
# Round trip through serialization, which is how the XML is actually produced
# --------------------------------------------------------------------------

def test_survives_xml_round_trip():
    system, top = _build([("LIG", [("C", 3), ("O", 1)])])
    apply_hmr(system, top)
    expected = _masses(system)

    restored = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    assert _masses(restored) == pytest.approx(expected, abs=1e-9)
