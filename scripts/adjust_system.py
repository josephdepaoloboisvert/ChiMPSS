#!/usr/bin/env python
"""
Adjust an OpenMM system XML file by:
1. Setting force groups (nonbonded forces in group 0+1, others in group 0)
2. Applying Hydrogen Mass Repartitioning (HMR) for larger MD timesteps

HMR scales hydrogen masses based on how many hydrogens are bonded to each heavy atom:
  - 1 H:   scale by 2.0x
  - 2 H:   scale by 2.5x
  - 3 H:   scale by 2.5x
  - 4+ H:  scale by 2.5x

Mass is transferred from the bonded heavy atom to preserve total system mass.

Usage:
    python ADJUST_SYSTEM.py <pdb_file> <xml_file>

Output files:
    <xml_file_base>_FG_sys.xml       : System with adjusted force groups
    <xml_file_base>_FG_HMR_sys.xml   : System with force groups and HMR applied
"""

import argparse
import os
import sys

import mdtraj as md
from openmm import *
from openmm.app import *
from openmm.unit import *


def adjust_system(pdb_file: str, xml_file: str):
    """
    Load a PDB and XML system file, adjust force groups, and apply HMR.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file with coordinates and topology.
    xml_file : str
        Path to OpenMM serialized system XML file.

    Outputs
    -------
    Writes two XML files:
      - {xml_base}_FG_sys.xml: Original system with adjusted force groups
      - {xml_base}_FG_HMR_sys.xml: With force groups and HMR applied
    """
    # Validation
    if not os.path.isfile(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    if not os.path.isfile(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")

    # Determine output paths
    xml_base = xml_file.replace('_sys.xml', '')
    fg_sys_out = f"{xml_base}_FG_sys.xml"
    fg_hmr_sys_out = f"{xml_base}_FG_HMR_sys.xml"

    print(f"Loading {pdb_file}")
    print(f"Loading {xml_file}")

    # Load topology and system
    top = PDBFile(pdb_file).topology
    with open(xml_file, 'r') as f:
        sys = XmlSerializer.deserialize(f.read())

    # ========================================================================
    # Step 1: Adjust force groups
    # ========================================================================
    print("Adjusting force groups...")
    for force in sys.getForces():
        if isinstance(force, NonbondedForce):
            force.setForceGroup(0)
            force.setReciprocalSpaceForceGroup(1)
        else:
            force.setForceGroup(0)

    # Write force-group adjusted system
    with open(fg_sys_out, 'w') as f:
        f.write(XmlSerializer.serialize(sys))
    print(f"Wrote {fg_sys_out}")

    # ========================================================================
    # Step 2: Apply Hydrogen Mass Repartitioning (HMR)
    # ========================================================================
    print("Applying Hydrogen Mass Repartitioning...")

    # HMR scaling factors: dH = (scale - 1) * element.hydrogen.mass per hydrogen
    # Indexed by (number of H on heavy atom - 1)
    hmr_scales = [2.0, 2.5, 2.5, 2.5]  # for 1, 2, 3, 4+ hydrogens
    dH_per_H = [
        element.hydrogen.mass * scale - element.hydrogen.mass
        for scale in hmr_scales
    ]

    # Select non-water atoms (to exclude water hydrogens from HMR)
    not_water_indices = set(md.load(pdb_file).top.select('not water'))

    # Find all covalent X-H bonds where X is not water
    covalent_h_bonds = []
    for bond in top.bonds():
        atoms = [bond.atom1, bond.atom2]
        is_h = [element.hydrogen == atom.element for atom in atoms]

        if True in is_h:
            h_atom = atoms[is_h.index(True)]
            heavy_atom = atoms[is_h.index(False)]
            h_ind = h_atom.index
            heavy_ind = heavy_atom.index

            if heavy_ind in not_water_indices:
                covalent_h_bonds.append([heavy_ind, h_ind])

    covalent_h_bonds.sort()

    # Count hydrogens per heavy atom
    heavy_atom_h_count = {}
    for heavy_ind, h_ind in covalent_h_bonds:
        heavy_atom_h_count[heavy_ind] = heavy_atom_h_count.get(heavy_ind, 0) + 1

    # Build scaling plan: [heavy_ind, h_ind, delta_mass]
    scale_plan = []
    for heavy_ind, h_ind in covalent_h_bonds:
        num_h = heavy_atom_h_count[heavy_ind]
        scale_idx = min(num_h - 1, 3)  # Cap at index 3 for 4+ hydrogens
        dH = dH_per_H[scale_idx]
        scale_plan.append([heavy_ind, h_ind, dH])

    # Mass before scaling (for verification)
    mass_before = sum(sys.getParticleMass(i) for i in range(sys.getNumParticles()))

    # Apply scaling
    for heavy_ind, h_ind, dH in scale_plan:
        current_h_mass = sys.getParticleMass(h_ind)
        current_heavy_mass = sys.getParticleMass(heavy_ind)
        sys.setParticleMass(h_ind, current_h_mass + dH)
        sys.setParticleMass(heavy_ind, current_heavy_mass - dH)

    # Mass after scaling (should be identical)
    mass_after = sum(sys.getParticleMass(i) for i in range(sys.getNumParticles()))
    mass_delta = mass_after - mass_before

    #print(f"  Scaled {len(scale_plan)} hydrogen bonds")
    print(f"  Mass before: {mass_before:.6f}")
    print(f"  Mass after:  {mass_after:.6f}")
    print(f"  Delta:       {mass_delta:.9f}")

    # Write HMR-adjusted system
    with open(fg_hmr_sys_out, 'w') as f:
        f.write(XmlSerializer.serialize(sys))
    print(f"Wrote {fg_hmr_sys_out}")

    print("\nSuccess!")
    return fg_sys_out, fg_hmr_sys_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pdb_file', type=str, help='Path to PDB file')
    parser.add_argument('xml_file', type=str, help='Path to OpenMM system XML file')

    args = parser.parse_args()

    adjust_system(args.pdb_file, args.xml_file)
    