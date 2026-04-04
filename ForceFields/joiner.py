import textwrap, sys, os
import numpy as np
#OpenFF
import openff
import openff.units
import openff.toolkit
import openff.interchange
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *

class Joiner():
    """
    Combine two OpenMM (System, Topology, Positions) sets into one.

    Intended for merging an OpenFF-parameterized ligand system with an
    OpenMM-parameterized receptor/environment system.

    Parameters
    ----------
    Ligand_Set : tuple of (System, Topology, Quantity)
        The ligand (System, Topology, Positions) 3-tuple to be added.
    Receptor_Set : tuple of (System, Topology, Quantity)
        The receptor/environment (System, Topology, Positions) 3-tuple.

    Attributes
    ----------
    lig_sys, lig_top, lig_pos : System, Topology, Quantity
        Components of the ligand set.
    rec_sys, rec_top, rec_pos : System, Topology, Quantity
        Components of the receptor set.
    """
    
    def __init__(self, Ligand_Set, Receptor_Set):
        """
        Store ligand and receptor sets for subsequent joining.

        See class docstring for parameter descriptions.
        """
        assert type(Ligand_Set) == type(Receptor_Set) and len(Ligand_Set) == len(Receptor_Set)
        assert False not in [type(Ligand_Set[i]) == type(Receptor_Set[i]) for i in range(3)]

        self.lig_sys, self.lig_top, self.lig_pos = Ligand_Set
        self.rec_sys, self.rec_top, self.rec_pos = Receptor_Set


    def main(self):
        """
        Combine ligand and receptor into a single (System, Topology, Positions) tuple.

        Returns
        -------
        result : tuple of (System, Topology, Quantity)
            The joined (System, Topology, Positions) 3-tuple.
        """
        comp_top, comp_positions = self._join_two_topologies((self.rec_top, self.lig_top), (self.rec_pos, self.lig_pos))
        comp_sys = self._join_two_systems(self.lig_sys, self.rec_sys)
        return comp_sys, comp_top, comp_positions
    
    def _join_two_topologies(self, tops: tuple, poss: tuple):
        """
        Merge two topologies and their associated positions arrays.

        The second topology/positions (index 1) are appended onto the first
        (index 0).

        Parameters
        ----------
        tops : tuple of (Topology, Topology)
            Two-tuple of OpenMM Topology objects to merge.
        poss : tuple of (Quantity, Quantity)
            Two-tuple of position arrays corresponding to ``tops``.

        Returns
        -------
        topology : Topology
            Merged OpenMM Topology.
        positions : Quantity
            Concatenated positions array.
        """
        assert len(tops) == 2 and len(poss) == 2
        modeller = Modeller(tops[0], poss[0])
        modeller.add(tops[1], poss[1])
        return modeller.topology, modeller.positions

    def _join_two_systems(self, sys1: System, sys2: System):
        """
        Add all force terms from ``sys1`` (ligand) into ``sys2`` (receptor).

        Transfers particles, constraints, NonbondedForce, PeriodicTorsionForce,
        HarmonicAngleForce, and HarmonicBondForce parameters from ``sys1`` to
        ``sys2``. Handles the naming discrepancy between OpenFF (``'Nonbonded
        force'``) and OpenMM (``'NonbondedForce'``) automatically.

        Parameters
        ----------
        sys1 : System
            OpenMM System produced by OpenFF interchange (ligand). Its
            parameters are copied into ``sys2``.
        sys2 : System
            OpenMM System produced by OpenMM (receptor). Modified in-place.

        Returns
        -------
        sys2 : System
            The receptor System with all ligand parameters added.
        """
        #Particles
        new_particle_indices = []
        for i in range(sys1.getNumParticles()):
            new_particle_indices.append(sys2.addParticle(sys1.getParticleMass(i)))
        
        #Contstraints (mostly wrt hydrogen distances)
        for i in range(sys1.getNumConstraints()):
            params = sys1.getConstraintParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            sys2.addConstraint(*params)
        
        #NonBonded
        sys1_force_name = 'Nonbonded force'
        sys2_force_name = 'NonbondedForce'

        try:
            force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
            sys1_force = sys1.getForces()[force_ind]
        except:
            force_ind = [force.getName() for force in sys1.getForces()].index(sys2_force_name)
            sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumParticles()):
            params = sys1_force.getParticleParameters(i)
            sys2_force.addParticle(*params)
        
        for i in range(sys1_force.getNumExceptions()):
            params = sys1_force.getExceptionParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            sys2_force.addException(*params)
    
        #Torsion
        sys1_force_name = 'PeriodicTorsionForce'
        sys2_force_name = 'PeriodicTorsionForce'
        
        force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
        sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumTorsions()):
            params = sys1_force.getTorsionParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            params[2] = new_particle_indices[params[2]]
            params[3] = new_particle_indices[params[3]]
            sys2_force.addTorsion(*params)
    
        #Angle
        sys1_force_name = 'HarmonicAngleForce'
        sys2_force_name = 'HarmonicAngleForce'
        
        force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
        sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumAngles()):
            params = sys1_force.getAngleParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            params[2] = new_particle_indices[params[2]]
            sys2_force.addAngle(*params)
    
        #Bond
        sys1_force_name = 'HarmonicBondForce'
        sys2_force_name = 'HarmonicBondForce'
        
        force_ind = [force.getName() for force in sys1.getForces()].index(sys1_force_name)
        sys1_force = sys1.getForces()[force_ind]
        
        force_ind = [force.getName() for force in sys2.getForces()].index(sys2_force_name)
        sys2_force = sys2.getForces()[force_ind]
        
        for i in range(sys1_force.getNumBonds()):
            params = sys1_force.getBondParameters(i)
            params[0] = new_particle_indices[params[0]]
            params[1] = new_particle_indices[params[1]]
            sys2_force.addBond(*params)
        
        return sys2