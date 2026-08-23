import os
from datetime import datetime

import MDAnalysis as mda
import mdtraj as md
import numpy as np
from openmm import *
from openmm import unit

#OpenMM
from openmm.app import *
from openmm.app.pdbfile import PDBFile
from pdbfixer import PDBFixer


class ProteinPreparer():
    """
    The purpose of this class is to prepare a crystal structure for molecular simulation. Steps include
        Addition of Missing atoms and side chains
        Protonation at user specified pH
        Creation of environment (Membrane and Solvent, Solvet) at user specified ion conc

    Attributes:
    ------------
        receptor_path (str):
            The path to the PDB file that contains the protein structure intended for preparation.

        prot_name (str):
            The name of the protein, extracted from the input file path, which is used in naming the output files.

        working_dir (str):
            The directory designated for storing all the intermediate and output files.

        pH (float):
            Specifies the pH level at which the protein is to be protonated; the default value is 7.

        env (str):
            Defines the type of environment to be created around the protein; 'MEM' indicates a combination of a membrane and solvent, while 'SOL' specifies a solvent-only environment.

        ion (float):
            The ionic strength of the solvent, measured in molar units, with a default value of 0.15 M NaCl.

    Methods:
    ---------
        init(pdb_path, working_dir: str, pH=7, env='MEM', ion_strength=0.15):
            Initializes the ProteinPreparer class with specified parameters for protein preparation, including the path to the PDB file, working directory, pH level for protonation, environmental setup (membrane and solvent or solvent only), and ion strength.

        main():
            Coordinates the main workflow for preparing the protein structure for molecular simulation. This includes steps for protonation, adding missing atoms and side chains, and setting up the simulation environment as specified by the user.

        _protonate_with_pdb2pqr(at_pH=7):
            Protonates the protein structure at the specified pH using pdb2pqr30, which also adds missing atoms but not missing residues. Outputs file paths to the protonated pdb and pqr files.

        _protonate_with_PDBFixer(at_pH=7):
            An alternative method to protonate the protein using PDBFixer, which can add missing hydrogens at the specified pH. This method is used if pdb2pqr30 is not employed or for additional protonation adjustments.

        _run_PDBFixer(mode: str = "MEM", out_file_fn: str = None, padding = 1.5, ionicStrength = 0.15):
            Generates a solvated and possibly membrane-added system based on the specified mode. It uses PDBFixer to create an environment around the protein, either with just solvent or with both membrane and solvent, according to the user's choice.
    """

    def __init__(self, pdb_path, working_dir: str, pH=7, env='MEM', ion_strength=0.15, verbose: bool=False, extra_charge: int=0):
        """
        Parameters:
            pdb_path: string path to protein structure file
            pH: default 7: pH to protonate at
            env: default 'MEM': 'MEM' for explicit membrane and solvent
                                'SOL' for explicit solvent only
                                no other modes supported
            extra_charge: formal charge of any solute that is NOT present during
                          solvation but will be joined afterwards -- in practice
                          the ligand. PDBFixer neutralizes only what it can see,
                          so without this the finished system carries the
                          ligand's charge.
        """
        self.receptor_path = pdb_path
        self.prot_name = self.receptor_path.split('.')[0]
        try:
            self.prot_name = self.prot_name.split('/')[-1]
        except Exception:
            pass
        self.working_dir = working_dir
        self.pH = pH
        self.env = env
        self.ion = ion_strength
        self.verbose = verbose
        self.extra_charge = int(extra_charge)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Welcome to ProteinPreparer', flush=True)

    def main(self):

        # Protonate (with pdb2pqr30 which also adds missing atoms, but not missing residues)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Protonating protein with pdb2pqr30', flush=True)
        self.H_pdb = self._protonate_with_pdb2pqr(at_pH=self.pH)

        # Check completion
        if os.path.exists(self.H_pdb):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Output written to:', self.H_pdb, flush=True)

        #Create Environment
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Creating environment with pdbfixer', flush=True)
        self._run_PDBFixer(mode=self.env, padding=1.5, ionicStrength=self.ion)

        # Check completion
        if os.path.exists(self.env_pdb):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Output written to:', self.env_pdb, flush=True)


        return self.env_pdb


    def _protonate_with_pdb2pqr(self, at_pH=7):
        """
        Protonates the given structure using pdb2pqr30 on the linux command line

        Parameters:
        -----------
            at_pH: pH for protonation (default 7)

        Returns:
        --------
            2-tuple = (protein_H_fn, protein_pqr_fn); file paths (as strings) to the protonated pdb and pqr file
        """
        self.H_pdb = os.path.join(self.working_dir, self.prot_name + '_H.pdb')
        protein_pqr_path = os.path.join(self.working_dir, self.prot_name + '.pqr')
        # my_cmd = f'pdb2pqr30 --ff AMBER --log-level CRITICAL --nodebump --keep-chain --ffout AMBER --pdb-output {self.H_pdb} --with-ph {at_pH} {self.receptor_path} {protein_pqr_path}'
        my_cmd = f'pdb2pqr30 --ff AMBER --nodebump --keep-chain --ffout AMBER --pdb-output {self.H_pdb} --with-ph {at_pH} {self.receptor_path} {protein_pqr_path}'
        print('Protanting using command line')
        print(f'Running {my_cmd}')
        os.system(my_cmd)
        return self.H_pdb

    def _protonate_with_PDBFixer(self, at_pH=7):
        if not hasattr(self, "protein_H_path"):
            H_pdb_path = os.path.join(self.working_dir, self.prot_name + '_H.pdb')
            if os.path.exists(H_pdb_path):
                self.H_pdb = H_pdb_path
            else:
                self.H_pdb = self.receptor_path
        fixer = PDBFixer(self.H_pdb)
        # fixer.findMissingResidues()
        # fixer.findMissingAtoms()
        # print('!!!missingTerminals', fixer.missingAtoms)
        # fixer.addMissingAtoms()
        fixer.addMissingHydrogens(at_pH)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.H_pdb, 'w'), keepIds=True)

        return self.H_pdb

    def _run_PDBFixer(self,
                      mode: str = "MEM",
                      out_file_fn: str = None,
                      padding = 2.0,
                      ionicStrength = 0.15):
        """
        Generates a solvated and membrane-added system (depending on MODE)
        MODE = 'MEM' for membrane and solvent
        MODE = 'SOL' for solvent only
        Parameters:
            mode: string: must be in ['MEM', 'SOL']
            solvated_file_fn: Filename to save solvated system; default is to add '_solvated' between the body and extension of the input file name
            padding: float or int: minimum nanometer distance between the boundary and any atoms in the input.  Default 1.5 nm = 15 A
            ionicStrength: float (not int) : molar strength of ionic solution. Default 0.15 M = 150 mmol
        Returns:
            solvated_file_fn: the filename of the solvated output
        """
        assert mode in ['MEM', 'SOL']
        self.fixer = PDBFixer(self.H_pdb)

        if mode == 'MEM':
            self.fixer.addMembrane('POPC', minimumPadding=padding * unit.nanometer, ionicStrength=ionicStrength * unit.molar)
        elif mode == 'SOL':
            self.fixer.addSolvent(padding=padding * unit.nanometer, ionicStrength=ionicStrength * unit.molar)

        # ADD hydrogens
        self.fixer.addMissingHydrogens()

        # Trim
        self._trim_env()



    def _trim_env(self, padding: float=15):
        """
        Remove the excess membrane and solvent added by calling PDBFixer.addMembrane()

        Protocol:
        ---------
            1. Get dimensions of protein and new periodic box
            2. Write corresponding CRYST1 line
            3. Identify atoms outside of box
            4. Identify corresponding resnames and resids outside of box
            5. Remove residues outside of box
            6. Overwrite original file ('pdb' parameter)

        Parameters:
        -----------
            pdb (str):
                String path to pdb file to trim.

            padding (float):
                Amount of padding (Angstrom) to trim to. Default is 15 Angstrom to accomodate the default 10 Angstrom NonBondededForce cutoff.
        """

        # Copy the fixer topology to a new topology,
        # excluding residues with atoms outside of the box
        def copyTopology(topology_o, residx_to_exclude = None):
          from openmm.app.topology import Topology
          topology_n = Topology()

          atom_indices = []

          chains = {}
          residues = {}
          atoms = {}
          # Add chains
          for c_o in topology_o.chains():
            c_n = topology_n.addChain()
            chains[c_o.id] = c_n
            # Add residues that are not excluded
            for r_o in c_o.residues():
              if r_o.index not in residx_to_exclude:
                r_n = topology_n.addResidue(r_o.name, c_n)
                residues[r_o.id] = r_n
                for a_o in r_o.atoms():
                  a_n = topology_n.addAtom(a_o.name, a_o.element, r_n)
                  atoms[a_o.id] = a_n
                  atom_indices.append(a_o.index)
          # Add bonds
          for b_o in topology_o.bonds():
            if (b_o.atom1.index in atom_indices) and (b_o.atom2.index in atom_indices):
              topology_n.addBond(atoms[b_o.atom1.id], atoms[b_o.atom2.id], \
                                 b_o.type, b_o.order)
          return (topology_n, atom_indices)



        u = mda.Universe(self.H_pdb)
        prot_sele = u.select_atoms('protein')
        max_coords = (np.array([prot_sele.positions[:,i].max() for i in range(3)]) + padding) * unit.angstroms
        min_coords = (np.array([prot_sele.positions[:,i].min() for i in range(3)]) - padding) * unit.angstroms

        # Identify residues with atoms outside of the box
        residx_out_of_box = set([a.residue.index for (a, p) in zip(self.fixer.topology.atoms(), self.fixer.positions.in_units_of(unit.angstroms)) \
         if (p<min_coords).any() or (p>max_coords).any()])

        (topology, atom_indices) = copyTopology(self.fixer.topology, \
                                                residx_to_exclude = residx_out_of_box)

        # Rebalance Na/Cl WITHOUT destroying neutrality.
        #
        # PDBFixer.addMembrane() already neutralized the system: it adds salt in
        # PAIRS to reach the requested ionic strength and then an EXCESS of one
        # counterion species to cancel the solute's net charge.  That imbalance
        # is the neutralization -- it is not stray salt to be tidied away.
        #
        # This block previously forced len(Na) == len(Cl), which discarded every
        # counterion and left each finished system sitting at the protein's net
        # charge (+7 to +14 across the five ChiMPSS systems).  Under PME that is
        # absorbed by a uniform neutralizing background rather than raising, so
        # it never surfaced as an error.
        #
        # Instead: measure the imbalance addMembrane created, offset it by the
        # charge of any solute still to be joined (the ligand), and preserve that
        # difference through the trim, removing only the minimum number of ions
        # needed -- which also keeps the ionic strength as close as possible to
        # the requested value.
        top = md.Topology.from_openmm(topology)
        positions = np.array(self.fixer.positions.value_in_unit(unit.angstroms))[atom_indices]
        if self.verbose:
            print('topology n_atoms', top.n_atoms, 'pos n_atoms', positions.shape[0])

        na_all = sum(1 for a in self.fixer.topology.atoms() if a.name.strip() == 'Na')
        cl_all = sum(1 for a in self.fixer.topology.atoms() if a.name.strip() == 'Cl')
        target_diff = (na_all - cl_all) - self.extra_charge

        na_idx = [int(i) for i in top.select('name Na')]
        cl_idx = [int(i) for i in top.select('name Cl')]
        excess = (len(na_idx) - len(cl_idx)) - target_diff

        # Correct the imbalance by CONVERTING ions, not deleting them: relabelling
        # one Na+ as Cl- moves the difference by 2 while leaving the total ion
        # count -- and therefore the ionic strength -- untouched.  Deleting
        # instead would drain the salt (it put 5HT2AR at 121 mM against a 150 mM
        # target).  Nothing is parameterized yet at this point, so a relabel here
        # is free.  Parity permits only even steps, so an odd excess needs one
        # single ion removed as well.
        convert, remove_atoms = [], []
        if excess > 0:                      # too many Na+ -> turn some into Cl-
            k = excess // 2
            convert = [('Cl', i) for i in na_idx[len(na_idx) - k:]] if k else []
            na_idx = na_idx[:len(na_idx) - k]
            cl_idx = cl_idx + [i for _, i in convert]
            if excess % 2:
                remove_atoms.append(na_idx.pop())
        elif excess < 0:                    # too many Cl- -> turn some into Na+
            k = (-excess) // 2
            convert = [('Na', i) for i in cl_idx[len(cl_idx) - k:]] if k else []
            cl_idx = cl_idx[:len(cl_idx) - k]
            na_idx = na_idx + [i for _, i in convert]
            if excess % 2:
                remove_atoms.append(cl_idx.pop())

        # Second pass: the trim removed waters and ions by geometry, so the ion
        # count may no longer match the requested ionic strength.  Drop whole
        # NaCl PAIRS -- which leaves the neutralizing difference untouched -- to
        # get as close as possible.  Following the convention addMembrane itself
        # uses, concentration is per water (n_ion = 2*c*n_water/55.4), not per box
        # volume, since the box is mostly lipid and protein that hold no ions.
        # This can only remove: if the trim already left too few ions, the count
        # simply stays below target.
        n_wat = sum(1 for r in top.residues if r.name.strip() in ('HOH', 'WAT'))
        if n_wat > 0:
            desired = 2.0 * self.ion * n_wat / 55.4
            total = len(na_idx) + len(cl_idx)
            n_pairs = 0
            while (total - 2 * (n_pairs + 1) >= abs(target_diff)
                   and abs(total - 2 * (n_pairs + 1) - desired) < abs(total - 2 * n_pairs - desired)):
                n_pairs += 1
            if n_pairs:
                remove_atoms += na_idx[len(na_idx) - n_pairs:]
                remove_atoms += cl_idx[len(cl_idx) - n_pairs:]
                na_idx = na_idx[:len(na_idx) - n_pairs]
                cl_idx = cl_idx[:len(cl_idx) - n_pairs]

        # Apply the relabels (skip any ion that is about to be deleted).
        dropped = set(remove_atoms)
        for species, i in convert:
            if i in dropped:
                continue
            atom = top.atom(i)
            atom.name = species
            atom.element = md.element.chlorine if species == 'Cl' else md.element.sodium
            atom.residue.name = species.upper()

        if self.verbose:
            ionic = 55.4 * (len(na_idx) + len(cl_idx)) / (2 * n_wat) if n_wat else 0.0
            print(f'ion balance: addMembrane gave Na-Cl={na_all - cl_all}, '
                  f'ligand charge={self.extra_charge}, target Na-Cl={target_diff}; '
                  f'converted {len(convert)}, removed {len(remove_atoms)} '
                  f'-> Na={len(na_idx)} Cl={len(cl_idx)}, net ion charge '
                  f'{len(na_idx) - len(cl_idx):+d}, ionic strength {ionic:.4f} M')
        remove_atoms = np.array(sorted(remove_atoms), dtype=int)
        if self.verbose:
            print('remove_atoms', remove_atoms)
        for i, atom in enumerate(sorted(remove_atoms)):
            top.delete_atom_by_index(atom - i)
        positions = np.delete(positions, remove_atoms, axis=0)

        if self.verbose:
            print('topology n_atoms', top.n_atoms, 'pos n_atoms', positions.shape[0])


        topology = top.to_openmm()
        topology.setUnitCellDimensions(max_coords - min_coords)

        # Save file
        if not hasattr(self, "env_pdb"):
            self.env_pdb = os.path.join(self.working_dir, self.prot_name + '_env.pdb')

        with open(self.env_pdb, "w") as F:
          PDBFile.writeFile(topology, positions, F, keepIds=True)







