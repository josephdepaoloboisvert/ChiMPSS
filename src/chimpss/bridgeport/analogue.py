import os
from datetime import datetime

import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from MDAnalysis.analysis.rms import rmsd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor

rdDepictor.SetPreferCoordGen(True)
try:
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.drawOptions.minFontSize = 20
    from IPython.display import display
except Exception:
    pass
import warnings
from copy import deepcopy
from typing import List

warnings.filterwarnings("ignore")

from chimpss.bridgeport.ligand import Ligand
from chimpss.bridgeport.ligand_utils import *


class Analogue(Ligand):
    """Prepare an analogue ligand by aligning it to a known experimental template.

    Inherits from :class:`Ligand`.  The typical workflow is:

    1. Call :meth:`get_MCS` to detect the maximum common substructure (MCS)
       between the analogue SMILES and the template's experimental coordinates.
    2. Optionally refine the MCS mapping with ``add_atoms`` / ``remove_atoms``.
    3. Call :meth:`generate_conformers` to produce n 3-D conformers of the
       analogue placed inside the binding site via torsion matching + alignment.

    Parameters
    ----------
    template : Ligand
        The experimental (template) ligand object whose coordinates define the
        binding pose.
    working_dir : str
        Directory where analogue PDB, SDF, and conformer files will be written.
    name : str
        Name for the analogue molecule (no underscores or periods).
    resname : str, optional
        PDB residue name.  Pass ``False`` if not applicable.
    smiles : str, optional
        SMILES string for the analogue.
    chainid : str, optional
        Chain ID if the analogue is a peptide.  Pass ``False`` otherwise.
    sequence : str, optional
        Amino-acid sequence string for peptide analogues.
    verbose : bool, optional
        Print detailed alignment diagnostics.  Default ``False``.
    visualize : bool, optional
        Render 2-D structures and MCS highlights in Jupyter.  Default ``True``.
    """

    def __init__(self, template: Ligand, working_dir: str, name: str,
                     resname: str=False, smiles: str=False,
                     chainid: str=False, sequence: str=False,
                     verbose: bool=False, visualize: bool=True):
        """Initialize an Analogue by extending the parent Ligand constructor."""

        # Initialize inheritated attributes
        super().__init__(working_dir, name, resname, smiles, chainid, sequence, verbose)
        self.visualize = visualize

        # Assign new attributes
        self.template = template
        self.conformer_dir = os.path.join(self.working_dir, f'{name}_conformers')
        if not os.path.exists(self.conformer_dir):
            os.mkdir(self.conformer_dir)



    def get_MCS(self,
                subImgSize: tuple=(600,600),
                add_atoms: List[List[int]]=None,
                remove_atoms: List[int]=None,
                strict: bool=False,
                removeHs: bool=True,
                from_pdb: bool=False,
                from_smiles: bool=True,
                sanitize: bool=True,
                proximityBonding: bool=True):
        """Detect the maximum common substructure between analogue and template.

        Builds rdkit molecules for both ligands, runs MCS detection, and
        optionally adjusts the mapping with user-supplied atom index lists.
        When ``visualize=True`` (set on the instance) the analogue and template
        are rendered side-by-side with matched atoms highlighted.

        Parameters
        ----------
        subImgSize : tuple, optional
            Pixel dimensions ``(width, height)`` for the 2-D structure grid image.
        add_atoms : list of [int, int], optional
            Extra atom-pair mappings to append to the automatic MCS.  Each
            element is ``[analogue_idx, template_idx]``.
        remove_atoms : list of int, optional
            Analogue atom indices to drop from the automatic MCS mapping.
        strict : bool, optional
            Require strict atom-type matching in the MCS search.  Default ``False``.
        removeHs : bool, optional
            Strip explicit hydrogens before MCS detection.  Default ``True``.
        from_pdb : bool, optional
            Load the analogue molecule from its PDB file.  Default ``False``.
        from_smiles : bool, optional
            Build the analogue molecule from ``self.smiles``.  Default ``True``.
        sanitize : bool, optional
            Sanitize the rdkit molecule after loading.  Default ``True``.
        proximityBonding : bool, optional
            Use proximity-based bonding when reading from PDB.  Default ``True``.
        """

        # Set attributes
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.proximityBonding = proximityBonding

        # Get rdkit molecules
        mol = self.return_rdkit_mol(from_pdb=from_pdb,
                                  from_smiles=from_smiles,
                                  sanitize=self.sanitize,
                                  removeHs=self.removeHs,
                                  proximityBonding=self.proximityBonding)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created analogue', self.name, 'from smiles:', self.smiles , flush=True)

        template_mol = self.template.return_rdkit_mol(from_pdb=True,
                          from_smiles=False,
                          sanitize=True,
                          removeHs=False,
                          proximityBonding=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Created known ligand', self.template.name, 'from smiles:', self.template.smiles , flush=True)

        # Detect MCS
        self.strict = strict
        self._detect_MCS(mol, template_mol)
        self.matching_inds = deepcopy(self.align_inds)
        self.template_matching_inds = deepcopy(self.template_align_inds)

        # Remove user specified atoms
        if remove_atoms is not None:
            self._remove_atoms_from_MCS(remove_atoms)

        # Add user specified atoms
        if add_atoms is not None:
            self._add_atoms_to_MCS(add_atoms)

        # Print matching atoms
        if self.verbose:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Found matching inds:', flush=True)
            for (atom, ref_atom) in zip(self.matching_inds, self.template_matching_inds):
                print(f'atom={atom}, ref_atom={ref_atom}')

        # Draw molecules
        if self.visualize:
            template_mol_copy = Chem.Mol(self.template_mol)
            Chem.rdDepictor.Compute2DCoords(template_mol_copy)
            dopts = Chem.Draw.rdMolDraw2D.MolDrawOptions()
            dopts.addAtomIndices = True
            print('Analogue, Template')
            display(Draw.MolsToGridImage([self.mol, template_mol_copy],
                                         subImgSize=subImgSize,
                                         highlightAtomLists=[self.matching_inds, self.template_matching_inds],
                                         drawOptions=dopts))



    def generate_conformers(self, n_conformers: int=1, align_all: bool=False, rmsd_thresh: float=3.0):
        """Generate analogue conformers placed inside the template binding site.

        Embeds the analogue molecule, matches internal coordinates to the
        template's MCS atoms, and aligns the result.  Only conformers whose
        RMSD to the template alignment atoms falls below ``rmsd_thresh`` are
        accepted.  Accepted conformers are written to ``self.conformer_dir``;
        the first conformer is also saved as ``self.pdb``.

        Parameters
        ----------
        n_conformers : int, optional
            Number of accepted conformers to generate.  Default ``1``.
        align_all : bool, optional
            If ``True``, use the full MCS (``matching_inds``) for the final
            superposition step rather than the auto-detected alignment subset.
            Default ``False``.
        rmsd_thresh : float, optional
            Maximum RMSD (Å) between aligned analogue and template atoms for
            a conformer to be accepted.  Default ``3.0``.
        """

        # Run setup methods
        self._load_molecules(load_template=True)

        # Get MDA atoms
        if align_all:
            self.align_inds = deepcopy(self.matching_inds)
            self.template_align_inds = deepcopy(self.template_matching_inds)

        self._get_MDA_atoms()


        # Iterate for the n_conformers
        n=0
        conformer = 0
        while n < n_conformers:

            # Generate conformer
            self._load_molecules()

            # Make selections
            self._make_selections()

            # Save MCS
            self.bat_pdb = os.path.join(self.working_dir, f'{self.name}_mcs.pdb')
            self.template_matching_sele.write(self.bat_pdb)
            self.template_matching_sele = mda.Universe(self.bat_pdb).select_atoms('all')

            # Match internal coordinates
            self.sele = match_internal_coordinates(ref_match=self.template_matching_sele,
                                       ref_match_atoms=self.template_matching_atoms,
                                       ref_match_resids=self.template_matching_resids,
                                       mobile=self.sele,
                                       mobile_match_atoms=self.matching_atoms,
                                       verbose=self.verbose)

            # Align
            alignto(mobile=self.align_sele, reference=self.template_align_sele, tol_mass=1000)

            # Save if RMSD is below threshold
            RMSD = rmsd(self.align_sele.positions.copy(), self.template_align_sele.positions.copy())
            if RMSD <= rmsd_thresh:
                conformer_out_pdb = os.path.join(self.conformer_dir, f'{self.name}_{n}.pdb')
                self.sele.write(conformer_out_pdb)
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved conformer to', conformer_out_pdb, flush=True)

                # Up that ticker :)
                n += 1
            conformer += 1

        # Save final structure
        self.sele.write(self.pdb)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//Saved first conformer to', self.pdb, flush=True)


    def _detect_MCS(self, mol1, mol2):
        """
        Return indices of maximum common substructure between two rdkit molecules

        mol1 should be analogue, mol2 should be template

        """

        # Get MCS
        self.align_inds, self.template_align_inds = get_rdkit_MCS(mol1, mol2, strict=self.strict)

        # Set final attributes
        self.mol = mol1
        self.template_mol = mol2


    def _load_molecules(self, load_template: bool=False):
        """Embed the analogue rdkit molecule and optionally reload the template.

        Parameters
        ----------
        load_template : bool, optional
            If ``True``, also reload ``self.template_mol`` from its PDB file
            and build ``self.template_sele`` as an MDAnalysis AtomGroup.
        """
        # Store bond orders from SMILES and save to .pdb for MDAnalysis
        self.mol = embed_rdkit_mol(self.mol, self.mol)
        Chem.MolToPDBFile(self.mol, self.pdb)

        # Load template from pdb
        if load_template:
            self.template_mol = self.template.return_rdkit_mol(from_pdb=True,
                                                               from_smiles=False,
                                                               sanitize=True,
                                                               removeHs=False, # Changed to False for MutatedPeptide to work, proceed w/ caution
                                                               proximityBonding=True)

            # Load with MDAnalysis
            self.template_sele = self.template.return_MDA_sele()



    def _get_MDA_atoms(self):
        """Translate rdkit atom indices to MDAnalysis atom names and residue IDs.

        Populates ``align_atoms``, ``template_align_atoms``,
        ``matching_atoms``, and ``template_matching_atoms`` / residue id
        counterparts used by :meth:`_make_selections`.
        """
        # Translate atoms for alignment
        self.align_atoms, _ = translate_rdkit_inds(self.mol, self.align_inds)
        self.template_align_atoms, self.template_align_resids = translate_rdkit_inds(self.template_mol, self.template_align_inds)

        # Translate atoms for torsion matching
        self.matching_atoms, _ = translate_rdkit_inds(self.mol, self.matching_inds)
        self.template_matching_atoms, self.template_matching_resids = translate_rdkit_inds(self.template_mol, self.template_matching_inds)




    def _add_atoms_to_MCS(self, add_atoms):
        """Append user-specified atom pairs to the MCS mapping.

        Parameters
        ----------
        add_atoms : list of [int, int]
            Each element is ``[analogue_idx, template_idx]`` to be added to
            ``self.matching_inds`` and ``self.template_matching_inds``.
        """

        # Add atoms
        for atom, temp_atom in add_atoms:

                # Add
                self.matching_inds.append(atom)
                self.template_matching_inds.append(temp_atom)



    def _remove_atoms_from_MCS(self, remove_atoms):
        """Remove user-specified analogue atoms from the MCS mapping.

        Parameters
        ----------
        remove_atoms : list of int
            Analogue atom indices to drop from ``self.matching_inds``; the
            paired template atom is removed from
            ``self.template_matching_inds`` automatically.
        """
        # Remove user specified atoms
        for atom in remove_atoms:

            # See if already in there
            if atom in self.matching_inds:

                # Find atoms
                atom_ind = self.matching_inds.index(atom)

                # Remove
                self.matching_inds.pop(atom_ind)
                self.template_matching_inds.pop(atom_ind)



    def _make_selections(self):
        """Build MDAnalysis AtomGroup selections for alignment and torsion matching.

        Populates ``self.sele``, ``self.align_sele``,
        ``self.template_align_sele``, ``self.matching_sele``, and
        ``self.template_matching_sele`` from the atom name/residue ID lists
        set by :meth:`_get_MDA_atoms`.
        """
        # Make selections
        self.sele = self.return_MDA_sele()
        self.align_sele = select(self.sele, self.align_atoms)
        self.template_align_sele = select(self.template_sele, self.template_align_atoms, self.template_align_resids)
        self.matching_sele = select(self.sele, self.matching_atoms)
        self.template_matching_sele = select(self.template_sele, self.template_matching_atoms, self.template_matching_resids)



    def visualize_alignment(self):
        """Render a 3-D overlay of the analogue (blue) and template (yellow) in py3Dmol."""
        import py3Dmol
        view = py3Dmol.view()
        print(f'BLUE: {self.name}')
        print(f'YELLOW: {self.template.name}')
        view.setBackgroundColor('white')
        view.addModel(open(self.pdb, 'r').read(),'pdb')
        view.addModel(open(self.template.pdb, 'r').read(),'pdb')
        view.setStyle({'model':0}, {'stick': {'colorscheme':'blueCarbon'}})
        view.setStyle({'model':1}, {'stick': {'colorscheme':'yellowCarbon'}})
        view.zoomTo()
        view.show()

