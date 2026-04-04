import textwrap, sys, os, pathlib, json, warnings
from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem

#OpenFF
import openff
import openff.units
import openff.toolkit
import openff.interchange
#OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *


nonbonded_methods = {'NoCutoff': NoCutoff,
                     'CutoffNonPeriodic': CutoffNonPeriodic,
                     'CutoffPeriodic': CutoffPeriodic,
                     'Ewald': Ewald,
                     'PME': PME}


class ForceFieldHandler():
    """
    Parameterize a structure file and return an OpenMM (System, Topology, Positions) tuple.

    Supports two modes selected automatically from the input file extension:

    * **OpenFF mode** (``.sdf`` input): ligand parameterized with the SMIRNOFF
      force field (``openff-2.1.0.offxml`` by default).
    * **OpenMM mode** (``.pdb`` input): protein/environment parameterized with
      Amber ff14SB + Lipid17 + OPC3 water model by default.

    Parameters
    ----------
    structure_file : str
        Path to an SDF or PDB file containing the structure to parameterize.
    force_field_files : list of str, optional
        Force-field file paths to use instead of (or in addition to) the
        defaults. Extensions must match the mode: ``.offxml`` for SDF input,
        ``.xml`` for PDB input. Default None.
    use_defaults : bool, optional
        If True, include the built-in default force fields alongside any
        files in ``force_field_files``. Default True.

    Attributes
    ----------
    default_xmls : dict
        Mapping of mode name to list of default force-field file paths.
    structure_file : str
        The input file path.
    working_mode : str
        ``'OpenFF'`` or ``'OpenMM'``, determined from the file extension.
    xmls : list of str
        Resolved list of force-field files used during parameterization.

    Examples
    --------
    Parameterize a ligand SDF::

        sys, top, pos = ForceFieldHandler('ligand.sdf').main()

    Parameterize a protein PDB::

        sys, top, pos = ForceFieldHandler('system.pdb').main()
    """
    
    def __init__(self, structure_file, force_field_files=None, use_defaults: bool=True):
        """
        Initialize ForceFieldHandler with a structure file and optional force fields.

        See class docstring for parameter descriptions.
        """
        self.default_xmls = {'OpenFF': ['openff-2.1.0.offxml'], 
                        'OpenMM': ['amber14/protein.ff14SB.xml', 
                                   'amber14/lipid17.xml',
                                   f'{pathlib.Path(__file__).parent.resolve()}/wat_opc3.xml']}
        self.structure_file = structure_file
        
        # Parse the structure file to see if the user is in OpenFF or OpenMM mode
        self.working_mode = self._parse_file(structure_file)
        
        #If force field files were provided, check their extensions, if not use the default
        if force_field_files is None and use_defaults == True:
            self.xmls = self.default_xmls[self.working_mode]
        elif type(force_field_files) != list:
            raise Exception('force_field_files parameter must be specified as a list of strings')
        else:
            mode_parse = [self._parse_file(ff_file) for ff_file in force_field_files]
            mode_check = [elem == self.working_mode for elem in mode_parse]
            if use_defaults == True:
                self.xmls = self.default_xmls['OpenMM']
            else:
                self.xmls = []
            for ff in force_field_files:
                self.xmls.insert(0, ff)

            if False in mode_check:
                bad_index = mode_check.index(False)
                bad_file = force_field_files[bad_index]
                raise Exception(f'{bad_file} was found incompatible with the structure file')

    def _parse_file(self, file_fn):
        """
        Determine the parameterization mode from a file extension.

        SDF and OFFXML files map to ``'OpenFF'``; PDB and XML files map to
        ``'OpenMM'``.

        Parameters
        ----------
        file_fn : str
            Path to the file whose extension will be inspected.

        Returns
        -------
        mode : str
            Either ``'OpenFF'`` or ``'OpenMM'``.

        Raises
        ------
        Exception
            If the file extension is not in the supported set.
        """
        ext = os.path.splitext(file_fn)[-1]
        supported_openff_types = ['.sdf', '.offxml']
        supported_openmm_types = ['.pdb', '.xml']

        if ext in supported_openff_types:
            mode = 'OpenFF'
        elif ext in supported_openmm_types:
            mode = 'OpenMM'
        else:
            raise Exception(f'The extension {ext} was not recognized!')
        return mode

    def main(self, use_pme: bool=True):
        """
        Parameterize the structure and return an OpenMM (System, Topology, Positions) tuple.

        Parameters
        ----------
        use_pme : bool or str, optional
            Nonbonded method for OpenMM mode. ``True`` uses PME (default),
            ``False`` uses the OpenMM default (NoCutoff), or pass a string key
            from ``{'NoCutoff', 'CutoffNonPeriodic', 'CutoffPeriodic',
            'Ewald', 'PME'}`` to select explicitly. Ignored in OpenFF mode.

        Returns
        -------
        result : tuple of (System, Topology, Quantity)
            A 3-tuple of the OpenMM System, Topology, and positions.
        """
        if self.working_mode == 'OpenFF':
            try:
                mol = openff.toolkit.Molecule.from_file(self.structure_file)
            except:
                warnings.warn('WARNING: ATTEMPTING TO ALLOW UNDEFINED STEREOCHEMISTRY. CHECK OUTPUT STRUCTURE CAREFULLY')
                mol = openff.toolkit.Molecule.from_file(self.structure_file, allow_undefined_stereo=True)
            ff = openff.toolkit.ForceField(*self.xmls)
            cubic_box = openff.units.Quantity(30 * np.eye(3), openff.units.unit.angstrom)
            self.interchange = openff.interchange.Interchange.from_smirnoff(topology=[mol], force_field=ff, box=cubic_box)
            positions = np.array(self.interchange.positions) * nanometer
            sys = self.interchange.to_openmm_system()
            top = self.interchange.to_openmm_topology()
    
        elif self.working_mode == 'OpenMM':
            ff = ForceField(*self.xmls)
            pdb = PDBFile(self.structure_file)
            top, positions = pdb.getTopology(), pdb.getPositions()

            if type(use_pme) == str:
                sys = ff.createSystem(top, nonbondedMethod=nonbonded_methods[use_pme])
            
            if use_pme: #This logic block needs to be default
                sys = ff.createSystem(top, nonbondedMethod=PME) 
            else:
                sys = ff.createSystem(top)

        return (sys, top, positions)
