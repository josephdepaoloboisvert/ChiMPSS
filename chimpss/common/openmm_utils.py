"""
Consolidated OpenMM utility functions for ChiMPSS.

Merges:
  MotorRow_utils.py      (get_positions_from_pdb, restrain_atoms,
                          unpack_infiles, parse_atom_inds, minimize_from_sys)
  Minimizer_utils.py     (identical functions — deleted)
  utils/utils.py         (describe_system, describe_state, change_resname)
  FultonMarketUtils.py   (convert_to_TrackedQuantity, build_sampler_states)
"""

import numpy as np
from openmm import Vec3, System, State, XmlSerializer, CustomExternalForce
from openmm.app import PDBFile, Simulation
from openmm.unit import kelvin, picosecond, femtosecond
import openmm.unit as unit
from openmmtools.states import SamplerState
from openmmtools.utils.utils import TrackedQuantity


# ---------------------------------------------------------------------------
#  PDB parsing
# ---------------------------------------------------------------------------

def get_positions_from_pdb(fname_pdb, lig_resname=None, lig_chain=None):
    """
    Parse coordinates and heavy-atom indices from a PDB file.

    Parameters
    ----------
    fname_pdb : str
        Path to the PDB file.
    lig_resname : str, optional
        Residue name used to identify ligand atoms.
    lig_chain : str, optional
        Chain identifier used to identify ligand atoms.

    Returns
    -------
    coords : np.ndarray
        (N, 3) array of atomic coordinates.
    prt_heavy_atoms : list[int]
        Indices of protein heavy atoms.
    mem_heavy_atoms : list[int]
        Indices of membrane heavy atoms.
    lig_heavy_atoms : list[list]
        ``[[atom_index, atom_name], ...]`` for ligand heavy atoms.
    """
    nameMembrane = ['DPP', 'POP']
    with open(fname_pdb, 'r') as f_pdb:
        l_pdb = f_pdb.read().split('\n')

    coords = []
    prt_heavy_atoms = []
    mem_heavy_atoms = []
    lig_heavy_atoms = []
    iatom = 0

    for line in l_pdb[:-1]:
        if line[:6] in ['ATOM  ', 'HETATM']:
            resname = line[17:20]
            chain = line[21:23].strip()

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = str(line[76:78].replace('', ''))

            coords.append(Vec3(x, y, z))

            if line[17:20] in nameMembrane and element != 'H':
                mem_heavy_atoms.append(iatom)
            elif line[:6] in ['ATOM  '] and element != 'H':
                if ((lig_resname is not None and resname == lig_resname)
                        or (lig_chain is not None and chain == lig_chain)) and element != 'H':
                    lig_atom_name = line[12:16].strip().strip('x')
                    lig_heavy_atoms.append([iatom, lig_atom_name])
                else:
                    prt_heavy_atoms.append(iatom)
            elif ((lig_resname is not None and resname == lig_resname)
                    or (lig_chain is not None and chain == lig_chain)) and element != 'H':
                lig_atom_name = line[12:16].strip().strip('x')
                lig_heavy_atoms.append([iatom, lig_atom_name])

            iatom += 1

    return np.array(coords), prt_heavy_atoms, mem_heavy_atoms, lig_heavy_atoms


# ---------------------------------------------------------------------------
#  Restraints
# ---------------------------------------------------------------------------

def restrain_atoms(system, crds, atom_inds, rst_name='fc_pos', restraint_strength=20.0):
    """
    Add a flat-bottom harmonic positional restraint to selected atoms.

    Parameters
    ----------
    system : openmm.System
    crds : array-like
        Coordinates (Angstrom) from which reference positions are taken.
    atom_inds : array-like of int
        Atom indices to restrain.
    rst_name : str
        Name of the global restraint-strength parameter in the force
        expression.  Default ``'fc_pos'``.
    restraint_strength : float
        Restraint force constant.  Default ``20.0``.

    Returns
    -------
    system : openmm.System
    """
    rest = CustomExternalForce(f'{rst_name}*periodicdistance(x,y,z,x0,y0,z0)^2')
    rest.addGlobalParameter(rst_name, restraint_strength)
    rest.addPerParticleParameter('x0')
    rest.addPerParticleParameter('y0')
    rest.addPerParticleParameter('z0')
    for atom_i in atom_inds:
        x, y, z = crds[int(atom_i)] / 10
        rest.addParticle(int(atom_i), [x, y, z])
    system.addForce(rest)

    return system


# ---------------------------------------------------------------------------
#  System / state I/O
# ---------------------------------------------------------------------------

def unpack_infiles(xml, pdb):
    """
    Deserialize an OpenMM System from XML and load a PDB file.

    Returns
    -------
    system : openmm.System
    topology : openmm.app.Topology
    positions : list[Vec3]
    """
    print(f'Unpacking {xml}, {pdb}')
    pdb = PDBFile(pdb)
    with open(xml) as f:
        system = XmlSerializer.deserialize(f.read())
    return system, pdb.topology, pdb.positions


def describe_system(sys):
    """Print box vectors, forces, and particle count for a System."""
    box_vecs = sys.getDefaultPeriodicBoxVectors()
    print('Box Vectors')
    for bv in box_vecs:
        print(bv)
    forces = sys.getForces()
    print('Forces')
    for force in forces:
        print(force)
    num_particles = sys.getNumParticles()
    print(num_particles, 'Particles')


def describe_state(state, name="State"):
    """Print potential energy and max force for an OpenMM State."""
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
          f"with maximum force {round(max_force, 2)} kJ/(mol nm)")


def _describe_state_from_sim(sim, name="State"):
    """
    Helper: extract a State from a Simulation and print its energy/forces.

    Returns
    -------
    pe : float
        Potential energy in kJ/mol.
    """
    state = sim.context.getState(getEnergy=True, getForces=True)
    pe = round(state.getPotentialEnergy()._value, 2)
    max_force = round(max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces()), 2)
    print(f"{name} has energy {pe} kJ/mol  with maximum force {max_force} kJ/(mol nm)")
    return pe


# ---------------------------------------------------------------------------
#  Atom index helpers
# ---------------------------------------------------------------------------

def parse_atom_inds(atom_inds, parse_atom_names, find_atom_names):
    """
    Map atom indices from one name list to another.

    Parameters
    ----------
    atom_inds : array-like
        Source atom indices (paired with *parse_atom_names*).
    parse_atom_names : list[str]
        Atom names corresponding to *atom_inds*.
    find_atom_names : list[str]
        Target atom names to populate.

    Returns
    -------
    parsed_atom_inds : np.ndarray
    """
    parsed_atom_inds = np.empty(len(find_atom_names), dtype=int)
    for atom_i, parse_atom_name in zip(atom_inds, parse_atom_names):
        if parse_atom_name in find_atom_names:
            find_atom_name_ind = list(find_atom_names).index(parse_atom_name)
            parsed_atom_inds[find_atom_name_ind] = atom_i
    return parsed_atom_inds


# ---------------------------------------------------------------------------
#  Minimization
# ---------------------------------------------------------------------------

def minimize_from_sys(sys, top, pos, temp=300.0, dt=2.0):
    """
    Build a Simulation, minimize, and return the result.

    Parameters
    ----------
    sys : openmm.System
    top : openmm.app.Topology
    pos : positions
    temp : float
        Temperature in Kelvin.  Default 300.
    dt : float
        Timestep in femtoseconds.  Default 2.

    Returns
    -------
    simulation : openmm.app.Simulation
    min_PE : float
        Minimized potential energy (kJ/mol).
    """
    from openmm import LangevinMiddleIntegrator
    integrator = LangevinMiddleIntegrator(temp * kelvin, 1 / picosecond, dt * femtosecond)
    simulation = Simulation(top, sys, integrator)
    simulation.context.setPositions(pos)
    _describe_state_from_sim(simulation, "Original state")
    simulation.minimizeEnergy()
    min_PE = _describe_state_from_sim(simulation, "Minimized state")
    return simulation, min_PE


# ---------------------------------------------------------------------------
#  PDB text manipulation
# ---------------------------------------------------------------------------

def change_resname(pdb_file_in, pdb_file_out, resname_in, resname_out):
    """
    Interactively rename a residue in a PDB file (prompts for confirmation).
    """
    with open(pdb_file_in, 'r') as f:
        lines = f.readlines()
    print('Effected Lines:')
    eff_lines = [line for line in lines if resname_in in line]
    for line in eff_lines:
        print(line, "-->", line.replace(resname_in, resname_out))
    user_input = input("Confirm to make these changes [y/n] :")
    if user_input == 'y':
        lines = [line.replace(resname_in, resname_out) for line in lines]
        with open(pdb_file_out, 'w') as f:
            f.writelines(lines)
        return pdb_file_out
    else:
        print('Aborting....')
        return None


# ---------------------------------------------------------------------------
#  openmmtools helpers
# ---------------------------------------------------------------------------

def convert_to_TrackedQuantity(arr, u):
    """Wrap a numpy array as an openmmtools TrackedQuantity with unit *u*."""
    return TrackedQuantity(
        unit.Quantity(value=np.ma.masked_array(data=arr, mask=False, fill_value=1e+20), unit=u)
    )


def build_sampler_states(n_replicates, pos, box_vec, velos=None):
    """
    Build a list of openmmtools SamplerState objects.

    Parameters
    ----------
    n_replicates : int
    pos : array-like
        Positions array, shape ``(n_replicates, n_atoms, 3)``.
    box_vec : array-like
        Box vectors, shape ``(n_replicates, 3, 3)``.
    velos : array-like, optional
        Velocities with same leading shape as *pos*.

    Returns
    -------
    list[SamplerState]
    """
    if velos is not None:
        return [SamplerState(positions=pos[i], box_vectors=box_vec[i], velocities=velos[i])
                for i in range(n_replicates)]
    else:
        return [SamplerState(positions=pos[i], box_vectors=box_vec[i])
                for i in range(n_replicates)]
