import os

import mdtraj as md
import numpy as np


def ensure_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return True


def slice_select(traj, selection):
    return traj.atom_slice(traj.top.select(selection))


def write_FASTA(sequence, name, fasta_path):
    FASTA = f""">P1;{name}
                 sequence; {name}:::::::::
                 {sequence}*"""
    with open(fasta_path, 'w') as f:
        f.write(FASTA)
    return fasta_path


def cif2pdb(cif_fn):
    pdb_fn = cif_fn.replace('.cif', '.obabel.pdb')
    _ = os.system(f'obabel -icif {cif_fn} -opdb -O{pdb_fn}')
    return pdb_fn


def remove_dummy_atoms(pdb_file):
    new_fn = pdb_file.replace('.pdb', '_no_dummy.pdb')
    with open(pdb_file, 'r') as f:
        lines = f.read().split('\n')
    new_lines = [line for line in lines if False in ['HETATM' in line, 'DUM' in line]]
    with open(new_fn, 'w') as f:
        f.write('\n'.join(new_lines))
    return new_fn


def isolate_chains(traj, work_dir, resname_limitation=None, from_file=False, verbose=False):
    """
    Write individual chain PDB files from a trajectory.

    resname_limitation: dict of chain index → list of resnames, 'protein', None, or 'dont'
    from_file: if True, traj is a filepath string to load
    """
    if from_file:
        traj = md.load(traj)
    ensure_exists(work_dir)
    for chain in traj.top.chains:
        if resname_limitation:
            limitation = resname_limitation[chain.index]
            if isinstance(limitation, list):
                parts = ' or '.join([f'resname {name}' for name in limitation])
                selection_string = f"chainid {chain.index} and ({parts})"
            elif limitation == 'protein':
                selection_string = f"chainid {chain.index} and protein"
            elif limitation is None:
                selection_string = f"chainid {chain.index}"
            elif limitation == 'dont':
                continue
            else:
                raise Exception('Limitation must be a list of resnames, "protein", None, or "dont"')
        else:
            selection_string = f"chainid {chain.index}"
        if verbose:
            print(chain.index, selection_string)
        chain_traj = traj.atom_slice(traj.top.select(selection_string))
        chain_pdb_fn = os.path.join(work_dir, f"chain{chain.index}ID{chain.chain_id}.pdb")
        chain_traj.save_pdb(chain_pdb_fn)
    return True


def change_resname(pdb_file_in, pdb_file_out, resname_in, resname_out):
    """Interactively replace all occurrences of resname_in with resname_out in a PDB file."""
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


def describe_system(sys):
    box_vecs = sys.getDefaultPeriodicBoxVectors()
    print('Box Vectors')
    [print(box_vec) for box_vec in box_vecs]
    forces = sys.getForces()
    print('Forces')
    [print(force) for force in forces]
    num_particles = sys.getNumParticles()
    print(num_particles, 'Particles')


def describe_state(state, name: str = "State"):
    max_force = max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces())
    print(f"{name} has energy {round(state.getPotentialEnergy()._value, 2)} kJ/mol ",
          f"with maximum force {round(max_force, 2)} kJ/(mol nm)")
