import os, sys, glob
import mdtraj as md
from .General import *


slice_select = lambda traj, selection: traj.atom_slice(traj.top.select(selection))


def write_FASTA(sequence, name, fasta_path):

    # Write FASTA
    FASTA = f""">P1;{name}
                 sequence; {name}:::::::::
                 {sequence}*"""
    
    with open(fasta_path, 'w') as f:
        f.write(FASTA)
        f.close()

    return fasta_path


def cif2pdb(cif_fn):
    pdb_fn = cif_fn.replace('.cif','.obabel.pdb')
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
    traj - trajectory to write out individual chains of
    work_dir - location to write the chains
    resname_limitation - if certain residues are to be isolated, provide a dictionary of keys as chain indices, and values as lists of resnames to keep
    from_file - if True, traj is the filepath to load a trajectory from, not a trajectory object itself.
    """
    if from_file:
        traj = md.load(traj)
    ensure_exists(work_dir)
    for chain in traj.top.chains:
        if resname_limitation:
            limitation = resname_limitation[chain.index]
            if type(limitation) == list:
                selection_string = f"chainid {chain.index} and ({' or '.join([f"resname {name}" for name in limitation])})"
            elif limitation == 'protein':
                selection_string = f"chainid {chain.index} and protein"
            elif limitation == None:
                selection_string = f"chainid {chain.index}"
            elif limitation == 'dont':
                continue
            else:
                raise Exception('Limitation must be either a list of resnames or protein')
        else:
            selection_string = f"chainid {chain.index}"
        if verbose:
            print(chain.index, selection_string)
        chain_traj = traj.atom_slice(traj.top.select(selection_string))
        chain_pdb_fn = os.path.join(work_dir, f"chain{chain.index}ID{chain.chain_id}.pdb")
        chain_traj.save_pdb(chain_pdb_fn)
    return True