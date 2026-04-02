"""
Consolidated I/O utilities for ChiMPSS.

Merges functions from:
  utility/FileManipulations.py  (write_FASTA, cif2pdb, remove_dummy_atoms,
                                 isolate_chains, slice_select)
  utils/utils.py                (write_FASTA duplicate)
  interactive/interactive_utils.py (read_json, write_json)
  utility/Reporting.py          (unique_residues, report_chain_information)
"""

import json
import os

import mdtraj as md

from .filesystem import ensure_exists
from .logging import timestamp


# ---- Trajectory helpers ----

def slice_select(traj, selection):
    """Atom-slice a trajectory using an MDTraj selection string."""
    return traj.atom_slice(traj.top.select(selection))


# ---- FASTA ----

def write_FASTA(sequence, name, fasta_path):
    """
    Write a FASTA file for MODELLER.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (one-letter codes).
    name : str
        Sequence identifier.
    fasta_path : str
        Output file path.

    Returns
    -------
    fasta_path : str
    """
    FASTA = (f">P1;{name}\n"
             f"                 sequence; {name}:::::::::\n"
             f"                 {sequence}*")

    with open(fasta_path, 'w') as f:
        f.write(FASTA)

    return fasta_path


# ---- PDB / CIF manipulation ----

def cif2pdb(cif_fn):
    """Convert a CIF file to PDB using Open Babel."""
    pdb_fn = cif_fn.replace('.cif', '.obabel.pdb')
    os.system(f'obabel -icif {cif_fn} -opdb -O{pdb_fn}')
    return pdb_fn


def remove_dummy_atoms(pdb_file):
    """Strip HETATM lines containing DUM atoms and write a cleaned PDB."""
    new_fn = pdb_file.replace('.pdb', '_no_dummy.pdb')
    with open(pdb_file, 'r') as f:
        lines = f.read().split('\n')
    new_lines = [line for line in lines if False in ['HETATM' in line, 'DUM' in line]]
    with open(new_fn, 'w') as f:
        f.write('\n'.join(new_lines))
    return new_fn


def isolate_chains(traj, work_dir, resname_limitation=None, from_file=False, verbose=False):
    """
    Write each chain of a trajectory to a separate PDB file.

    Parameters
    ----------
    traj : mdtraj.Trajectory or str
        Trajectory object, or file path if *from_file* is True.
    work_dir : str
        Directory to write chain PDB files.
    resname_limitation : dict, optional
        Keys are chain indices, values are a list of residue names to keep,
        ``'protein'``, ``None`` (keep all), or ``'dont'`` (skip chain).
    from_file : bool
        If True, *traj* is interpreted as a file path.
    verbose : bool
        Print selection strings.

    Returns
    -------
    bool
        True on success.
    """
    if from_file:
        traj = md.load(traj)
    ensure_exists(work_dir)
    for chain in traj.top.chains:
        if resname_limitation:
            limitation = resname_limitation[chain.index]
            if type(limitation) == list:
                selection_string = (
                    f"chainid {chain.index} and "
                    f"({' or '.join([f'resname {name}' for name in limitation])})"
                )
            elif limitation == 'protein':
                selection_string = f"chainid {chain.index} and protein"
            elif limitation is None:
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


# ---- JSON ----

def read_json(json_fn):
    """Read a JSON file and return its contents as a dict."""
    with open(json_fn, 'r') as f:
        return json.load(f)


def write_json(data, json_fn):
    """Write a dict to a JSON file with indentation."""
    with open(json_fn, 'w') as f:
        json.dump(data, f, indent=6)


# ---- Topology reporting ----

def unique_residues(traj):
    """
    Return a dict mapping chain index to its unique residue names.

    Parameters
    ----------
    traj : mdtraj.Trajectory

    Returns
    -------
    dict[int, list[str]]
    """
    result = {}
    for chain in traj.top.chains:
        result[chain.index] = []
        for res in chain.residues:
            if res.name not in result[chain.index]:
                result[chain.index].append(res.name)
    return result


def report_chain_information(traj):
    """
    Build a timestamped multi-line report of chain composition.

    Parameters
    ----------
    traj : mdtraj.Trajectory

    Returns
    -------
    str
    """
    report = [timestamp("Begin Reporting Chain Information")]
    for chain in traj.top.chains:
        report.append(timestamp(
            f"\tchainID={chain.chain_id} chainIndex={chain.index} "
            f"N_atoms={chain.n_atoms} N_residues={chain.n_residues}"
        ))
        seen = []
        for res in chain.residues:
            if res.name not in seen:
                seen.append(res.name)
        report.append(timestamp(f"\tUnique Residues in Chain {' '.join(sorted(seen))}"))
    return '\n'.join(report)
