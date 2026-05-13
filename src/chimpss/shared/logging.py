from datetime import datetime


def timestamp(x):
    """Return a string of the form ``YYYY-MM-DD HH:MM:SS.ffffff://<x>``."""
    return f"{datetime.now()}://{x}"


def printf(x):
    """Print *x* with a ``MM/DD/YYYY HH:MM:SS//`` timestamp prefix, flushed immediately."""
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True)


def unique_residues(t):
    """Return a dict mapping chain index to a list of unique residue names.

    Parameters
    ----------
    t : mdtraj.Trajectory

    Returns
    -------
    dict
        ``{chain_index: [resname, ...]}`` preserving first-seen order.
    """
    result = {}
    for chain in t.top.chains:
        result[chain.index] = []
        for res in chain.residues:
            if res.name not in result[chain.index]:
                result[chain.index].append(res.name)
    return result


def report_chain_information(traj):
    """Return a multi-line string summarising chain composition for *traj*.

    Each chain line reports its ID, index, atom count, residue count, and the
    sorted list of unique residue names found in that chain.

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
        unique_res = []
        for res in chain.residues:
            if res.name not in unique_res:
                unique_res.append(res.name)
        report.append(timestamp(f"\tUnique Residues in Chain {' '.join(sorted(unique_res))}"))
    return '\n'.join(report)
