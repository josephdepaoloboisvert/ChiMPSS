from datetime import datetime

timestamp = lambda x: f"{datetime.now()}://{x}"
printf = lambda x : print(f"{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}//{x}", flush=True)

def unique_residues(t):
    unique_residues = {}
    for chain in t.top.chains:
        unique_residues[chain.index] = []
        for res in chain.residues:
            if res.name not in unique_residues[chain.index]:
                unique_residues[chain.index].append(res.name)
    return unique_residues

def report_chain_information(traj):
    report = [timestamp("Begin Reporting Chain Information")]
    for chain in traj.top.chains:
        report.append(timestamp(f"\tchainID={chain.chain_id} chainIndex={chain.index} N_atoms={chain.n_atoms} N_residues={chain.n_residues}"))
        unique_residues = []
        for res in chain.residues:
            if res.name not in unique_residues:
                unique_residues.append(res.name)
        report.append(timestamp(f"\tUnique Residues in Chain {' '.join(sorted(unique_residues))}"))
    return '\n'.join(report)