from datetime import datetime

timestamp = lambda x: f"{datetime.now()}://{x}"

def report_chain_information(traj):
    report = [timestamp("Begin Reporting Chain Information")]
    for chain in traj.top.chains:
        report.append(timestamp(f"\tchainID={chain.chain_id} chainIndex={chain.index} N_atoms={chain.n_atoms} N_residues={chain.n_residues}"))
        if chain.n_residues < 5:
            report.append(timestamp(f"\t\tSmall Chain {chain.chain_id} has residues {[res.name for res in chain.residues]}"))
    return '\n'.join(report)