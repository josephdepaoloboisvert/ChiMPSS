from datetime import datetime


timestamp = lambda x: f"{datetime.now()}://{x}"
printf = lambda x: print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True)


def unique_residues(t):
    result = {}
    for chain in t.top.chains:
        result[chain.index] = []
        for res in chain.residues:
            if res.name not in result[chain.index]:
                result[chain.index].append(res.name)
    return result


def report_chain_information(traj):
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
