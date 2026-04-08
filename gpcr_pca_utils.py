"""
Shared utilities for generate_pca_gpcrs.py and project_pca_gpcrs.py.

Contains:
  - Structure_Analyzer: fetch + cache GPCRdb/RCSB data per PDB code
  - naming_from_convention: BW / Wootten / etc. label extraction
  - fetch_all_parallel: parallel HTTP fetch with per-structure disk cache
  - build_bw_assignments: map structure resids → BW labels
  - conservation_filter: keep labels above a prevalence threshold
  - build_resids_copopulated: per-structure resid lists for conserved labels
  - select_conserved_atoms: apply user selection within conserved resids
"""

import os
import json
import requests
import numpy as np
import MDAnalysis as mda
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── amino acid conversion ─────────────────────────────────────────────────────

_THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
three_to_one = lambda x: _THREE2ONE.get(x, 'X')


def make_a_request(url, timeout=30):
    return requests.get(url, timeout=timeout).json()


# ── BW (and related) label extraction ────────────────────────────────────────

def naming_from_convention(seq_info, scheme='BW'):
    """
    Extract residue labels from a GPCRdb residues/extended API response.

    Args:
        seq_info: list of residue dicts from GPCRdb
        scheme:   one of 'BW', 'Wootten', 'Pin', 'Wang', 'Fungal'

    Returns:
        dict {sequence_number (int): label (str)}
        Residues with no label for the requested scheme get '-1'.
    """
    assert scheme in ('BW', 'Wootten', 'Pin', 'Wang', 'Fungal'), \
        f"Unknown scheme '{scheme}'"
    labels = {}
    for aa in seq_info:
        alts = aa.get('alternative_generic_numbers') or []
        matches = [e['label'] for e in alts if e.get('scheme') == scheme]
        labels[aa['sequence_number']] = matches[0] if matches else '-1'
    return labels


# ── Structure_Analyzer ────────────────────────────────────────────────────────

class Structure_Analyzer:
    """
    Wraps GPCRdb and RCSB requests for one PDB entry.

    Disk cache: each instance writes/reads <stor_dir>/cache/<PDB>.json so that
    a second run avoids all HTTP calls for metadata.  The PDB file itself is
    also skipped if it already exists on disk.
    """

    def __init__(self, pdb_code: str, stor_dir: str):
        self.pdb_code = pdb_code.upper()
        self.stor_dir = stor_dir
        self.meta = {}
        self._cache_dir = os.path.join(stor_dir, 'cache')
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── cache I/O ─────────────────────────────────────────────────────────────

    def _cache_path(self):
        return os.path.join(self._cache_dir, f'{self.pdb_code}.json')

    def load_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.meta = json.load(f)
            return True
        return False

    def save_cache(self):
        with open(self._cache_path(), 'w') as f:
            json.dump(self.meta, f)

    # ── individual fetchers (idempotent: skip if already in self.meta) ────────

    def request_structure_info(self):
        if 'structure' not in self.meta:
            self.meta['structure'] = make_a_request(
                f"https://gpcrdb.org/services/structure/{self.pdb_code}")

    def request_rcsb_pdb(self, and_load=False):
        self.request_structure_info()
        pdb_path = os.path.join(self.stor_dir, f'{self.pdb_code}.pdb')
        if not os.path.exists(pdb_path):
            resp = requests.get(
                f'https://files.rcsb.org/download/{self.pdb_code}.pdb',
                timeout=60)
            resp.raise_for_status()
            with open(pdb_path, 'wb') as f:
                f.write(resp.content)
        if and_load:
            self.load_pdb()

    def request_gene_info(self):
        self.request_structure_info()
        if 'gene' not in self.meta:
            self.meta['gene'] = make_a_request(
                f"https://gpcrdb.org/services/protein/"
                f"{self.meta['structure']['protein']}")

    def request_naming_info(self):
        self.request_structure_info()
        if 'naming' not in self.meta:
            self.meta['naming'] = make_a_request(
                f"https://gpcrdb.org/services/residues/extended/"
                f"{self.meta['structure']['protein']}/")

    def request_gene_structure_alignment(self):
        self.request_structure_info()
        if 'alignment' not in self.meta:
            self.meta['alignment'] = make_a_request(
                f"https://gpcrdb.org/services/alignment/protein/"
                f"{self.meta['structure']['pdb_code']},"
                f"{self.meta['structure']['protein']}")

    # ── bulk fetch (used by the parallel helper below) ────────────────────────

    def fetch_all(self):
        """
        Fetch all remote data, using the disk cache when available.

        Returns:
            (True, None) on success
            (False, error_string) on failure
        """
        try:
            cached = self.load_cache()
            if not cached:
                self.request_structure_info()
                self.request_gene_info()
                self.request_naming_info()
                self.request_gene_structure_alignment()
                self.save_cache()
            # PDB file is not stored in the JSON cache — always check
            self.request_rcsb_pdb(and_load=False)
            return True, None
        except Exception as exc:
            return False, str(exc)

    # ── local operations ──────────────────────────────────────────────────────

    def load_pdb(self):
        """
        Load the PDB file into MDAnalysis.  Sets:
          self.u_pdb    — AtomGroup for the preferred chain
          self.backbone — backbone atoms within that chain
        """
        self.request_structure_info()
        info = self.meta['structure']
        pdb_path = os.path.join(self.stor_dir, f'{info["pdb_code"]}.pdb')
        self.u_pdb = mda.Universe(pdb_path).select_atoms(
            f'chainid {info["preferred_chain"]}')
        self.backbone = self.u_pdb.select_atoms('backbone')

    def getSchemeNames(self, scheme='BW'):
        self.request_naming_info()
        return naming_from_convention(self.meta['naming'], scheme=scheme)

    def getPrimarySequence(self):
        if not hasattr(self, 'backbone'):
            self.load_pdb()
        return ''.join(three_to_one(r.resname) for r in self.backbone.residues)


# ── parallel fetch ────────────────────────────────────────────────────────────

def fetch_all_parallel(tests, max_workers=16):
    """
    Call fetch_all() on each Structure_Analyzer in parallel threads.

    Returns:
        dict {pdb_code: (ok, err_or_None)}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(t.fetch_all): t.pdb_code for t in tests}
        for fut in as_completed(futures):
            code = futures[fut]
            results[code] = fut.result()
            ok, err = results[code]
            if not ok:
                print(f"  FAILED {code}: {err}")
    return results


# ── BW assignment helpers ─────────────────────────────────────────────────────

def build_bw_assignments(tests):
    """
    For each structure, map PDB resid → BW label using GPCRdb sequence numbers.

    GPCRdb uses UniProt sequence numbers, which often match PDB resids directly.
    Residues whose resid is not in the BW map are silently skipped (common for
    non-standard residue numbering in crystal structures).

    Returns:
        dict {pdb_code: {resid (int): bw_label (str)}}
    """
    bw_assignments = {}
    for test in tests:
        try:
            bw_map = test.getSchemeNames()
            if not hasattr(test, 'u_pdb'):
                test.load_pdb()
            ca_resids = test.backbone.select_atoms('name CA').residues.resids
            bw_assignments[test.pdb_code] = {
                resid: bw_map[resid]
                for resid in ca_resids
                if resid in bw_map
            }
        except Exception as exc:
            print(f"  BW error {test.pdb_code}: {exc}")
    return bw_assignments


def _bw_sort_key(label):
    """Sort 'H.PP' labels numerically by helix then position."""
    try:
        h, p = label.split('.')
        return (int(h), int(p))
    except ValueError:
        return (999, 999)


def conservation_filter(bw_assignments, n_structures, threshold=0.90):
    """
    Return BW labels present in >= threshold fraction of structures,
    sorted by helix and position.

    Args:
        bw_assignments: output of build_bw_assignments
        n_structures:   denominator for conservation calculation
        threshold:      fractional cutoff (0–1)

    Returns:
        list of BW label strings
    """
    counts = {}
    for assignments in bw_assignments.values():
        for label in assignments.values():
            if label != '-1':
                counts[label] = counts.get(label, 0) + 1
    conserved = [lbl for lbl, n in counts.items()
                 if n / n_structures >= threshold]
    return sorted(conserved, key=_bw_sort_key)


def build_resids_copopulated(tests, bw_assignments, conserved_bw):
    """
    For each structure, find the single PDB resid corresponding to each
    conserved BW label.  Structures where any label maps to 0 or 2+ resids
    are excluded (ambiguous or missing).

    Returns:
        dict {pdb_code: [[resid], [resid], ...]}
        where the inner lists are in the same order as conserved_bw.
    """
    result = {}
    for test in tests:
        code = test.pdb_code
        if code not in bw_assignments:
            continue
        rev = {lbl: rid for rid, lbl in bw_assignments[code].items()
               if lbl != '-1'}
        resids = [[rev[lbl]] if lbl in rev else [] for lbl in conserved_bw]
        if all(len(r) == 1 for r in resids):
            result[code] = resids
    return result


# ── atom selection ────────────────────────────────────────────────────────────

def select_conserved_atoms(u_chain, resids, selection):
    """
    Select atoms from a chain-filtered AtomGroup (u_chain) that match both
    the conserved residue list and the user-supplied selection string.

    Args:
        u_chain:   MDAnalysis AtomGroup already filtered to the correct chain
        resids:    [[resid], ...] in conserved_bw order
        selection: MDAnalysis selection string, e.g. 'name CA' or 'backbone'

    Returns:
        MDAnalysis AtomGroup
    """
    resid_str = " or resid ".join(str(r[0]) for r in resids)
    return u_chain.select_atoms(f"(resid {resid_str}) and ({selection})")
