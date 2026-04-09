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

# MD topology resnames that differ from canonical 3-letter codes but represent
# the same amino acid (protonation variants, CHARMM-specific names, etc.)
_MD_RESNAME_NORM = {
    # histidine protonation variants
    'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS',
    # cysteine
    'CYX': 'CYS', 'CYM': 'CYS',
    # protonated ASP / GLU
    'ASH': 'ASP', 'GLH': 'GLU',
    # neutral LYS
    'LYN': 'LYS',
}
# residues to skip entirely (capping groups, water, etc.)
_NON_PROTEIN = {'ACE', 'NME', 'NH2', 'NHE', 'CBT', 'FOR',
                'WAT', 'HOH', 'TIP', 'SOL', 'NA', 'CL', 'MG', 'ZN'}


def _normalize_resname(resname):
    """Return canonical 3-letter code for common MD resname variants."""
    return _MD_RESNAME_NORM.get(resname.upper(), resname.upper())


# ── sequence utilities for resid-offset auto-detection ───────────────────────

def build_gpcrdb_sequence(naming_info):
    """
    Build a sequence dict from GPCRdb naming data.

    Args:
        naming_info: list of residue dicts from GPCRdb residues/extended endpoint
                     (stored in Structure_Analyzer.meta['naming'])

    Returns:
        dict {sequence_number (int): aa_1letter (str)}
        Residues whose amino_acid field is missing or unknown are omitted.
    """
    seq = {}
    for aa in naming_info:
        seqnum = aa.get('sequence_number')
        letter = aa.get('amino_acid', '')
        if seqnum is not None and letter and letter != 'X':
            seq[seqnum] = letter.upper()
    return seq


def build_traj_sequence(u, chain=None):
    """
    Extract the protein backbone sequence from an MDAnalysis Universe.

    Capping groups and non-protein residues are filtered out.  MD-specific
    protonation-state resnames (HID, CYX, ASH, …) are normalised to their
    canonical equivalents before 3→1 conversion.

    Args:
        u:     MDAnalysis Universe (first frame is used)
        chain: chain ID string, or None for no chain filter

    Returns:
        list of (resid, aa_1letter) tuples in residue order,
        skipping any residue that maps to 'X' after normalisation.
    """
    chain_sel = f"chainid {chain} and " if chain else ""
    residues = u.select_atoms(f"{chain_sel}backbone").residues
    result = []
    for res in residues:
        if res.resname.upper() in _NON_PROTEIN:
            continue
        canonical = _normalize_resname(res.resname)
        letter = _THREE2ONE.get(canonical, 'X')
        if letter != 'X':
            result.append((res.resid, letter))
    return result


def sliding_window_align(traj_seq, gpcrdb_seq):
    """
    Find the integer offset such that trajectory resids best match GPCRdb
    sequence numbers, using a sliding-window identity search.

    The trajectory sequence is treated as a contiguous window of the GPCRdb
    (UniProt) sequence.  For each possible alignment position the fraction of
    identical residues is computed; the best-scoring position determines the
    offset.

    Args:
        traj_seq:    list of (resid, aa_1letter) from build_traj_sequence
        gpcrdb_seq:  dict {seqnum: aa_1letter} from build_gpcrdb_sequence

    Returns:
        offset     (int)   : value to add to GPCRdb seqnums to get traj resids
                             i.e.  traj_resid = GPCRdb_seqnum + offset
        confidence (float) : fraction of traj residues matched at best position
        n_matched  (int)   : number of identical residues at best position
        n_compared (int)   : number of positions compared (= len(traj_seq))
    """
    if not traj_seq or not gpcrdb_seq:
        return 0, 0.0, 0, 0

    seqnums   = sorted(gpcrdb_seq)
    gpcrdb_aa = [gpcrdb_seq[s] for s in seqnums]
    traj_aa   = [aa for _, aa in traj_seq]
    traj_resids = [r for r, _ in traj_seq]

    n_traj = len(traj_aa)
    n_gpcrdb = len(gpcrdb_aa)

    best_matches = -1
    best_start   = 0

    # Slide the trajectory window across the GPCRdb sequence.
    # Allow the window to extend slightly beyond either end (by at most
    # n_traj//2 positions) so that N- or C-terminally truncated systems
    # can still align.
    lo = max(0, -n_traj // 2)
    hi = n_gpcrdb  # exclusive

    for start in range(lo, hi):
        matches = 0
        for ti, ga in enumerate(traj_aa):
            gi = start + ti
            if 0 <= gi < n_gpcrdb and ga == gpcrdb_aa[gi]:
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_start   = start

    # offset: traj_resid[0] should correspond to gpcrdb_seqnums[best_start]
    if 0 <= best_start < n_gpcrdb:
        gpcrdb_start_seqnum = seqnums[best_start]
    else:
        gpcrdb_start_seqnum = seqnums[0] + best_start  # extrapolated

    offset     = traj_resids[0] - gpcrdb_start_seqnum
    confidence = best_matches / n_traj if n_traj > 0 else 0.0

    return offset, confidence, best_matches, n_traj


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


def is_standard_bw(label):
    """
    Return True if label is a well-formed single-helix BW label, e.g. '3.50'.

    Rejects anything with a multi-digit helix number (e.g. '23.50', '45.12')
    which GPCRdb sometimes returns for non-TM regions or numbering artefacts.
    """
    parts = label.split('.')
    return (len(parts) == 2
            and len(parts[0]) == 1
            and parts[0].isdigit()
            and parts[1].isdigit())


def conservation_filter(bw_assignments, n_structures, threshold=0.90):
    """
    Return BW labels present in >= threshold fraction of structures,
    sorted by helix and position.

    Only single-digit helix labels (1.xx – 9.xx) are considered; labels with
    multi-digit helix numbers (e.g. '23.50') are silently dropped.

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
            if label != '-1' and is_standard_bw(label):
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
