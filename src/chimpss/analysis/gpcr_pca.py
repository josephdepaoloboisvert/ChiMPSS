"""
GPCR PCA library: GPCRdb fetching, BW-numbering, conservation filtering,
PCA training helpers, and trajectory projection utilities.

Consolidated from:
  - gpcr_pca_utils.py       (Structure_Analyzer, BW helpers, fetch utilities)
  - generate_pca_gpcrs.py   (BW position spec parsing)
  - project_pca_gpcrs.py    (projection and imputation helpers)

CLI entry points:
  chimpss-generate-pca  →  src/chimpss/cli/generate_pca.py
  chimpss-project-pca   →  src/chimpss/cli/project_pca.py
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import MDAnalysis as mda
import numpy as np
import requests
from MDAnalysis.analysis import align

# ── amino acid conversion ─────────────────────────────────────────────────────

_THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
def three_to_one(x):
    return _THREE2ONE.get(x, 'X')

_MD_RESNAME_NORM = {
    'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS',
    'CYX': 'CYS', 'CYM': 'CYS',
    'ASH': 'ASP', 'GLH': 'GLU',
    'LYN': 'LYS',
}
_NON_PROTEIN = {'ACE', 'NME', 'NH2', 'NHE', 'CBT', 'FOR',
                'WAT', 'HOH', 'TIP', 'SOL', 'NA', 'CL', 'MG', 'ZN'}


def _normalize_resname(resname):
    return _MD_RESNAME_NORM.get(resname.upper(), resname.upper())


# ── sequence utilities for resid-offset auto-detection ───────────────────────

def build_gpcrdb_sequence(naming_info):
    """
    Build a sequence dict from GPCRdb naming data.

    Returns dict {sequence_number (int): aa_1letter (str)}.
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

    Returns list of (resid, aa_1letter) tuples.
    """
    chain_sel = f"chainid {chain} and " if chain else ""
    residues  = u.select_atoms(f"{chain_sel}backbone").residues
    result = []
    for res in residues:
        if res.resname.upper() in _NON_PROTEIN:
            continue
        canonical = _normalize_resname(res.resname)
        letter    = _THREE2ONE.get(canonical, 'X')
        if letter != 'X':
            result.append((res.resid, letter))
    return result


def sliding_window_align(traj_seq, gpcrdb_seq):
    """
    Find the integer offset mapping trajectory resids to GPCRdb sequence numbers.

    Returns (offset, confidence, n_matched, n_compared).
    """
    if not traj_seq or not gpcrdb_seq:
        return 0, 0.0, 0, 0

    seqnums     = sorted(gpcrdb_seq)
    gpcrdb_aa   = [gpcrdb_seq[s] for s in seqnums]
    traj_aa     = [aa for _, aa in traj_seq]
    traj_resids = [r for r, _ in traj_seq]

    n_traj   = len(traj_aa)
    n_gpcrdb = len(gpcrdb_aa)

    best_matches = -1
    best_start   = 0

    lo = max(0, -n_traj // 2)
    hi = n_gpcrdb

    for start in range(lo, hi):
        matches = 0
        for ti, ga in enumerate(traj_aa):
            gi = start + ti
            if 0 <= gi < n_gpcrdb and ga == gpcrdb_aa[gi]:
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_start   = start

    if 0 <= best_start < n_gpcrdb:
        gpcrdb_start_seqnum = seqnums[best_start]
    else:
        gpcrdb_start_seqnum = seqnums[0] + best_start

    offset     = traj_resids[0] - gpcrdb_start_seqnum
    confidence = best_matches / n_traj if n_traj > 0 else 0.0
    return offset, confidence, best_matches, n_traj


def make_a_request(url, timeout=30):
    return requests.get(url, timeout=timeout).json()


# ── BW label extraction ───────────────────────────────────────────────────────

def naming_from_convention(seq_info, scheme='BW'):
    """
    Extract residue labels from a GPCRdb residues/extended API response.

    Returns dict {sequence_number (int): label (str)}.
    """
    assert scheme in ('BW', 'Wootten', 'Pin', 'Wang', 'Fungal'), \
        f"Unknown scheme '{scheme}'"
    labels = {}
    for aa in seq_info:
        alts    = aa.get('alternative_generic_numbers') or []
        matches = [e['label'] for e in alts if e.get('scheme') == scheme]
        labels[aa['sequence_number']] = matches[0] if matches else '-1'
    return labels


# ── Structure_Analyzer ────────────────────────────────────────────────────────

class Structure_Analyzer:
    """
    Wraps GPCRdb and RCSB requests for one PDB entry with a disk cache.
    """

    def __init__(self, pdb_code: str, stor_dir: str):
        self.pdb_code  = pdb_code.upper()
        self.stor_dir  = stor_dir
        self.meta      = {}
        self._cache_dir = os.path.join(stor_dir, 'cache')
        os.makedirs(self._cache_dir, exist_ok=True)

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

    def request_structure_info(self):
        if 'structure' not in self.meta:
            self.meta['structure'] = make_a_request(
                f"https://gpcrdb.org/services/structure/{self.pdb_code}")

    def request_rcsb_pdb(self, and_load=False):
        """Download the coordinates, falling back to mmCIF when needed.

        RCSB no longer publishes a legacy PDB file for every entry: structures
        that exceed the format's limits are mmCIF-only, and a plain ``.pdb``
        request returns 404.  That is 75 of the 1513 human GPCR structures
        GPCRdb currently lists, and the share grows as large cryo-EM complexes
        accumulate -- so treating a 404 as fatal silently shrinks the training
        set over time.

        Only the receptor's preferred chain is converted, which also sidesteps
        the reason the entry has no PDB file in the first place.
        """
        self.request_structure_info()
        pdb_path = os.path.join(self.stor_dir, f'{self.pdb_code}.pdb')
        if os.path.exists(pdb_path):
            if and_load:
                self.load_pdb()
            return

        resp = requests.get(
            f'https://files.rcsb.org/download/{self.pdb_code}.pdb', timeout=60)
        if resp.status_code == 404:
            cif_path = os.path.join(self.stor_dir, f'{self.pdb_code}.cif')
            cif = requests.get(
                f'https://files.rcsb.org/download/{self.pdb_code}.cif', timeout=120)
            cif.raise_for_status()
            with open(cif_path, 'wb') as f:
                f.write(cif.content)
            convert_cif_to_pdb(
                cif_path, pdb_path,
                chain=self.meta['structure'].get('preferred_chain'))
        else:
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

    def fetch_all(self):
        """Fetch all remote data, using the disk cache when available."""
        try:
            cached = self.load_cache()
            if not cached:
                self.request_structure_info()
                self.request_gene_info()
                self.request_naming_info()
                self.request_gene_structure_alignment()
                self.save_cache()
            self.request_rcsb_pdb(and_load=False)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def load_pdb(self):
        self.request_structure_info()
        info      = self.meta['structure']
        pdb_path  = os.path.join(self.stor_dir, f'{info["pdb_code"]}.pdb')
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


def convert_cif_to_pdb(cif_path, pdb_path, chain=None):
    """Write one chain of an mmCIF file out as PDB.

    Needs an mmCIF parser, which is not part of the declared environment:
    ``gemmi`` is tried first, then Biopython.  If neither is importable the
    caller gets a clear instruction rather than a stack trace, and the
    structure is simply skipped.
    """
    try:
        import gemmi
    except ImportError:
        pass
    else:
        st = gemmi.read_structure(cif_path)
        st.setup_entities()
        if chain:
            for model in st:
                for name in [ch.name for ch in model if ch.name != chain]:
                    model.remove_chain(name)
        st.write_pdb(pdb_path)
        return pdb_path

    try:
        from Bio.PDB import MMCIFParser, PDBIO, Select
    except ImportError:
        raise RuntimeError(
            f'{os.path.basename(cif_path)} is mmCIF-only and no mmCIF parser is '
            f'available. Install one with:  conda install -c conda-forge gemmi')

    structure = MMCIFParser(QUIET=True).get_structure('s', cif_path)

    class _Chain(Select):
        def accept_chain(self, ch):
            return chain is None or ch.id == chain

        def accept_model(self, model):
            return model.id == 0

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path, select=_Chain())
    return pdb_path


def validate_numbering(observed, gpcrdb_seq, identity_threshold=0.90):
    """Decide how a structure's residue numbering maps onto GPCRdb.

    Deposited PDB entries almost always carry UniProt-consistent author
    numbering, so ``resid == sequence_number`` holds directly: across the 479
    human structures in the 612-code set the median identity under that
    assumption is 99.6%.  Running a global alignment on them instead is a
    *regression* -- crystallisation chimeras carry fusion partners (BRIL, T4L)
    inside the same chain and disordered loops leave gaps, and the aligner
    degraded 205 of the 441 structures whose direct numbering was already
    correct.

    But direct numbering does fail outright for 38 of those 479, and there the
    alignment recovers 23.  So the rule is: trust the numbering, verify it
    against the sequence, and fall back to alignment only where the evidence
    says the numbering is wrong.

    ``observed`` is ``[(resid, one_letter), ...]`` for the residues actually
    present.  Returns ``(mode, mapping, stats)`` where ``mode`` is ``'direct'``
    or ``'aligned'`` and ``mapping`` is ``{gpcrdb_seqnum: resid}``.
    """
    hits = [(r, a) for r, a in observed if r in gpcrdb_seq]
    direct_identity = (sum(1 for r, a in hits if gpcrdb_seq[r] == a) / len(hits)
                       if hits else 0.0)
    stats = {'direct_identity': direct_identity, 'n_checked': len(hits)}

    if direct_identity >= identity_threshold:
        stats['identity'] = direct_identity
        return 'direct', {r: r for r, _ in hits}, stats

    mapping, astats = align_sequence_to_gpcrdb(observed, gpcrdb_seq)
    stats['aligned_identity'] = astats['identity']
    stats['offsets'] = astats['offsets']
    if astats['identity'] > direct_identity:
        stats['identity'] = astats['identity']
        return 'aligned', mapping, stats

    stats['identity'] = direct_identity
    return 'direct', {r: r for r, _ in hits}, stats


# ── parallel fetch ────────────────────────────────────────────────────────────

def fetch_all_parallel(tests, max_workers=16):
    """Call fetch_all() on each Structure_Analyzer in parallel threads."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(t.fetch_all): t.pdb_code for t in tests}
        for fut in as_completed(futures):
            code          = futures[fut]
            results[code] = fut.result()
            ok, err       = results[code]
            if not ok:
                print(f"  FAILED {code}: {err}")
    return results


# ── BW assignment helpers ─────────────────────────────────────────────────────

def build_bw_assignments(tests, validate=True, identity_threshold=0.90,
                         diagnostics=None):
    """Map structure resids to BW labels for each Structure_Analyzer.

    With ``validate=True`` the direct ``resid == sequence_number`` assumption is
    checked against the observed sequence and replaced by a gap-aware alignment
    only where it demonstrably fails -- see :func:`validate_numbering` for why
    that is the right way round.  Pass a dict as ``diagnostics`` to receive the
    per-structure mode and identity.
    """
    bw_assignments = {}
    for test in tests:
        try:
            bw_map = test.getSchemeNames()
            if not hasattr(test, 'u_pdb'):
                test.load_pdb()
            ca = test.backbone.select_atoms('name CA').residues

            if not validate:
                bw_assignments[test.pdb_code] = {
                    resid: bw_map[resid]
                    for resid in ca.resids
                    if resid in bw_map
                }
                continue

            observed = [(int(r.resid),
                         three_to_one(_normalize_resname(r.resname)))
                        for r in ca]
            observed = [(r, a) for r, a in observed if a != 'X']
            gpcrdb_seq = build_gpcrdb_sequence(test.meta.get('naming', []))

            mode, mapping, stats = validate_numbering(
                observed, gpcrdb_seq, identity_threshold=identity_threshold)

            if mode == 'direct':
                bw_assignments[test.pdb_code] = {
                    resid: bw_map[resid]
                    for resid, _ in observed
                    if resid in bw_map
                }
            else:
                bw_assignments[test.pdb_code] = {
                    mapping[seqnum]: label
                    for seqnum, label in bw_map.items()
                    if label != '-1' and seqnum in mapping
                }
            if diagnostics is not None:
                diagnostics[test.pdb_code] = dict(stats, mode=mode)
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
    """Return True for single-helix BW labels like '3.50'; reject '23.50'."""
    parts = label.split('.')
    return (len(parts) == 2
            and len(parts[0]) == 1
            and parts[0].isdigit()
            and parts[1].isdigit())


def conservation_filter(bw_assignments, n_structures, threshold=0.90):
    """Return BW labels present in >= threshold fraction of structures."""
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
    For each structure find the single PDB resid for each conserved BW label.
    Structures with any ambiguous or missing label are excluded.
    """
    result = {}
    for test in tests:
        code = test.pdb_code
        if code not in bw_assignments:
            continue
        rev    = {lbl: rid for rid, lbl in bw_assignments[code].items()
                  if lbl != '-1'}
        resids = [[rev[lbl]] if lbl in rev else [] for lbl in conserved_bw]
        if all(len(r) == 1 for r in resids):
            result[code] = resids
    return result


def select_conserved_atoms(u_chain, resids, selection):
    """Select atoms matching conserved residues and an MDAnalysis selection string."""
    resid_str = " or resid ".join(str(r[0]) for r in resids)
    return u_chain.select_atoms(f"(resid {resid_str}) and ({selection})")


# ── BW position spec parsing (from generate_pca_gpcrs.py) ────────────────────

def _parse_bw_list(spec):
    """
    Parse a BW label list from a file path or an inline string.

    Accepts comma/space-separated labels.  Non-standard helix labels are
    dropped with a warning.

    Returns list of BW label strings sorted by helix then position.
    """
    if os.path.exists(spec):
        with open(spec, 'r') as fh:
            text = fh.read()
    else:
        text = spec

    labels = [tok.strip() for tok in text.replace(',', ' ').split()
              if tok.strip()]

    bad = [lbl for lbl in labels if '.' not in lbl or len(lbl.split('.')) != 2]
    if bad:
        raise ValueError(
            f"Malformed BW labels (expected 'H.PP' format): {bad}")

    non_standard = [lbl for lbl in labels if not is_standard_bw(lbl)]
    if non_standard:
        print(f"  WARNING: dropping {len(non_standard)} non-standard labels "
              f"(multi-digit helix): {non_standard}")
    labels = [lbl for lbl in labels if is_standard_bw(lbl)]
    return sorted(labels, key=_bw_sort_key)


def load_bw_positions(spec):
    return _parse_bw_list(spec)


def load_bw_exclude(spec):
    return set(_parse_bw_list(spec))


# ── Gap-aware sequence mapping ────────────────────────────────────────────────

def _gotoh_align(a, b, match=2.0, mismatch=-1.0,
                 gap_open=-10.0, gap_extend=-0.5):
    """Global pairwise alignment with affine gaps (Gotoh, *JMB* 1982, 162, 705).

    Returns a list of ``(i, j)`` index pairs, one per aligned column in which
    both sequences have a residue.  Gap columns are simply absent from the
    result, which is exactly what a residue-number mapping needs.

    Three matrices are carried: ``M`` for a residue aligned to a residue,
    ``Ix`` for a gap in ``b``, ``Iy`` for a gap in ``a``.  The ``Iy``
    recurrence looks leftward within the same row, so the inner loop stays
    sequential rather than vectorised; at a few hundred residues per sequence
    the whole alignment is well under a second and it is cached by the caller.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    NEG = float('-inf')

    M = [[NEG] * (m + 1) for _ in range(n + 1)]
    Ix = [[NEG] * (m + 1) for _ in range(n + 1)]
    Iy = [[NEG] * (m + 1) for _ in range(n + 1)]
    # traceback: 0 = from M, 1 = from Ix, 2 = from Iy
    pM = [[0] * (m + 1) for _ in range(n + 1)]
    pX = [[0] * (m + 1) for _ in range(n + 1)]
    pY = [[0] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0.0
    for i in range(1, n + 1):
        Ix[i][0] = gap_open + (i - 1) * gap_extend
        pX[i][0] = 1
    for j in range(1, m + 1):
        Iy[0][j] = gap_open + (j - 1) * gap_extend
        pY[0][j] = 2

    for i in range(1, n + 1):
        ai = a[i - 1]
        Mi, Mp = M[i], M[i - 1]
        Xi, Xp = Ix[i], Ix[i - 1]
        Yi, Yp = Iy[i], Iy[i - 1]
        for j in range(1, m + 1):
            # gap in b: consume a[i-1]
            o, e = Mp[j] + gap_open, Xp[j] + gap_extend
            if o >= e:
                Xi[j], pX[i][j] = o, 0
            else:
                Xi[j], pX[i][j] = e, 1
            # gap in a: consume b[j-1]
            o, e = Mi[j - 1] + gap_open, Yi[j - 1] + gap_extend
            if o >= e:
                Yi[j], pY[i][j] = o, 0
            else:
                Yi[j], pY[i][j] = e, 2
            # residue against residue
            s = match if ai == b[j - 1] else mismatch
            best, ptr = Mp[j - 1], 0
            if Xp[j - 1] > best:
                best, ptr = Xp[j - 1], 1
            if Yp[j - 1] > best:
                best, ptr = Yp[j - 1], 2
            Mi[j], pM[i][j] = best + s, ptr

    i, j = n, m
    state = max(((M[n][m], 0), (Ix[n][m], 1), (Iy[n][m], 2)))[1]
    pairs = []
    while i > 0 or j > 0:
        if state == 0:
            if i == 0 or j == 0:
                break
            pairs.append((i - 1, j - 1))
            state = pM[i][j]
            i, j = i - 1, j - 1
        elif state == 1:
            state = pX[i][j]
            i -= 1
        else:
            state = pY[i][j]
            j -= 1
    pairs.reverse()
    return pairs


def align_sequence_to_gpcrdb(traj_seq, gpcrdb_seq, **kwargs):
    """Map GPCRdb sequence numbers to trajectory resids, tolerating gaps.

    :func:`sliding_window_align` assumes a *single* integer offset holds across
    the whole chain.  That is false for most crystallisation constructs: an
    internal deletion (commonly ICL3) shifts every downstream residue, and if
    the topology was renumbered contiguously the break is invisible to a
    global offset search.  On ``NTSR1_ML-301`` the best single offset reaches
    only 64.6% identity and silently mis-assigns all of TM6, TM7 and H8 by
    four residues -- a third of the conserved Ballesteros-Weinstein set.  A
    gap-aware global alignment recovers 342/342 residues at 90.4% identity and
    resolves two offsets, ``+1`` and ``+5``.

    ``traj_seq`` is a list of ``(resid, one_letter)`` as produced by
    :func:`build_traj_sequence`; ``gpcrdb_seq`` is a
    ``{sequence_number: one_letter}`` dict from :func:`build_gpcrdb_sequence`.

    Returns ``(mapping, stats)`` where ``mapping`` is
    ``{gpcrdb_seqnum: traj_resid}`` and ``stats`` carries the identity, the
    number of aligned pairs and the distinct offsets observed -- more than one
    offset is the signature of an internal indel.
    """
    if not traj_seq or not gpcrdb_seq:
        return {}, {'identity': 0.0, 'n_aligned': 0, 'offsets': {},
                    'n_traj': len(traj_seq), 'n_gpcrdb': len(gpcrdb_seq)}

    nums = sorted(gpcrdb_seq)
    ref = ''.join(gpcrdb_seq[k] for k in nums)
    rids = [r for r, _ in traj_seq]
    qry = ''.join(aa for _, aa in traj_seq)

    pairs = _gotoh_align(ref, qry, **kwargs)
    mapping = {nums[i]: rids[j] for i, j in pairs}

    n_ident = sum(1 for i, j in pairs if ref[i] == qry[j])
    offsets = {}
    for i, j in pairs:
        off = nums[i] - rids[j]
        offsets[off] = offsets.get(off, 0) + 1

    stats = {
        'identity': n_ident / len(pairs) if pairs else 0.0,
        'n_aligned': len(pairs),
        'offsets': dict(sorted(offsets.items(), key=lambda kv: -kv[1])),
        'n_traj': len(traj_seq),
        'n_gpcrdb': len(gpcrdb_seq),
    }
    return mapping, stats


def map_conserved_resids_aligned(bw_map, conserved_bw, seq_map, available):
    """Translate conserved BW labels to trajectory resids through ``seq_map``.

    The gap-aware counterpart of :func:`map_conserved_resids`.  ``seq_map`` is
    the ``{gpcrdb_seqnum: traj_resid}`` dict from
    :func:`align_sequence_to_gpcrdb`; ``available`` is the set of resids that
    actually carry backbone atoms in the topology.

    Returns ``(resids, missing_info)`` with the same shape as
    :func:`map_conserved_resids`, so either mapper can feed the same
    downstream selection and imputation code.
    """
    label_to_seqnum = {lbl: num for num, lbl in bw_map.items() if lbl != '-1'}
    available = set(available)

    resids, missing_info = [], []
    for idx, label in enumerate(conserved_bw):
        seqnum = label_to_seqnum.get(label)
        if seqnum is None:
            missing_info.append((label, idx, 'not in GPCRdb BW map for this protein'))
            resids.append([])
            continue
        rid = seq_map.get(seqnum)
        if rid is None:
            missing_info.append(
                (label, idx, f'GPCRdb seqnum {seqnum} not aligned to any trajectory residue'))
            resids.append([])
        elif rid not in available:
            missing_info.append(
                (label, idx, f'GPCRdb seqnum {seqnum} -> resid {rid} has no backbone atoms'))
            resids.append([])
        else:
            resids.append([rid])
    return resids, missing_info


# ── Projection helpers (from project_pca_gpcrs.py) ───────────────────────────

def auto_detect_resid_offset(u, analyzer, chain=None, user_offset=0):
    """
    Detect the integer offset between GPCRdb sequence numbers and MD resids
    via a sliding-window sequence identity search.

    Returns the offset to use (int).
    """
    naming_info = analyzer.meta.get('naming', [])
    if not naming_info:
        print("  Auto-offset: GPCRdb naming data unavailable — skipping.")
        return user_offset

    gpcrdb_seq = build_gpcrdb_sequence(naming_info)
    traj_seq   = build_traj_sequence(u, chain=chain)

    if not traj_seq:
        print("  Auto-offset: no backbone residues found — skipping.")
        return user_offset

    chain_sel   = f"chainid {chain} and " if chain else ""
    bb_residues = u.select_atoms(f"{chain_sel}backbone").residues
    traj_min    = int(bb_residues.resids.min())
    traj_max    = int(bb_residues.resids.max())

    print("Auto-detecting resid offset...")
    print(f"  Trajectory sequence : {len(traj_seq)} residues "
          f"(resids {traj_min}–{traj_max})")
    gmin, gmax = min(gpcrdb_seq), max(gpcrdb_seq)
    print(f"  GPCRdb sequence     : {len(gpcrdb_seq)} residues "
          f"(seqnums {gmin}–{gmax})")

    detected, confidence, n_matched, n_compared = sliding_window_align(
        traj_seq, gpcrdb_seq)

    print(f"  Best alignment      : traj resid {traj_seq[0][0]} → "
          f"GPCRdb seqnum {traj_seq[0][0] - detected}  "
          f"({confidence:.1%} identity, {n_matched}/{n_compared} matched)")

    HIGH = 0.90
    LOW  = 0.70
    user_supplied = (user_offset != 0)

    if user_supplied:
        if detected == user_offset:
            print(f"  Sanity check        : auto-detected offset {detected:+d} "
                  f"matches --resid_offset ✓")
        else:
            print(f"  WARNING: auto-detected offset ({detected:+d}, "
                  f"{confidence:.0%} confidence) differs from "
                  f"--resid_offset {user_offset:+d}.  Using your supplied value.")
        return user_offset

    if confidence >= HIGH:
        print(f"  Applying offset     : {detected:+d}  "
              f"(confidence {confidence:.1%} — high)")
        return detected
    elif confidence >= LOW:
        print(f"  Applying offset     : {detected:+d}  "
              f"(confidence {confidence:.1%} — moderate)")
        return detected
    else:
        print(f"  WARNING: confidence {confidence:.1%} too low — proceeding "
              f"with offset 0. Use --resid_offset if numbering is known.")
        return 0


def fetch_bw_map(pdb_code, stor_dir):
    """Fetch GPCRdb BW map for pdb_code using disk cache."""
    analyzer = Structure_Analyzer(pdb_code.upper(), stor_dir=stor_dir)
    print(f"Fetching GPCRdb data for {pdb_code.upper()}...")
    ok, err = analyzer.fetch_all()
    if not ok:
        sys.exit(f"ERROR: could not fetch data for {pdb_code}: {err}")
    bw_map  = analyzer.getSchemeNames()
    protein = analyzer.meta.get('structure', {}).get('protein', '?')
    print(f"  Protein slug: {protein}  |  {len(bw_map)} residues in BW map")
    return analyzer, bw_map


def fetch_naming_info(pdb_code, stor_dir):
    """Fetch only the GPCRdb residue data for a structure's parent receptor.

    :func:`fetch_bw_map` calls :meth:`Structure_Analyzer.fetch_all`, which also
    downloads the crystal structure's coordinates from RCSB.  Those coordinates
    are irrelevant when the goal is to number an MD trajectory, and insisting on
    them turns a missing file into a fatal error: RCSB no longer publishes a
    legacy PDB file for every entry -- 6ZA8, the parent of ``NTSR1_ML-301``, is
    mmCIF-only and returns 404 for ``.pdb``.

    Returns ``(naming_info, bw_map)`` where ``naming_info`` is the raw
    ``residues/extended`` payload and ``bw_map`` is ``{seqnum: BW label}``.
    """
    analyzer = Structure_Analyzer(pdb_code.upper(), stor_dir=stor_dir)
    if not analyzer.load_cache():
        analyzer.request_structure_info()
        analyzer.request_naming_info()
        analyzer.save_cache()
    else:
        analyzer.request_naming_info()

    naming_info = analyzer.meta.get('naming', [])
    if not naming_info:
        raise RuntimeError(
            f'GPCRdb returned no residue data for {pdb_code.upper()}')
    return naming_info, analyzer.getSchemeNames()


def map_conserved_resids(bw_map, conserved_bw, u, chain=None, resid_offset=0):
    """
    Translate conserved BW labels → resids present in universe u.

    Returns (resids, missing_info) where missing_info is a list of
    (bw_label, bw_index, reason) tuples.
    """
    label_to_seqnum = {lbl: seqnum for seqnum, lbl in bw_map.items()
                       if lbl != '-1'}
    chain_sel = f"chainid {chain} and " if chain else ""
    available = set(u.select_atoms(f"{chain_sel}backbone").residues.resids)

    resids      = []
    missing_info = []
    for idx, label in enumerate(conserved_bw):
        if label not in label_to_seqnum:
            missing_info.append((label, idx, "not in GPCRdb BW map for this protein"))
            resids.append([])
        else:
            traj_resid = label_to_seqnum[label] + resid_offset
            if traj_resid not in available:
                chain_note = f" in chain {chain}" if chain else ""
                reason = (f"GPCRdb seqnum {label_to_seqnum[label]} "
                          f"→ trajectory resid {traj_resid} not found{chain_note}")
                missing_info.append((label, idx, reason))
                resids.append([])
            else:
                resids.append([traj_resid])

    return resids, missing_info


def build_mobile_ag(u, resids, selection, chain=None):
    """Select atoms matching conserved resids and selection; skips empty entries."""
    present    = [r for r in resids if r]
    resid_str  = " or resid ".join(str(r[0]) for r in present)
    chain_filter = f"chainid {chain} and " if chain else ""
    return u.select_atoms(f"{chain_filter}(resid {resid_str}) and ({selection})")


def _imputation_setup(missing_info, conserved_bw, expected_n, pca):
    """
    Pre-compute indices and diagnostics for mean imputation of missing residues.

    Returns a dict with keys: present_feat_idx, missing_feat_idx,
    trimmed_ref_atom_idx, pc1_imputed_frac, pc2_imputed_frac.
    """
    n_residues    = len(conserved_bw)
    atoms_per_res = expected_n // n_residues
    feats_per_res = atoms_per_res * 3

    if expected_n % n_residues != 0:
        raise ValueError(
            f"expected_n ({expected_n}) is not divisible by n_residues "
            f"({n_residues}). Cannot determine atoms-per-residue for imputation.")

    missing_bw_indices = {bw_idx for _, bw_idx, _ in missing_info}
    present_bw_indices = [i for i in range(n_residues)
                          if i not in missing_bw_indices]

    def feat_range(bw_i):
        return np.arange(bw_i * feats_per_res, (bw_i + 1) * feats_per_res)

    def atom_range(bw_i):
        return np.arange(bw_i * atoms_per_res, (bw_i + 1) * atoms_per_res)

    present_feat_idx     = np.concatenate([feat_range(i) for i in present_bw_indices])
    missing_feat_idx     = np.concatenate([feat_range(i) for i in missing_bw_indices])
    trimmed_ref_atom_idx = np.concatenate([atom_range(i) for i in present_bw_indices])

    def loading_frac(pc_idx):
        comps   = pca.components_[pc_idx]
        total   = np.sum(comps ** 2)
        imputed = np.sum(comps[missing_feat_idx] ** 2)
        return imputed / total if total > 0 else 0.0

    return {
        'present_feat_idx':     present_feat_idx,
        'missing_feat_idx':     missing_feat_idx,
        'trimmed_ref_atom_idx': trimmed_ref_atom_idx,
        'pc1_imputed_frac':     loading_frac(0),
        'pc2_imputed_frac':     loading_frac(1),
    }


def project_trajectory(u, mobile_ag, ref_ag, pca,
                       imputation=None, report_every=100):
    """
    Iterate all frames, align mobile_ag to ref_ag, and project onto PCA space.

    Returns np.ndarray of shape (n_frames, n_components), dtype float32.
    """
    n_frames = len(u.trajectory)

    if imputation is None:
        n_features = mobile_ag.n_atoms * 3
        vectorized = np.empty((n_frames, n_features), dtype=np.float32)
        print(f"Projecting {n_frames} frame(s)...")
        for i, _ts in enumerate(u.trajectory):
            align.alignto(mobile_ag, ref_ag, select='all', weights='mass')
            vectorized[i] = mobile_ag.positions.flatten()
            if (i + 1) % report_every == 0 or (i + 1) == n_frames:
                print(f"  {i + 1}/{n_frames}", end='\r', flush=True)
    else:
        present_feat_idx     = imputation['present_feat_idx']
        trimmed_ref_atom_idx = imputation['trimmed_ref_atom_idx']
        trimmed_ref_ag       = ref_ag[trimmed_ref_atom_idx]
        n_features           = pca.mean_.shape[0]
        mean_template        = pca.mean_.astype(np.float32)

        vectorized = np.empty((n_frames, n_features), dtype=np.float32)
        print(f"Projecting {n_frames} frame(s) [mean imputation active]...")
        for i, _ts in enumerate(u.trajectory):
            align.alignto(mobile_ag, trimmed_ref_ag, select='all', weights='mass')
            vec = mean_template.copy()
            vec[present_feat_idx] = mobile_ag.positions.flatten()
            vectorized[i] = vec
            if (i + 1) % report_every == 0 or (i + 1) == n_frames:
                print(f"  {i + 1}/{n_frames}", end='\r', flush=True)

    print()
    return pca.transform(vectorized).astype(np.float32)


# ── verbose diagnostic helpers ────────────────────────────────────────────────

def _verbose_mapping_report(bw_map, conserved_bw, resids, u, chain, resid_offset):
    label_to_seqnum = {lbl: seqnum for seqnum, lbl in bw_map.items()
                       if lbl != '-1'}
    chain_sel = f"chainid {chain} and " if chain else ""
    resid_to_resname = {res.resid: res.resname
                        for res in u.select_atoms(f"{chain_sel}backbone").residues}

    print("\n── BW Mapping Table ─────────────────────────────────────────────────────")
    print(f"  {'BW':>8}  {'GPCRdb seq':>10}  {'Traj resid':>10}  {'Resname':>7}  Status")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*15}")

    for label, resid_list in zip(conserved_bw, resids):
        seqnum = label_to_seqnum.get(label)
        if seqnum is None:
            print(f"  {label:>8}  {'—':>10}  {'—':>10}  {'—':>7}  NOT IN BW MAP")
            continue
        traj_resid = seqnum + resid_offset
        if not resid_list:
            print(f"  {label:>8}  {seqnum:>10}  {traj_resid:>10}  {'—':>7}  MISSING IN TRAJ")
            continue
        resname = resid_to_resname.get(traj_resid, '???')
        print(f"  {label:>8}  {seqnum:>10}  {traj_resid:>10}  {resname:>7}  ok")
    print()


def _verbose_loadings_report(pca, conserved_bw, expected_n, n_top=10):
    n_residues    = len(conserved_bw)
    atoms_per_res = expected_n // n_residues
    feats_per_res = atoms_per_res * 3

    print("── PC Loading Contributions by BW Position ──────────────────────────────")
    for pc_idx in range(min(3, pca.n_components_)):
        comps       = pca.components_[pc_idx]
        rms_per_res = np.array([
            np.sqrt(np.sum(comps[i * feats_per_res:(i + 1) * feats_per_res] ** 2))
            for i in range(n_residues)
        ])
        top_inds = np.argsort(rms_per_res)[::-1][:n_top]
        total_sq = np.sum(comps ** 2)

        print(f"\n  PC{pc_idx + 1}  ({pca.explained_variance_ratio_[pc_idx]:.1%} variance)"
              f"  — top {n_top} positions by RMS loading:")
        print(f"  {'BW':>8}  {'RMS loading':>11}  {'% of PC norm':>12}")
        for i in top_inds:
            frac = rms_per_res[i] ** 2 / total_sq
            print(f"  {conserved_bw[i]:>8}  {rms_per_res[i]:>11.4f}  {frac:>12.1%}")
    print()


def _verbose_training_comparison(projections, meta, prefix):
    train_proj_path = f"{prefix}_train_projections.npy"
    if not os.path.exists(train_proj_path):
        print("── Training-set comparison ──────────────────────────────────────────────")
        print(f"  (skipped — '{train_proj_path}' not found)")
        return

    train_proj    = np.load(train_proj_path)
    codes         = meta['codes_retained']
    str_meta      = meta.get('structure_meta', {})
    states        = [str_meta.get(c, {}).get('state') or 'Unknown' for c in codes]
    unique_states = sorted(set(states))
    n_pcs_show    = min(3, projections.shape[1], train_proj.shape[1])

    print("── Training-set state comparison ────────────────────────────────────────")
    print(f"  {'State':>14}  {'N':>5}  "
          + "  ".join(f"{'PC'+str(k+1)+' mean':>10}  {'±':>1}  {'std':>6}"
                      for k in range(n_pcs_show)))

    centroids = {}
    for state in unique_states:
        mask  = np.array([s == state for s in states])
        sub   = train_proj[mask, :n_pcs_show]
        means = sub.mean(axis=0)
        stds  = sub.std(axis=0)
        centroids[state] = means
        pc_cols = "  ".join(f"{means[k]:>10.2f}  ±  {stds[k]:>6.2f}"
                            for k in range(n_pcs_show))
        print(f"  {state:>14}  {mask.sum():>5}  {pc_cols}")

    traj_mean = projections[:, :n_pcs_show].mean(axis=0)
    traj_std  = projections[:, :n_pcs_show].std(axis=0)
    pc_cols   = "  ".join(f"{traj_mean[k]:>10.2f}  ±  {traj_std[k]:>6.2f}"
                          for k in range(n_pcs_show))
    print(f"  {'YOUR TRAJ':>14}  {len(projections):>5}  {pc_cols}")

    print("\n  Euclidean distance from trajectory mean to each state centroid (PC1–PC2):")
    dists = {state: np.linalg.norm(traj_mean[:2] - cent[:2])
             for state, cent in centroids.items()}
    for state, dist in sorted(dists.items(), key=lambda x: x[1]):
        closest = " ← closest" if dist == min(dists.values()) else ""
        print(f"    {state:>14}:  {dist:8.2f}{closest}")
    print()
