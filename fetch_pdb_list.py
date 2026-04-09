#!/usr/bin/env python
"""
Fetch a list of GPCR PDB codes from GPCRdb and write them to a text file.

Queries the GPCRdb structure list endpoint and applies your choice of filters
(species, activation state, experimental method, resolution, receptor family,
specific proteins).  The output file feeds directly into generate_pca_gpcrs.py.

Usage examples:
  # All human GPCR structures (default)
  python fetch_pdb_list.py

  # Human, inactive only, X-ray ≤ 3.0 Å
  python fetch_pdb_list.py --state Inactive --method xray --max_resolution 3.0

  # All structures for a specific receptor
  python fetch_pdb_list.py --protein 5ht2a_human --species all

  # All structures in a GPCRdb receptor family (slug from --list_families)
  python fetch_pdb_list.py --family 001_001_001_001

  # Print available protein families and exit
  python fetch_pdb_list.py --list_families
"""

import sys
import json
import argparse
from collections import Counter

import requests

GPCRDB = "https://gpcrdb.org"
TIMEOUT = 60


def get(url):
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        '--out', default='pdb_list.txt',
        help='Output file of PDB codes, one per line (default: pdb_list.txt)')
    p.add_argument(
        '--species', default='Homo sapiens',
        help='Species filter (default: "Homo sapiens"). '
             'Pass "" or "all" to include all species.')
    p.add_argument(
        '--state', default='all',
        help='Activation state filter: Active, Inactive, Intermediate, '
             'or "all" (default: all).  Case-insensitive.')
    p.add_argument(
        '--method', default='all',
        choices=['xray', 'em', 'all'],
        help='Experimental method: xray, em, or all (default: all).')
    p.add_argument(
        '--max_resolution', type=float, default=None,
        help='Maximum resolution in Å.  Structures with no resolution '
             'reported (e.g. EM without stated resolution) are kept unless '
             '--method xray is also set.')
    p.add_argument(
        '--protein', nargs='+', default=None,
        help='One or more GPCRdb protein slugs to include exclusively '
             '(e.g. --protein 5ht2a_human adrb2_human).  '
             'Overrides --family.')
    p.add_argument(
        '--family', default=None,
        help='GPCRdb receptor family slug.  All proteins in this family are '
             'included.  Use --list_families to browse available slugs.')
    p.add_argument(
        '--min_date', default=None,
        help='Earliest publication date to include (YYYY-MM-DD).')
    p.add_argument(
        '--list_families', action='store_true',
        help='Print all top-level GPCRdb receptor families with their slugs '
             'and exit.  Use the slug with --family.')
    return p.parse_args()


# ── family lookup ──────────────────────────────────────────────────────────────

def fetch_family_proteins(family_slug):
    """
    Return the set of protein slugs belonging to a GPCRdb family.
    """
    url = f"{GPCRDB}/services/proteinfamily/proteins/{family_slug}/"
    try:
        data = get(url)
    except Exception as exc:
        sys.exit(f"ERROR fetching family '{family_slug}': {exc}\n"
                 f"  Use --list_families to see valid slugs.")
    return {entry['entry_name'] for entry in data}


def list_families():
    """
    Print a human-readable tree of GPCRdb receptor families.
    """
    url = f"{GPCRDB}/services/proteinfamily/"
    try:
        families = get(url)
    except Exception as exc:
        sys.exit(f"ERROR fetching family list: {exc}")

    print(f"\n{'Slug':<35}  {'Name'}")
    print(f"{'─'*35}  {'─'*40}")
    for fam in families:
        slug  = fam.get('slug', '')
        name  = fam.get('name', '')
        depth = slug.count('_') // 3      # rough indentation from slug depth
        indent = '  ' * min(depth, 3)
        print(f"{slug:<35}  {indent}{name}")
    print()


# ── method normalisation ───────────────────────────────────────────────────────

_XRAY_KEYWORDS = {'x-ray', 'xray', 'x ray', 'diffraction', 'crystallography'}
_EM_KEYWORDS   = {'electron', 'cryo-em', 'cryoem', 'em'}

def _classify_method(structure):
    """Return 'xray', 'em', or 'other' for a structure dict."""
    raw = (structure.get('structure_type') or
           structure.get('method') or '').lower()
    if any(k in raw for k in _XRAY_KEYWORDS):
        return 'xray'
    if any(k in raw for k in _EM_KEYWORDS):
        return 'em'
    return 'other'


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list_families:
        list_families()
        return

    # ── 1. Fetch all structures from GPCRdb ───────────────────────────────────
    print(f"Fetching structure list from GPCRdb...")
    try:
        all_structures = get(f"{GPCRDB}/services/structure/")
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"  {len(all_structures)} total structures in GPCRdb")

    # ── 2. Build allowed protein set (family or explicit list) ────────────────
    allowed_proteins = None   # None → no protein-level restriction

    if args.protein:
        allowed_proteins = {p.lower() for p in args.protein}
        print(f"  Protein filter: {sorted(allowed_proteins)}")
    elif args.family:
        print(f"  Fetching proteins for family '{args.family}'...")
        allowed_proteins = fetch_family_proteins(args.family)
        print(f"  {len(allowed_proteins)} proteins in family '{args.family}'")

    # ── 3. Apply filters ──────────────────────────────────────────────────────
    species_filter = args.species.strip()
    use_species    = bool(species_filter) and species_filter.lower() != 'all'
    use_state      = args.state.lower() != 'all'
    use_method     = args.method != 'all'

    kept = []
    drop_reasons = Counter()

    for s in all_structures:
        # protein / family
        if allowed_proteins is not None:
            if (s.get('protein') or '').lower() not in allowed_proteins:
                drop_reasons['protein/family'] += 1
                continue

        # species
        if use_species:
            if (s.get('species') or '').lower() != species_filter.lower():
                drop_reasons['species'] += 1
                continue

        # state
        if use_state:
            state = (s.get('state') or '').lower()
            if state != args.state.lower():
                drop_reasons['state'] += 1
                continue

        # experimental method
        if use_method:
            if _classify_method(s) != args.method:
                drop_reasons['method'] += 1
                continue

        # resolution
        if args.max_resolution is not None:
            res = s.get('resolution')
            if res is not None and float(res) > args.max_resolution:
                drop_reasons['resolution'] += 1
                continue
            # if res is None and method is xray, flag as no-resolution (keep)

        # publication date
        if args.min_date:
            pub = s.get('publication_date') or ''
            if pub < args.min_date:
                drop_reasons['date'] += 1
                continue

        kept.append(s)

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print(f"\nFilter summary:")
    print(f"  {'Total fetched':<30}: {len(all_structures)}")
    for reason, n in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {'Dropped — ' + reason:<30}: {n}")
    print(f"  {'─'*34}")
    print(f"  {'Retained':<30}: {len(kept)}")

    if kept:
        # State breakdown
        states = Counter(s.get('state') or 'Unknown' for s in kept)
        print(f"\n  State breakdown:")
        for state, n in sorted(states.items()):
            print(f"    {state:<20}: {n}")

        # Method breakdown
        methods = Counter(_classify_method(s) for s in kept)
        print(f"\n  Method breakdown:")
        for method, n in sorted(methods.items()):
            print(f"    {method:<20}: {n}")

        # Species breakdown (only show if --species all)
        if not use_species:
            species = Counter(s.get('species') or 'Unknown' for s in kept)
            print(f"\n  Species breakdown (top 5):")
            for sp, n in species.most_common(5):
                print(f"    {sp:<30}: {n}")

    # ── 5. Write output ───────────────────────────────────────────────────────
    pdb_codes = sorted({s['pdb_code'].upper() for s in kept})

    with open(args.out, 'w') as fh:
        fh.write('\n'.join(pdb_codes) + '\n')

    print(f"\nWrote {len(pdb_codes)} PDB codes → {args.out}")
    print(f"  Feed into: python generate_pca_gpcrs.py {args.out}")


if __name__ == '__main__':
    main()
