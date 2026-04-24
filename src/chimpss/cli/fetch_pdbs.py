"""
CLI entry point for fetching GPCR PDB codes from GPCRdb.

Console script: chimpss-fetch-pdbs

Usage examples:
  chimpss-fetch-pdbs
  chimpss-fetch-pdbs --state Inactive --method xray --max_resolution 3.0
  chimpss-fetch-pdbs --protein 5ht2a_human adrb2_human
  chimpss-fetch-pdbs --family 001_001_001_001
  chimpss-fetch-pdbs --list_families
"""

import argparse
import sys
from collections import Counter

from chimpss.analysis.pdb_fetch import (
    GPCRDB,
    classify_method,
    fetch_family_proteins,
    get,
    list_families,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='pdb_list.txt',
                   help='Output file of PDB codes, one per line (default: pdb_list.txt)')
    p.add_argument('--species', default='Homo sapiens',
                   help='Species filter (default: "Homo sapiens"). '
                        'Pass "" or "all" to include all species.')
    p.add_argument('--state', default='all',
                   help='Activation state filter: Active, Inactive, Intermediate, '
                        'or "all" (default: all). Case-insensitive.')
    p.add_argument('--method', default='all',
                   choices=['xray', 'em', 'all'],
                   help='Experimental method: xray, em, or all (default: all).')
    p.add_argument('--max_resolution', type=float, default=None,
                   help='Maximum resolution in Angstroms.')
    p.add_argument('--protein', nargs='+', default=None,
                   help='One or more GPCRdb protein slugs to include exclusively.')
    p.add_argument('--family', default=None,
                   help='GPCRdb receptor family slug. Use --list_families to browse.')
    p.add_argument('--min_date', default=None,
                   help='Earliest publication date to include (YYYY-MM-DD).')
    p.add_argument('--list_families', action='store_true',
                   help='Print all GPCRdb receptor families with their slugs and exit.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_families:
        list_families()
        return

    print("Fetching structure list from GPCRdb...")
    try:
        all_structures = get(f"{GPCRDB}/services/structure/")
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"  {len(all_structures)} total structures in GPCRdb")

    allowed_proteins = None
    if args.protein:
        allowed_proteins = {p.lower() for p in args.protein}
        print(f"  Protein filter: {sorted(allowed_proteins)}")
    elif args.family:
        print(f"  Fetching proteins for family '{args.family}'...")
        allowed_proteins = fetch_family_proteins(args.family)
        print(f"  {len(allowed_proteins)} proteins in family '{args.family}'")

    species_filter = args.species.strip()
    use_species    = bool(species_filter) and species_filter.lower() != 'all'
    use_state      = args.state.lower() != 'all'
    use_method     = args.method != 'all'

    kept = []
    drop_reasons = Counter()

    for s in all_structures:
        if allowed_proteins is not None:
            if (s.get('protein') or '').lower() not in allowed_proteins:
                drop_reasons['protein/family'] += 1
                continue
        if use_species:
            if (s.get('species') or '').lower() != species_filter.lower():
                drop_reasons['species'] += 1
                continue
        if use_state:
            if (s.get('state') or '').lower() != args.state.lower():
                drop_reasons['state'] += 1
                continue
        if use_method:
            if classify_method(s) != args.method:
                drop_reasons['method'] += 1
                continue
        if args.max_resolution is not None:
            res = s.get('resolution')
            if res is not None and float(res) > args.max_resolution:
                drop_reasons['resolution'] += 1
                continue
        if args.min_date:
            pub = s.get('publication_date') or ''
            if pub < args.min_date:
                drop_reasons['date'] += 1
                continue
        kept.append(s)

    print("\nFilter summary:")
    print(f"  {'Total fetched':<30}: {len(all_structures)}")
    for reason, n in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {'Dropped — ' + reason:<30}: {n}")
    print(f"  {'─'*34}")
    print(f"  {'Retained':<30}: {len(kept)}")

    if kept:
        states = Counter(s.get('state') or 'Unknown' for s in kept)
        print("\n  State breakdown:")
        for state, n in sorted(states.items()):
            print(f"    {state:<20}: {n}")

        methods = Counter(classify_method(s) for s in kept)
        print("\n  Method breakdown:")
        for method, n in sorted(methods.items()):
            print(f"    {method:<20}: {n}")

        if not use_species:
            species = Counter(s.get('species') or 'Unknown' for s in kept)
            print("\n  Species breakdown (top 5):")
            for sp, n in species.most_common(5):
                print(f"    {sp:<30}: {n}")

    pdb_codes = sorted({s['pdb_code'].upper() for s in kept})
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(pdb_codes) + '\n')

    print(f"\nWrote {len(pdb_codes)} PDB codes → {args.out}")
    print(f"  Feed into: chimpss-generate-pca {args.out}")


if __name__ == '__main__':
    main()
