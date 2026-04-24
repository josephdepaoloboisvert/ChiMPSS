"""
Library functions for querying the GPCRdb structure list.

CLI entry point: chimpss-fetch-pdbs  (src/chimpss/cli/fetch_pdbs.py)
"""

import sys

import requests

GPCRDB = "https://gpcrdb.org"
TIMEOUT = 60


def get(url):
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_family_proteins(family_slug):
    """Return the set of protein slugs belonging to a GPCRdb family."""
    url = f"{GPCRDB}/services/proteinfamily/proteins/{family_slug}/"
    try:
        data = get(url)
    except Exception as exc:
        sys.exit(f"ERROR fetching family '{family_slug}': {exc}\n"
                 f"  Use --list_families to see valid slugs.")
    return {entry['entry_name'] for entry in data}


def list_families():
    """Print a human-readable tree of GPCRdb receptor families."""
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
        depth = slug.count('_') // 3
        indent = '  ' * min(depth, 3)
        print(f"{slug:<35}  {indent}{name}")
    print()


_XRAY_KEYWORDS = {'x-ray', 'xray', 'x ray', 'diffraction', 'crystallography'}
_EM_KEYWORDS   = {'electron', 'cryo-em', 'cryoem', 'em'}


def classify_method(structure):
    """Return 'xray', 'em', or 'other' for a structure dict."""
    raw = (structure.get('structure_type') or
           structure.get('method') or '').lower()
    if any(k in raw for k in _XRAY_KEYWORDS):
        return 'xray'
    if any(k in raw for k in _EM_KEYWORDS):
        return 'em'
    return 'other'
