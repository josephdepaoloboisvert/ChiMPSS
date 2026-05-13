#!/usr/bin/env python
# Backward-compatibility shim — retained for one release cycle.
# The canonical entry point is: chimpss-fetch-pdbs
# New code should import library functions from chimpss.analysis.pdb_fetch.
from chimpss.cli.fetch_pdbs import main

if __name__ == '__main__':
    main()
