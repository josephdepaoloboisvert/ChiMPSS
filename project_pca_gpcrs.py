#!/usr/bin/env python
# Backward-compatibility shim — retained for one release cycle.
# The canonical entry point is: chimpss-project-pca
# New code should import library functions from chimpss.analysis.gpcr_pca.
from chimpss.cli.project_pca import main

if __name__ == '__main__':
    main()
