"""Sphinx configuration for ChiMPSS documentation."""

import os
import sys

# Make the installed package importable without a full environment.
# On RTD the package is installed via pip (see .readthedocs.yaml), so
# this path insert is only a local fallback.
sys.path.insert(0, os.path.abspath('../src'))

# ── Project metadata ─────────────────────────────────────────────────────────
project = 'ChiMPSS'
author = 'josephdepaoloboisvert'
copyright = f'2024, {author}'
release = '0.1.2'

# ── Mock heavy simulation dependencies ──────────────────────────────────────
# These packages are not available in the RTD build environment.
# Mocking them lets autodoc import chimpss modules and extract docstrings
# without needing a full conda simulation stack.
autodoc_mock_imports = [
    # OpenMM simulation stack
    "openmm",
    # Trajectory / structure analysis
    "MDAnalysis",
    "mdtraj",
    # Cheminformatics
    "rdkit",
    # Force-field generation
    "openff",
    "openff.toolkit",
    # Structure preparation
    "pdbfixer",
    "parmed",
    "modeller",
    # Parallel-tempering / sampling
    "mpiplus",
    "openmmtools",
    "pymbar",
    # MPI
    "mpi4py",
    # Data / numerics
    "scipy",
    "sklearn",
    "joblib",
    "netCDF4",
    "jax",
    # Visualisation / I/O
    "matplotlib",
    "seaborn",
    "py3Dmol",
    "nglview",
    "IPython",
    # HTTP (pdb/GPCRdb fetching)
    "requests",
]

# ── Extensions ───────────────────────────────────────────────────────────────
extensions = [
    'sphinx.ext.autodoc',       # pull docstrings from source
    'sphinx.ext.napoleon',      # NumPy / Google docstring styles
    'sphinx.ext.viewcode',      # [source] links next to each item
    'sphinx.ext.autosummary',   # summary tables
    'sphinx.ext.intersphinx',   # cross-links to numpy / openmm docs
]

# ── autodoc ──────────────────────────────────────────────────────────────────
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# ── napoleon (NumPy-style docstrings) ────────────────────────────────────────
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── autosummary ──────────────────────────────────────────────────────────────
autosummary_generate = True

# ── intersphinx ──────────────────────────────────────────────────────────────
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
}

# ── HTML output ──────────────────────────────────────────────────────────────
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}
html_static_path = ['_static']

# ── General ──────────────────────────────────────────────────────────────────
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
