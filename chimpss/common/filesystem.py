"""
Filesystem utilities for ChiMPSS.

Consolidates: utility/General.py (ensure_exists)
"""

import os


def ensure_exists(directory):
    """Create *directory* (and parents) if it does not already exist."""
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return True
