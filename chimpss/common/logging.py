"""
Centralized logging utilities for ChiMPSS.

Consolidates the printf lambda previously duplicated across:
  FultonMarketUtils, ContactNetwork, Reporting, GetContactsHelper,
  retro_convergence_utils, DistanceMatrixUtils
"""

from datetime import datetime


def printf(msg):
    """Print a timestamped message to stdout (flushed)."""
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} // {msg}", flush=True)


def timestamp(msg):
    """Return a timestamped string (no print)."""
    return f"{datetime.now()} // {msg}"
