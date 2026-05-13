# Backward-compatibility shim — retained for one release cycle.
# New code should import from chimpss.analysis.distance_matrix directly.
from chimpss.analysis.distance_matrix import DistanceMatrix  # noqa: F401

__all__ = ["DistanceMatrix"]
