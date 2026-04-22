# Backward-compatibility shim — retained for one release cycle.
# New code should import from chimpss.analysis.distance_matrix directly.
from chimpss.analysis.distance_matrix import (  # noqa: F401
    _calc_torsions,
    _calc_torsion_matrix,
    _calc_CA_dist,
    _calc_CA_matrix,
    _calc_hbond_dist,
    _calc_hbond_matrix,
    _compute_component_matrices,
    _compute_matrix,
)
