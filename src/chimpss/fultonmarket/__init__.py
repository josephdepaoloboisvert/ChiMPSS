import faulthandler
import warnings

import numpy as np

# Configure simulation environment once at package import.
# These settings mirror the originals scattered across FultonMarket modules.
faulthandler.enable()
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

from chimpss.fultonmarket.analysis import FultonMarketAnalysis
from chimpss.fultonmarket.fulton_market import FultonMarket
from chimpss.fultonmarket.randolph import Randolph
from chimpss.fultonmarket.retro_convergence import (
    add_equil_metric_to_plot,
    plot_convergence_metric,
)
from chimpss.fultonmarket.utils import (
    build_sampler_states,
    frobenius_norm,
    geometric_distribution,
    jsd_distance_matrices,
    printf,
    truncate_ncdf,
)

__all__ = [
    "FultonMarket",
    "Randolph",
    "FultonMarketAnalysis",
    "printf",
    "geometric_distribution",
    "build_sampler_states",
    "truncate_ncdf",
    "frobenius_norm",
    "jsd_distance_matrices",
    "plot_convergence_metric",
    "add_equil_metric_to_plot",
]
