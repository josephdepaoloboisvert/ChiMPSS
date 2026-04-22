import faulthandler
import warnings
import numpy as np

# Configure simulation environment once at package import.
# These settings mirror the originals scattered across FultonMarket modules.
faulthandler.enable()
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

from chimpss.fultonmarket.fulton_market import FultonMarket
from chimpss.fultonmarket.randolph import Randolph
from chimpss.fultonmarket.analysis import FultonMarketAnalysis
from chimpss.fultonmarket.utils import (
    printf,
    geometric_distribution,
    build_sampler_states,
    truncate_ncdf,
    frobenius_norm,
    jsd_distance_matrices,
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
]
