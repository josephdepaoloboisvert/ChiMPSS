# Backward-compatibility shim — retained for one release cycle.
import sys
import types

from chimpss.fultonmarket.analysis import FultonMarketAnalysis

__all__ = ["FultonMarketAnalysis"]

# `from FultonMarket import FultonMarketAnalysis` returns this module (not the class)
# because Python's import system always prefers a submodule over a package-level binding.
# Make the module itself callable so `callable(FultonMarketAnalysis)` is True.
class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return FultonMarketAnalysis(*args, **kwargs)

sys.modules[__name__].__class__ = _CallableModule
