# ContactNetworkBuilder lives in chimpss.fultonmarket.contact_network (migrated in Phase 4).
# This module re-exports it under the analysis namespace for cross-cutting use.
from chimpss.fultonmarket.contact_network import ContactNetworkBuilder

__all__ = ["ContactNetworkBuilder"]
