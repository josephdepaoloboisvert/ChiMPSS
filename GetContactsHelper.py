# Backward-compatibility shim — retained for one release cycle.
# New code should import from chimpss.fultonmarket.contact_network directly.
from chimpss.fultonmarket.contact_network import ContactNetworkBuilder  # noqa: F401

__all__ = ["ContactNetworkBuilder"]
