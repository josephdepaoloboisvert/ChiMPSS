# ContactNetworkBuilder lives in chimpss.fultonmarket.contact_network (migrated in Phase 4).
# This module re-exports it under the analysis namespace for cross-cutting use.
import os
import shutil

from chimpss.fultonmarket.contact_network import ContactNetworkBuilder

__all__ = ["ContactNetworkBuilder", "check_getcontacts"]


def check_getcontacts(script_path: str = None) -> str:
    """
    Verify that ``get_dynamic_contacts.py`` is accessible.

    Parameters
    ----------
    script_path : str, optional
        Explicit path to ``get_dynamic_contacts.py``. If omitted, the
        function searches PATH and then ``~/getcontacts/``.

    Returns
    -------
    str
        Resolved path to the script.

    Raises
    ------
    RuntimeError
        If the script cannot be found.
    """
    candidates = []

    if script_path:
        candidates.append(script_path)

    which = shutil.which("get_dynamic_contacts.py")
    if which:
        candidates.append(which)

    candidates.append(
        os.path.expanduser("~/getcontacts/get_dynamic_contacts.py")
    )

    for path in candidates:
        if os.path.isfile(path):
            print(f"get_dynamic_contacts.py found: {path}")
            return path

    raise RuntimeError(
        "get_dynamic_contacts.py not found. "
        "Clone getcontacts (https://github.com/getcontacts/getcontacts) and "
        "either add it to PATH or pass its path explicitly: "
        "check_getcontacts('/path/to/get_dynamic_contacts.py')"
    )
