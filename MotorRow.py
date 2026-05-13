# Backward-compatibility shim — retained for one release cycle.
# New code should import from chimpss.motorrow directly:
#   from chimpss.motorrow import MotorRow
from chimpss.motorrow import MotorRow

__all__ = ["MotorRow"]
