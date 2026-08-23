"""Compatibility alias for :mod:`dmi.transport.ring`."""

import sys
from dmi.transport import ring as _canonical

sys.modules[__name__] = _canonical
