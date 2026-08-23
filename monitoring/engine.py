"""Compatibility alias for :mod:`dmi.engine`."""

import sys
from dmi import engine as _canonical

sys.modules[__name__] = _canonical
