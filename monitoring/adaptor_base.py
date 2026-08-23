"""Compatibility alias for :mod:`dmi.adapters.base`."""

import sys
from dmi.adapters import base as _canonical

sys.modules[__name__] = _canonical
