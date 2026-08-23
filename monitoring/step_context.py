"""Compatibility alias for :mod:`dmi.adapters.types`."""

import sys
from dmi.adapters import types as _canonical

sys.modules[__name__] = _canonical
