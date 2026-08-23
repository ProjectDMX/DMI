"""Compatibility alias for :mod:`dmi.hooks.point`."""

import sys
from dmi.hooks import point as _canonical

sys.modules[__name__] = _canonical
