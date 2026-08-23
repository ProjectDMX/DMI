"""Compatibility alias for :mod:`dmi.hooks.selection`."""

import sys
from dmi.hooks import selection as _canonical

sys.modules[__name__] = _canonical
