"""Compatibility alias for :mod:`dmi.api.v1`."""

import sys
from dmi.api import v1 as _canonical

sys.modules[__name__] = _canonical
