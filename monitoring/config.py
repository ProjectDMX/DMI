"""Compatibility alias for :mod:`dmi.config`."""

import sys
from dmi import config as _canonical

sys.modules[__name__] = _canonical
