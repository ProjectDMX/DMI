"""Compatibility alias for :mod:`dmi.storage.internals`."""

import sys
from dmi.storage import internals as _canonical

sys.modules[__name__] = _canonical
