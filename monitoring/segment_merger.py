"""Compatibility alias for :mod:`dmi.storage.reassembly`."""

import sys
from dmi.storage import reassembly as _canonical

sys.modules[__name__] = _canonical
