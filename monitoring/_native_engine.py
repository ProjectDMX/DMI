"""Compatibility alias for :mod:`dmi.transport.native`."""

import sys
from dmi.transport import native as _canonical

sys.modules[__name__] = _canonical
