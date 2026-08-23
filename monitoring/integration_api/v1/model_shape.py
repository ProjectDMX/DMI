"""Compatibility alias for :mod:`dmi.api.v1.model_shape`."""

import sys
from dmi.api.v1 import model_shape as _canonical

sys.modules[__name__] = _canonical
