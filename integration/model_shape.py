"""Compatibility alias for :mod:`dmi.adapters.huggingface.model_shape`."""

import sys
from dmi.adapters.huggingface import model_shape as _canonical

sys.modules[__name__] = _canonical
