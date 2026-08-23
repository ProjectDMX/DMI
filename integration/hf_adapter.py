"""Compatibility alias for :mod:`dmi.adapters.huggingface.adapter`."""

import sys
from dmi.adapters.huggingface import adapter as _canonical

sys.modules[__name__] = _canonical
