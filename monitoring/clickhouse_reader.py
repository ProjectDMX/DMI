"""Compatibility alias for :mod:`dmi.storage.clickhouse`."""

import sys
from dmi.storage import clickhouse as _canonical

sys.modules[__name__] = _canonical
