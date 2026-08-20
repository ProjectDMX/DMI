"""Versioned public API for external framework integrations.

Import a concrete version (for example ``monitoring.integration_api.v1``)
rather than importing framework adapters from DMI core.
"""

DMI_INTEGRATION_API_VERSION = 1

__all__ = ["DMI_INTEGRATION_API_VERSION"]
