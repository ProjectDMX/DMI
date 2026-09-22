"""Exceptions raised by the configuration layer."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Base class for every configuration-layer failure."""


class DescriptorError(ConfigurationError):
    """A model descriptor is malformed or describes an unsupported model."""


class UnsupportedConfigVersion(ConfigurationError):
    """An on-disk document declares a version this build cannot parse."""


class ConfigValidationError(ConfigurationError):
    """A configuration is well-formed but not legal for the given model.

    Carries the individual issues so a caller can attach each one to the
    control that produced it instead of showing a single opaque message.
    """

    def __init__(self, issues):
        self.issues = list(issues)
        detail = "; ".join(issue.message for issue in self.issues)
        super().__init__(detail or "Configuration is invalid.")


__all__ = [
    "ConfigurationError",
    "DescriptorError",
    "UnsupportedConfigVersion",
    "ConfigValidationError",
]
