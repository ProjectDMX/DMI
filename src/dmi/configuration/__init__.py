"""Structured DMI configuration: schema, YAML, validation, and compilation.

YAML does not replace DMI's existing configuration mechanisms. It sits in
front of them::

    config.yaml
        -> load_config()      knows YAML, not the runtime
        -> DMIConfig          canonical in-process representation
        -> compile_config()   knows the runtime, not YAML
        -> CompiledDMIConfig
        -> existing DMI runtime

Typical use::

    from dmi.configuration import load_config, validate_config, compile_config

    config = load_config("attention-debug.dmi.yaml")
    issues = validate_config(config, descriptor)
    compiled = compile_config(config, model_context)

See ``docs/dmi-configurator-plan.md`` for the design.
"""

from __future__ import annotations

from .catalog_adapter import (
    HookInfo,
    catalog_payload,
    describe_hooks,
    grouped_hooks,
    hook_ids,
    per_layer_hook_ids,
)
from .compatibility import from_legacy_hook_selection, to_legacy_hook_selection
from .compiler import CompiledDMIConfig, ModelContext, compile_config
from .errors import (
    ConfigurationError,
    ConfigValidationError,
    DescriptorError,
    UnsupportedConfigVersion,
)
from .manifest import (
    descriptor_to_dict,
    load_descriptor,
    parse_descriptor,
    save_descriptor,
    to_model_shape_config,
)
from .schema import (
    CONFIG_VERSION,
    DESCRIPTOR_SCHEMA_VERSION,
    SUPPORTED_ARCHITECTURES,
    DMIConfig,
    LayerSelection,
    ModelDescriptor,
    ModelIdentity,
    ModelTopology,
    ObservationConfig,
    RuntimePolicy,
)
from .validation import Issue, ensure_valid, is_valid, validate_config
from .yaml import (
    config_to_dict,
    dump_config,
    load_config,
    normalize_config,
    parse_config,
    save_config,
)

__all__ = [
    # schema
    "CONFIG_VERSION",
    "DESCRIPTOR_SCHEMA_VERSION",
    "SUPPORTED_ARCHITECTURES",
    "DMIConfig",
    "LayerSelection",
    "ModelDescriptor",
    "ModelIdentity",
    "ModelTopology",
    "ObservationConfig",
    "RuntimePolicy",
    # descriptors
    "load_descriptor",
    "parse_descriptor",
    "descriptor_to_dict",
    "save_descriptor",
    "to_model_shape_config",
    # catalog
    "HookInfo",
    "hook_ids",
    "per_layer_hook_ids",
    "describe_hooks",
    "grouped_hooks",
    "catalog_payload",
    # yaml
    "load_config",
    "save_config",
    "dump_config",
    "parse_config",
    "config_to_dict",
    "normalize_config",
    # validation
    "Issue",
    "validate_config",
    "is_valid",
    "ensure_valid",
    # compilation
    "ModelContext",
    "CompiledDMIConfig",
    "compile_config",
    # compatibility
    "to_legacy_hook_selection",
    "from_legacy_hook_selection",
    # errors
    "ConfigurationError",
    "DescriptorError",
    "UnsupportedConfigVersion",
    "ConfigValidationError",
]
