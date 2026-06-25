"""
Root conftest: compatibility shims loaded before any test module is imported.
"""
import huggingface_hub

# huggingface_hub >= 1.0 removed is_offline_mode() as a top-level export;
# the vendored integration/transformers fork (4.57.0.dev0) still imports it
# from the package root in ~11 files.  Restore it here so the fork loads
# cleanly on modern huggingface_hub without requiring a submodule bump.
if not hasattr(huggingface_hub, "is_offline_mode"):
    from huggingface_hub import constants as _hf_constants

    def _is_offline_mode() -> bool:
        return bool(_hf_constants.HF_HUB_OFFLINE)

    huggingface_hub.is_offline_mode = _is_offline_mode
