from __future__ import annotations

"""Hook Points.

Helpers to access activations in models.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn
import torch.utils.hooks as hooks
from torch import Tensor

from .utils import Slice, SliceInput

from .engine import MonitoringEngine

try:
    from torch.cuda import nvtx as _nvtx
except Exception:  # pragma: no cover - nvtx not always available
    _nvtx = None

_MONITORING_DEBUG = False


def set_monitoring_debug(enabled: bool) -> None:
    """Set shared debug mode for hook-side NVTX/stat paths."""

    global _MONITORING_DEBUG
    _MONITORING_DEBUG = bool(enabled)


def _nvtx_enabled() -> bool:
    return bool(_MONITORING_DEBUG and _nvtx is not None)


def _hook_stats_enabled() -> bool:
    return bool(_MONITORING_DEBUG)

# Global accumulators for lightweight hook-side profiling (microseconds)
_hook_total_calls = 0

# For synchronous path diagnostics
_sync_build_us = 0.0
_sync_move_us = 0.0
_sync_remove_batch_us = 0.0
_sync_slice_us = 0.0
_sync_cache_set_us = 0.0
_per_hook_sync_move_us = defaultdict(float)
_per_hook_sync_build_us = defaultdict(float)
_per_hook_sync_slice_us = defaultdict(float)

def get_monitoring_hook_stats() -> dict:
    """Return aggregated hook-side profiling stats (cheap to call)."""
    return {
        "total_calls": _hook_total_calls,
        "sync_build_us": int(_sync_build_us),
        "sync_move_us": int(_sync_move_us),
        "sync_remove_batch_us": int(_sync_remove_batch_us),
        "sync_slice_us": int(_sync_slice_us),
        "sync_cache_set_us": int(_sync_cache_set_us),
        "per_hook_top": {
            "sync_build_us": sorted(_per_hook_sync_build_us.items(), key=lambda kv: -kv[1])[:10],
            "sync_move_us": sorted(_per_hook_sync_move_us.items(), key=lambda kv: -kv[1])[:10],
            "sync_slice_us": sorted(_per_hook_sync_slice_us.items(), key=lambda kv: -kv[1])[:10],
        },
    }


@contextmanager
def _nvtx_range(message: str):
    if _nvtx_enabled():
        _nvtx.range_push(message)
        try:
            yield
        finally:
            _nvtx.range_pop()
    else:
        yield


@dataclass
class LensHandle:
    """Dataclass that holds information about a PyTorch hook."""

    hook: hooks.RemovableHandle
    """Reference to the Hook's Removable Handle."""

    is_permanent: bool = False
    """Indicates if the Hook is Permanent."""

    context_level: Optional[int] = None
    """Context level associated with the hooks context manager for the given hook."""


# Define type aliases
NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]


@runtime_checkable
class _HookFunctionProtocol(Protocol):
    """Protocol for hook functions."""

    def __call__(self, tensor: Tensor, *, hook: "HookPoint") -> Union[Any, None]:
        ...


HookFunction = _HookFunctionProtocol  # Callable[..., _HookFunctionProtocol]

DeviceType = Optional[torch.device]
_grad_t = Union[tuple[Tensor, ...], Tensor]


class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """

    def __init__(self):
        super().__init__()
        self.fwd_hooks: list[LensHandle] = []
        self.bwd_hooks: list[LensHandle] = []
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name: Optional[str] = None

    def add_perma_hook(self, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd") -> None:
        self.add_hook(hook, dir=dir, is_permanent=True)

    def add_hook(
        self,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
    ) -> None:
        """
        Hook format is fn(activation, hook_name)
        Change it into PyTorch hook format (this includes input and output,
        which are the same for a HookPoint)
        If prepend is True, add this hook before all other hooks
        """

        def full_hook(
            module: torch.nn.Module,
            module_input: Any,
            module_output: Any,
        ):
            if (
                dir == "bwd"
            ):  # For a backwards hook, module_output is a tuple of (grad,) - I don't know why.
                module_output = module_output[0]
            hook_name = self.name or "unnamed"
            with _nvtx_range(f"TL::Hook[{hook_name}:{dir}]"):
                return hook(module_output, hook=self)

        # annotate the `full_hook` with the string representation of the `hook` function
        if isinstance(hook, partial):
            # partial.__repr__() can be extremely slow if arguments contain large objects, which
            # is common when caching tensors.
            full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
        else:
            full_hook.__name__ = hook.__repr__()

        # NVTX tag to capture hook registration cost
        try:
            from torch.cuda import nvtx as _nvtx  # type: ignore
        except Exception:
            _nvtx = None  # type: ignore
        nvtx_enabled = _nvtx_enabled()
        if nvtx_enabled:
            hk = self.name or "unnamed"
            _nvtx.range_push(f"TL::RegisterHook[{hk}:{dir}]")

        if dir == "fwd":
            pt_handle = self.register_forward_hook(full_hook, prepend=prepend)
            visible_hooks = self.fwd_hooks
        elif dir == "bwd":
            pt_handle = self.register_full_backward_hook(full_hook, prepend=prepend)
            visible_hooks = self.bwd_hooks
        else:
            raise ValueError(f"Invalid direction {dir}")

        handle = LensHandle(pt_handle, is_permanent, level)

        if prepend:
            # we could just pass this as an argument in PyTorch 2.0, but for now we manually do this...
            visible_hooks.insert(0, handle)

        else:
            visible_hooks.append(handle)

        if nvtx_enabled:
            _nvtx.range_pop()

    def remove_hooks(
        self,
        dir: Literal["fwd", "bwd", "both"] = "fwd",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ) -> None:
        def _remove_hooks(handles: list[LensHandle]) -> list[LensHandle]:
            output_handles = []
            for handle in handles:
                if including_permanent:
                    handle.hook.remove()
                elif (not handle.is_permanent) and (level is None or handle.context_level == level):
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles

        if dir == "fwd" or dir == "both":
            self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        if dir == "bwd" or dir == "both":
            self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def forward(self, x: Tensor) -> Tensor:
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        if self.name is None:
            raise ValueError("Name cannot be None")
        split_name = self.name.split(".")
        return int(split_name[1])


# %%

__all__ = [
    "HookPoint",
    "HookedRootModule",
    "HookFunction",
    "NamesFilter",
    "LensHandle",
    "get_monitoring_hook_stats",
]
class HookedRootModule(nn.Module):
    """A class building on nn.Module to interface nicely with HookPoints.

    Adds various nice utilities, most notably run_with_hooks to run the model with temporary hooks,
    and run_with_cache to run the model on some input and return a cache of all activations.

    Notes:

    The main footgun with PyTorch hooking is that hooks are GLOBAL state. If you add a hook to the
    module, and then run it a bunch of times, the hooks persist. If you debug a broken hook and add
    the fixed version, the broken one is still there. To solve this, run_with_hooks will remove
    hooks at the end by default, and I recommend using the API of this and run_with_cache. If you
    want to add hooks into global state, I recommend being intentional about this, and I recommend
    using reset_hooks liberally in your code to remove any accidentally remaining global state.

    The main time this goes wrong is when you want to use backward hooks (to cache or intervene on
    gradients). In this case, you need to keep the hooks around as global state until you've run
    loss.backward() (and so need to disable the reset_hooks_end flag on run_with_hooks)
    """

    name: Optional[str]
    mod_dict: dict[str, nn.Module]
    hook_dict: dict[str, HookPoint]

    def __init__(self, *args: Any):
        super().__init__()
        self.is_caching = False
        self.context_level = 0
        self.monitoring_engine: Optional[MonitoringEngine] = None
        # Native global callback registration state
        self._native_callbacks_registered: bool = False
        self._native_handles: dict[str, hooks.RemovableHandle] = {}
        self._native_enabled_hooks_key: Optional[int] = None

    def setup(self):
        """
        Sets up model.

        This function must be called in the model's `__init__` method AFTER defining all layers. It
        adds a parameter to each module containing its name, and builds a dictionary mapping module
        names to the module instances. It also initializes a hook dictionary for modules of type
        "HookPoint".
        """
        self.mod_dict = {}
        self.hook_dict = {}
        for name, module in self.named_modules():
            if name == "":
                continue
            module.name = name
            self.mod_dict[name] = module
            # TODO: is the bottom line the same as "if "HookPoint" in str(type(module)):"
            if isinstance(module, HookPoint):
                self.hook_dict[name] = module

    def hook_points(self):
        return self.hook_dict.values()

    def remove_all_hook_fns(
        self,
        direction: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ):
        for hp in self.hook_points():
            hp.remove_hooks(direction, including_permanent=including_permanent, level=level)

    def clear_contexts(self):
        for hp in self.hook_points():
            hp.clear_context()

    def reset_hooks(
        self,
        clear_contexts: bool = True,
        direction: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ):
        try:
            from torch.cuda import nvtx as _nvtx  # type: ignore
        except Exception:
            _nvtx = None  # type: ignore
        nvtx_enabled = _nvtx_enabled()
        if nvtx_enabled:
            _nvtx.range_push("TL::ResetHooks")
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction, including_permanent, level=level)
        self.is_caching = False
        if nvtx_enabled:
            _nvtx.range_pop()

    def get_monitoring_engine(self) -> MonitoringEngine:
        """Return a monitoring engine instance, creating one on demand."""

        if self.monitoring_engine is None:
            self.monitoring_engine = MonitoringEngine()
        return self.monitoring_engine

    def check_and_add_hook(
        self,
        hook_point: HookPoint,
        hook_point_name: str,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
    ) -> None:
        """Runs checks on the hook, and then adds it to the hook point"""

        self.check_hooks_to_add(
            hook_point,
            hook_point_name,
            hook,
            dir=dir,
            is_permanent=is_permanent,
            prepend=prepend,
        )
        hook_point.add_hook(hook, dir=dir, is_permanent=is_permanent, level=level, prepend=prepend)

    def check_hooks_to_add(
        self,
        hook_point: HookPoint,
        hook_point_name: str,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        prepend: bool = False,
    ) -> None:
        """Override this function to add checks on which hooks should be added"""
        pass

    def add_hook(
        self,
        name: Union[str, Callable[[str], bool]],
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
    ) -> None:
        if isinstance(name, str):
            hook_point = self.mod_dict[name]
            assert isinstance(
                hook_point, HookPoint
            )  # TODO does adding assert meaningfully slow down performance? I've added them for type checking purposes.
            self.check_and_add_hook(
                hook_point,
                name,
                hook,
                dir=dir,
                is_permanent=is_permanent,
                level=level,
                prepend=prepend,
            )
        else:
            # Otherwise, name is a Boolean function on names
            for hook_point_name, hp in self.hook_dict.items():
                if name(hook_point_name):
                    self.check_and_add_hook(
                        hp,
                        hook_point_name,
                        hook,
                        dir=dir,
                        is_permanent=is_permanent,
                        level=level,
                        prepend=prepend,
                    )

    def add_perma_hook(
        self,
        name: Union[str, Callable[[str], bool]],
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
    ) -> None:
        self.add_hook(name, hook, dir=dir, is_permanent=True)

    def _enable_hook_with_name(self, name: str, hook: Callable, dir: Literal["fwd", "bwd"]):
        """This function takes a key for the mod_dict and enables the related hook for that module

        Args:
            name (str): The module name
            hook (Callable): The hook to add
            dir (Literal[&quot;fwd&quot;, &quot;bwd&quot;]): The direction for the hook
        """
        self.mod_dict[name].add_hook(hook, dir=dir, level=self.context_level)  # type: ignore[operator]

    def _enable_hooks_for_points(
        self,
        hook_points: Iterable[tuple[str, HookPoint]],
        enabled: Callable,
        hook: Callable,
        dir: Literal["fwd", "bwd"],
    ):
        """Enables hooks for a list of points

        Args:
            hook_points (Dict[str, HookPoint]): The hook points
            enabled (Callable): _description_
            hook (Callable): _description_
            dir (Literal[&quot;fwd&quot;, &quot;bwd&quot;]): _description_
        """
        for hook_name, hook_point in hook_points:
            if enabled(hook_name):
                hook_point.add_hook(hook, dir=dir, level=self.context_level)

    def _enable_hook(self, name: Union[str, Callable], hook: Callable, dir: Literal["fwd", "bwd"]):
        """Enables an individual hook on a hook point

        Args:
            name (str): The name of the hook
            hook (Callable): The actual hook
            dir (Literal[&quot;fwd&quot;, &quot;bwd&quot;], optional): The direction of the hook. Defaults to "fwd".
        """
        if isinstance(name, str):
            self._enable_hook_with_name(name=name, hook=hook, dir=dir)
        else:
            self._enable_hooks_for_points(
                hook_points=self.hook_dict.items(), enabled=name, hook=hook, dir=dir
            )

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        """
        A context manager for adding temporary hooks to the model.

        Args:
            fwd_hooks: List[Tuple[name, hook]], where name is either the name of a hook point or a
                Boolean function on hook names and hook is the function to add to that hook point.
            bwd_hooks: Same as fwd_hooks, but for the backward pass.
            reset_hooks_end (bool): If True, removes all hooks added by this context manager when the context manager exits.
            clear_contexts (bool): If True, clears hook contexts whenever hooks are reset.

        Example:

        .. code-block:: python

            with model.hooks(fwd_hooks=my_hooks):
                hooked_loss = model(text, return_type="loss")
        """
        try:
            self.context_level += 1

            try:
                from torch.cuda import nvtx as _nvtx  # type: ignore
            except Exception:
                _nvtx = None  # type: ignore
            nvtx_enabled = _nvtx_enabled()
            if nvtx_enabled:
                _nvtx.range_push("TL::EnableHooks[fwd]")
            for name, hook in fwd_hooks:
                self._enable_hook(name=name, hook=hook, dir="fwd")
            if nvtx_enabled:
                _nvtx.range_pop()
            if bwd_hooks:
                if nvtx_enabled:
                    _nvtx.range_push("TL::EnableHooks[bwd]")
                for name, hook in bwd_hooks:
                    self._enable_hook(name=name, hook=hook, dir="bwd")
                if nvtx_enabled:
                    _nvtx.range_pop()
            yield self
        finally:
            if reset_hooks_end:
                self.reset_hooks(
                    clear_contexts, including_permanent=False, level=self.context_level
                )
            self.context_level -= 1

    def run_with_hooks(
        self,
        *model_args: Any,  # TODO: unsure about whether or not this Any typing is correct or not; may need to be replaced with something more specific?
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any,
    ):
        """
        Runs the model with specified forward and backward hooks.

        Args:
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                either the name of a hook point or a boolean function on hook names, and hook is the
                function to add to that hook point. Hooks with names that evaluate to True are added
                respectively.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                backward pass.
            reset_hooks_end (bool): If True, all hooks are removed at the end, including those added
                during this run. Default is True.
            clear_contexts (bool): If True, clears hook contexts whenever hooks are reset. Default is
                False.
            *model_args: Positional arguments for the model.
            **model_kwargs: Keyword arguments for the model's forward function. See your related
                models forward pass for details as to what sort of arguments you can pass through.

        Note:
            If you want to use backward hooks, set `reset_hooks_end` to False, so the backward hooks
            remain active. This function only runs a forward pass.
        """
        if len(bwd_hooks) > 0 and reset_hooks_end:
            logging.warning(
                "WARNING: Hooks will be reset at the end of run_with_hooks. This removes the backward hooks before a backward pass can occur."
            )

        with self.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_model:
            return hooked_model.forward(*model_args, **model_kwargs)

    def add_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device: DeviceType = None,  # TODO: unsure about whether or not this device typing is correct or not?
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> dict:
        """Adds hooks to the model to cache activations. Note: It does NOT actually run the model to get activations, that must be done separately.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Defaults to same device as model.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list

        assert callable(names_filter), "names_filter must be a callable"

        self.is_caching = True

        def save_hook(tensor: Tensor, hook: HookPoint, is_backward: bool):
            assert hook.name is not None
            hook_name = hook.name
            if is_backward:
                hook_name += "_grad"
            if remove_batch_dim:
                cache[hook_name] = tensor.detach().to(device)[0]
            else:
                cache[hook_name] = tensor.detach().to(device)

        for name, hp in self.hook_dict.items():
            if names_filter(name):
                hp.add_hook(partial(save_hook, is_backward=False), "fwd")
                if incl_bwd:
                    hp.add_hook(partial(save_hook, is_backward=True), "bwd")
        return cache

    def run_with_cache(
        self,
        *model_args: Any,
        forward_fn: Optional[Callable[..., Any]] = None,
        names_filter: NamesFilter = None,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
        incl_bwd: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
        **model_kwargs: Any,
    ):
        """
        Runs the model and returns the model output and a Cache object.

        Args:
            *model_args: Positional arguments for the model.
            forward_fn (Callable, optional): Override the forward callable used to run the model.
                Defaults to calling the model instance directly.
            names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
                list of str, or a function that takes a string and returns a bool. Defaults to None, which
                means cache everything.
            device (str or torch.Device, optional): The device to cache activations on. Defaults to the
                model device. WARNING: Setting a different device than the one used by the model leads to
                significant performance degradation.
            remove_batch_dim (bool, optional): If True, removes the batch dimension when caching. Only
                makes sense with batch_size=1 inputs. Defaults to False.
            incl_bwd (bool, optional): If True, calls backward on the model output and caches gradients
                as well. Assumes that the model outputs a scalar (e.g., return_type="loss"). Custom loss
                functions are not supported. Defaults to False.
            reset_hooks_end (bool, optional): If True, removes all hooks added by this function at the
                end of the run. Defaults to True.
            clear_contexts (bool, optional): If True, clears hook contexts whenever hooks are reset.
                Defaults to False.
            pos_slice:
                The slice to apply to the cache output. Defaults to None, do nothing.
            **model_kwargs: Keyword arguments for the model's forward function. See your related
                models forward pass for details as to what sort of arguments you can pass through.

        Returns:
            tuple: A tuple containing the model output and a Cache object.

        """

        pos_slice = Slice.unwrap(pos_slice)
        call_forward = forward_fn if forward_fn is not None else self

        # Build hooks list and cache dict (CPU work)
        with _nvtx_range("TL::GetCachingHooks"):
            cache_dict, fwd, bwd = self.get_caching_hooks(
                names_filter,
                incl_bwd,
                device,
                remove_batch_dim=remove_batch_dim,
                pos_slice=pos_slice,
            )

        # 若使用原生全局回调且本步没有需要临时挂载的 hooks，则跳过 hooks 上下文，避免每步 ResetHooks 开销
        use_ctx = True
        try:
            engine = self.monitoring_engine
            native_backend = getattr(engine, "_native_backend", None) if engine is not None else None
            native_using = bool(getattr(engine, "_using_native_backend", False) and native_backend is not None)
            native_callback_active = native_using
            if native_callback_active and not fwd and not bwd:
                use_ctx = False
        except Exception:
            use_ctx = True

        if use_ctx:
            with self.hooks(
                fwd_hooks=fwd,
                bwd_hooks=bwd,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
            ):
                # Main forward (and optional backward) on the critical path
                with _nvtx_range("TL::ModelForward"):
                    model_out = call_forward(*model_args, **model_kwargs)
                    if incl_bwd:
                        model_out.backward()
        else:
            # Forward without transient hook context (native global callback active)
            with _nvtx_range("TL::ModelForward"):
                model_out = call_forward(*model_args, **model_kwargs)
                if incl_bwd:
                    model_out.backward()

        # 若使用原生全局回调，前向结束后一次性收集 futures 写入 cache
        try:
            engine = self.monitoring_engine
            if engine is not None:
                if hasattr(engine, "is_capture_enabled") and not engine.is_capture_enabled():
                    return model_out, cache_dict
                native_backend = getattr(engine, "_native_backend", None)
                native_using = bool(getattr(engine, "_using_native_backend", False) and native_backend is not None)
                native_callback_active = native_using
                if native_callback_active and native_backend is not None:
                    # Bulk fill Python cache with BackendFutures for this step
                    with _nvtx_range("MonEng::CollectFutures"):
                        step_id_i = int(getattr(engine, "_current_step_id", 0))
                        native_backend.collect_step_futures_into(step_id_i, cache_dict)
        except Exception:
            pass

        # Register host_engine DB submission (native backend only).
        try:
            engine = self.monitoring_engine
            if engine is not None and hasattr(engine, "_register_db_step"):
                input_ids = model_kwargs.get("input_ids") if model_kwargs else None
                if input_ids is None and model_args:
                    input_ids = model_args[0]
                attention_mask = model_kwargs.get("attention_mask") if model_kwargs else None
                past_key_values = model_kwargs.get("past_key_values") if model_kwargs else None
                engine._register_db_step(cache_dict, input_ids, attention_mask, past_key_values)
        except Exception:
            pass

        return model_out, cache_dict

    def prepare_monitoring(
        self,
        *,
        names_filter: NamesFilter = None,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
    ) -> None:
        engine = self.monitoring_engine
        if engine is None:
            return

        pos_slice = Slice.unwrap(pos_slice)

        native_backend = getattr(engine, "_native_backend", None)
        native_using = bool(getattr(engine, "_using_native_backend", False) and native_backend is not None)
        native_callback_active = native_using
        if not native_callback_active or native_backend is None:
            return

        precomputed_enabled_names: Optional[list[str]] = None
        selection_full = False
        if names_filter is None:
            engine_config = getattr(engine, "config", None)
            hooks = getattr(engine_config, "hooks", None) if engine_config is not None else None
            if hooks is not None and hasattr(hooks, "is_full") and hooks.is_full():
                selection_full = True
            elif hasattr(engine, "get_compiled_hook_names"):
                compiled = engine.get_compiled_hook_names(self.hook_dict.keys(), cache_key=id(self.hook_dict))
                if compiled is not None:
                    precomputed_enabled_names, compiled_set = compiled
                    names_filter = compiled_set

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        elif isinstance(names_filter, set):
            filter_set = names_filter
            names_filter = lambda name: name in filter_set
        elif callable(names_filter):
            names_filter = names_filter
        else:
            raise ValueError("names_filter must be a string, list of strings, or function")

        enabled_names: list[str] = []
        if not selection_full:
            if precomputed_enabled_names is not None:
                enabled_names = precomputed_enabled_names
            else:
                for _nm, _ in self.hook_dict.items():
                    if names_filter(_nm):
                        enabled_names.append(_nm)

        if selection_full:
            if self._native_enabled_hooks_key != -1:
                try:
                    native_backend.set_enabled_hooks(None)
                    self._native_enabled_hooks_key = -1
                except Exception:
                    pass
        else:
            enabled_key = id(enabled_names)
            if self._native_enabled_hooks_key != enabled_key:
                try:
                    native_backend.set_enabled_hooks(enabled_names)
                    self._native_enabled_hooks_key = enabled_key
                except Exception:
                    pass

        if self._native_callbacks_registered:
            return

        try:
            from monitoring.task import _encode_slice_native  # type: ignore
        except Exception:
            _encode_slice_native = None  # type: ignore

        for reg_name, reg_hp in self.hook_dict.items():
            slice_tuple = _encode_slice_native(pos_slice) if _encode_slice_native is not None else (0,)
            callback = native_backend.create_global_hook_callback_sig(
                reg_name,
                bool(remove_batch_dim),
                slice_tuple,
                device if device is not None else None,
            )

            def gated_callback(
                tensor: Tensor,
                *,
                hook: HookPoint,
                _cb=callback,
                _root=self,
            ):
                try:
                    engine = getattr(_root, "monitoring_engine", None)
                    if engine is not None and not engine.is_capture_enabled():
                        return None
                except Exception:
                    pass
                return _cb(tensor)

            reg_hp.add_hook(gated_callback, dir="fwd", is_permanent=True)
        self._native_callbacks_registered = True

    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
    ) -> tuple[dict, list, list]:
        """Creates hooks to cache activations. Note: It does not add the hooks to the model.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
            fwd_hooks (list): The forward hooks.
            bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
        """
        if cache is None:
            cache = {}

        pos_slice = Slice.unwrap(pos_slice)

        engine_capture_enabled = True
        if self.monitoring_engine is not None:
            try:
                engine_capture_enabled = bool(self.monitoring_engine.is_capture_enabled())
            except Exception:
                engine_capture_enabled = True

        precomputed_enabled_names: Optional[list[str]] = None
        selection_full = False
        if names_filter is None and self.monitoring_engine is not None:
            engine_config = getattr(self.monitoring_engine, "config", None)
            if engine_config is not None:
                try:
                    hooks = getattr(engine_config, "hooks", None)
                    if hooks is not None and hasattr(hooks, "is_full") and hooks.is_full():
                        selection_full = True
                    else:
                        compiled = None
                        if hasattr(self.monitoring_engine, "get_compiled_hook_names"):
                            compiled = self.monitoring_engine.get_compiled_hook_names(
                                self.hook_dict.keys(),
                                cache_key=id(self.hook_dict),
                            )
                        if compiled is not None:
                            precomputed_enabled_names, compiled_set = compiled
                            names_filter = compiled_set
                except Exception:
                    names_filter = None

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        elif isinstance(names_filter, set):
            filter_set = names_filter
            names_filter = lambda name: name in filter_set
        elif callable(names_filter):
            names_filter = names_filter
        else:
            raise ValueError("names_filter must be a string, list of strings, or function")
        assert callable(names_filter)  # Callable[[str], bool]

        self.is_caching = True

        engine = None
        if self.monitoring_engine is not None and not incl_bwd:
            engine = self.monitoring_engine

        native_backend = None
        native_using = False
        if engine is not None:
            native_backend = getattr(engine, "_native_backend", None)
            native_using = bool(getattr(engine, "_using_native_backend", False) and native_backend is not None)
        native_callback_active = native_using
        if engine is not None and not native_callback_active:
            raise RuntimeError(
                "MonitoringEngine native callback backend is required; Python fallback path was removed"
            )

        def save_hook(tensor: Tensor, hook: HookPoint, is_backward: bool = False):
            # for attention heads the pos dimension is the third from last
            if hook.name is None:
                raise RuntimeError("Hook should have been provided a name")

            hook_name = hook.name
            if is_backward:
                hook_name += "_grad"

            if engine is not None and not engine_capture_enabled:
                return

            range_label = f"TL::Cache[{hook_name}]"
            with _nvtx_range(range_label):
                global _hook_total_calls
                global _sync_build_us, _sync_move_us, _sync_remove_batch_us, _sync_slice_us, _sync_cache_set_us
                _hook_total_calls += 1 if _hook_stats_enabled() else 0

                resid_stream = tensor.detach() if tensor.requires_grad else tensor

                if (
                    hook.name.endswith("hook_q")
                    or hook.name.endswith("hook_k")
                    or hook.name.endswith("hook_v")
                    or hook.name.endswith("hook_z")
                    or hook.name.endswith("hook_result")
                ):
                    pos_dim = -3
                else:
                    # for all other components the pos dimension is the second from last
                    # including the attn scores where the dest token is the second from last
                    pos_dim = -2

                if engine is not None and not is_backward:
                    if native_callback_active:
                        cache[hook_name] = None
                        return
                    raise RuntimeError(
                        "MonitoringEngine native callback backend is required; Python fallback path was removed"
                    )

                # sync build（进入同步分支到移动前的轻量准备）
                t_sb0 = time.perf_counter() if _hook_stats_enabled() else None
                t_mv0 = time.perf_counter() if _hook_stats_enabled() else None
                resid_stream = resid_stream.to(device)
                if _hook_stats_enabled():
                    if t_sb0 is not None:
                        sb_us = (t_mv0 - t_sb0) * 1e6 if t_mv0 is not None else 0.0
                        _sync_build_us += sb_us
                        _per_hook_sync_build_us[hook_name] += sb_us
                    mv_us = (time.perf_counter() - t_mv0) * 1e6
                    _sync_move_us += mv_us
                    _per_hook_sync_move_us[hook_name] += mv_us
                if remove_batch_dim:
                    t_rm0 = time.perf_counter() if _hook_stats_enabled() else None
                    resid_stream = resid_stream[0]
                    if _hook_stats_enabled():
                        _sync_remove_batch_us += (time.perf_counter() - t_rm0) * 1e6

                if (
                    tensor.dim() >= -pos_dim
                ):  # check if the residual stream has a pos dimension before trying to slice
                    t_sl0 = time.perf_counter() if _hook_stats_enabled() else None
                    resid_stream = pos_slice.apply(resid_stream, dim=pos_dim)
                    if _hook_stats_enabled():
                        sl_us = (time.perf_counter() - t_sl0) * 1e6
                        _sync_slice_us += sl_us
                        _per_hook_sync_slice_us[hook_name] += sl_us
                if _hook_stats_enabled():
                    t_c0 = time.perf_counter()
                    cache[hook_name] = resid_stream
                    _sync_cache_set_us += (time.perf_counter() - t_c0) * 1e6
                else:
                    cache[hook_name] = resid_stream

        fwd_hooks = []
        bwd_hooks = []

        # Precompute enabled names and inform native backend (for global callbacks).
        # If enabled names are precomputed, skip per-step recomputation.
        enabled_names: list[str] = []
        if engine_capture_enabled and not selection_full:
            if precomputed_enabled_names is not None:
                enabled_names = precomputed_enabled_names
            else:
                with _nvtx_range("TL::ComputeEnabledNames"):
                    for _nm, _ in self.hook_dict.items():
                        if names_filter(_nm):
                            enabled_names.append(_nm)
        if native_callback_active and native_backend is not None and engine_capture_enabled:
            # Inform native backend of enabled names only when it changes.
            if selection_full:
                if self._native_enabled_hooks_key != -1:
                    try:
                        with _nvtx_range("MonEng::SetEnabledHooks"):
                            native_backend.set_enabled_hooks(None)
                        self._native_enabled_hooks_key = -1
                    except Exception:
                        pass
            else:
                enabled_key = id(enabled_names)
                if self._native_enabled_hooks_key != enabled_key:
                    try:
                        with _nvtx_range("MonEng::SetEnabledHooks"):
                            native_backend.set_enabled_hooks(enabled_names)
                        self._native_enabled_hooks_key = enabled_key
                    except Exception:
                        pass

        if native_callback_active and native_backend is not None:
            # Permanent registration path: register global callback once for all hook points
            if not self._native_callbacks_registered:
                try:
                    from monitoring.task import _encode_slice_native  # type: ignore
                except Exception:
                    _encode_slice_native = None  # type: ignore
                # Register for all known hook points once
                for reg_name, reg_hp in self.hook_dict.items():
                    slice_tuple = _encode_slice_native(pos_slice) if _encode_slice_native is not None else (0,)
                    with _nvtx_range(f"TL::BuildCallback[{reg_name}]"):
                        callback = native_backend.create_global_hook_callback_sig(
                            reg_name,
                            bool(remove_batch_dim),
                            slice_tuple,
                            device if device is not None else None,
                        )

                    def gated_callback(
                        tensor: Tensor,
                        *,
                        hook: HookPoint,
                        _cb=callback,
                        _root=self,
                    ):
                        try:
                            engine = getattr(_root, "monitoring_engine", None)
                            if engine is not None and not engine.is_capture_enabled():
                                return None
                        except Exception:
                            pass
                        return _cb(tensor)

                    # Register as permanent so ResetHooks won't remove them
                    reg_hp.add_hook(gated_callback, dir="fwd", is_permanent=True)
                self._native_callbacks_registered = True
            # No per-step hook registration needed in global native callback mode
            return cache, fwd_hooks, bwd_hooks

        # Build per-step hook definitions if not using global native callbacks
        hook_iter = precomputed_enabled_names if precomputed_enabled_names is not None else self.hook_dict.keys()
        for name in hook_iter:
            if precomputed_enabled_names is None and not names_filter(name):
                continue
            hp = self.hook_dict[name]
            with _nvtx_range("TL::CollectHookDefs"):
                fwd_hooks.append((name, partial(save_hook, is_backward=False)))
                if incl_bwd:
                    bwd_hooks.append((name, partial(save_hook, is_backward=True)))

        return cache, fwd_hooks, bwd_hooks

    def cache_all(
        self,
        cache: Optional[dict],
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
    ):
        logging.warning(
            "cache_all is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=lambda name: True,
            cache=cache,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )

    def cache_some(
        self,
        cache: Optional[dict],
        names: Callable[[str], bool],
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
    ):
        """Cache a list of hook provided by names, Boolean function on names"""
        logging.warning(
            "cache_some is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=names,
            cache=cache,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )


# %%
