from __future__ import annotations

"""Hook Points.

Helpers to access activations in models.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn
import torch.utils.hooks as hooks
from torch import Tensor


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
        self._name: Optional[str] = None

        # ring_producer_op dispatch keys -- set when self.name is assigned.
        # Stored as plain ints so torch.compile treats them as compile-time
        # constants and bakes them directly into the captured CUDA graph.
        self._ring_hook_type: int = 0
        self._ring_hook_id: int = 0

        # Master switch: False = completely bypass this hook (compiled out
        # under CUDA graphs).  Set once before generate() to select which
        # hooks are active.  Changing triggers Dynamo recompilation.
        self.enabled: bool = True

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        object.__setattr__(self, "_name", value)
        if value is not None:
            from .ring_transport import _hook_type_from_name, _hook_id_from_name
            object.__setattr__(self, "_ring_hook_type", _hook_type_from_name(value))
            object.__setattr__(self, "_ring_hook_id", _hook_id_from_name(value))

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

        NOTE: These are user-provided Python callback hooks, unrelated to ring
        transport.  They will NOT fire during CUDA graph replay (only the
        captured GPU kernels replay; Python hooks run only during eager or
        capture).  Ring transport data capture uses torch.ops.ring.producer
        called directly in HookPoint.forward(), not these hooks.
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
        if not self.enabled:
            return x
        if self._name is None or not x.is_cuda:
            return x

        x_cont = x.contiguous()

        # Eager safety-net path: when the active transport asks for it,
        # dispatch dynamically based on whether the hook's tensor fits.
        # CPU knows x_cont.nbytes exactly; no upper-bound math.
        from . import ring_transport as _rt
        transport = _rt._active_transport
        if transport is not None and transport.force_eager:
            engine = transport._ring_engine
            if engine is not None:
                nbytes = x_cont.nbytes
                if nbytes <= engine.available_capacity():
                    engine.reserve_one(nbytes)
                    torch.ops.ring.producer(
                        x_cont, self._ring_hook_type, self._ring_hook_id)
                elif nbytes <= engine.payload_cap():
                    engine.flush_and_wait()
                    engine.reserve_one(nbytes)
                    torch.ops.ring.producer(
                        x_cont, self._ring_hook_type, self._ring_hook_id)
                else:
                    # Single tensor larger than the whole ring -- bypass via
                    # cpu_direct.  Flush first so submit_cpu_direct consumes
                    # the FIFO meta for THIS hook (prior ring entries finish
                    # first; this hook's pre-pushed meta becomes the head).
                    engine.flush_and_wait()
                    transport.submit_cpu_direct(
                        x_cont.cpu(),
                        self._ring_hook_type, self._ring_hook_id)
                return x_cont

        # Fast path: torch.compile-serializable, CUDA-graph captureable.
        # C++ ring_producer_impl no-ops when g_active_engine is null
        # (monitoring inactive) and otherwise launches the producer kernel.
        torch.ops.ring.producer(
            x_cont, self._ring_hook_type, self._ring_hook_id)
        return x_cont

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
