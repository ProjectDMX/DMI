from __future__ import annotations

"""DMI hook points.

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


from ..engine import MonitoringEngine
from .dispatch import dispatch_producer
from .specs import HOOK_TYPE_TO_SHORT_NAME, align_up_py

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


def _source_rows(x: Any) -> int:
    """Physical source rows of a contiguous HookPoint tensor: the product of
    every dimension before hidden (1 for a 1-D ``[hidden]`` tensor).  The
    native producers derive the same count as nbytes / (hidden x element
    size)."""
    shape = tuple(int(d) for d in x.shape)
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows


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

        # ring_producer_op dispatch keys -- set by install_ring_hooks()
        # from the spec's hook_type / layer_no.  Until install runs, both
        # are None, which makes HookPoint.forward() short-circuit (the
        # producer kernel never fires).  Stored as plain ints once set
        # so torch.compile treats them as compile-time constants and
        # bakes them directly into the captured CUDA graph.
        self._ring_hook_type: Optional[int] = None
        self._ring_hook_id: Optional[int] = None

        # Shared GPU payload-view tensor used as the Tensor(a!) mutation
        # alias for every producer op call from this HookPoint's
        # engine.  Pointing every HookPoint's `_ring_payload` at the
        # same physical memory creates a real R/W chain in the FX graph
        # that prevents inductor from reordering successive producer
        # launches.  Set by install_ring_hooks().
        self._ring_payload: Optional[torch.Tensor] = None

        # Master switch: False = completely bypass this hook (compiled out
        # under CUDA graphs).  Set once before generate() to select which
        # hooks are active.  Changing triggers Dynamo recompilation.
        self.enabled: bool = True

        # Strip-mode state.  Three modes signaled by these two attrs:
        #   _strip_tensor is None                    -> producer (static)
        #   _strip_tensor set, _strip_row_bytes > 0  -> producer_prefix
        #   _strip_tensor set, _strip_row_bytes == 0 -> producer_chunked
        #
        # For prefix mode: _strip_tensor is a 1-element int64 device
        # tensor holding the row count; kernel computes
        # bytes = row_count[0] * row_bytes.
        # For chunked mode: _strip_tensor is a K-element int64 device
        # tensor holding per-chunk byte counts; K = numel.
        #
        # CUDA-graph capture contract (caller's responsibility -- not
        # enforced at runtime):
        #   * _strip_tensor's data_ptr() and numel() must be stable
        #     across replays;
        #   * _strip_row_bytes must be fixed for any captured graph.
        # Both fixes are natural: they reflect structural properties
        # tied to the captured shape signature.  Per-step variation
        # lives in the tensor's *values*, which the kernel re-reads
        # each replay.
        #
        # Set ONCE at attach time + update VALUES in place each step
        # (via a separate kernel on the same stream as the producer).
        # Framework adapters must honor this contract.
        self._strip_tensor: Optional[torch.Tensor] = None
        self._strip_row_bytes: int = 0

        # Gather mode: when set, takes precedence over strip mode.  The
        # producer copies the row runs listed in this per-HookPoint row table
        # (dmi.hook_rows.encode_row_table layout), packed, re-reading the
        # VALUES each replay under the same stable-address contract as
        # _strip_tensor; _strip_row_bytes is the row stride.  A table whose
        # first word is -1 disables the producer for the step.
        # Armed by dmi.hook_rows.configure_hook_gather for one attachment:
        # install_ring_hooks / uninstall_ring_hooks clear it.
        self._gather_table: Optional[torch.Tensor] = None

        # In-graph projection heads.  Set by the integration
        # (dmi.hook_rows.configure_hook_heads, never by install_ring_hooks)
        # for one attachment; install_ring_hooks / uninstall_ring_hooks
        # clear them.  K fp32 scores per row selected by _head_table, under
        # their own spec and toggle key (_head_hook_type, _head_hook_id).
        # Dispatch (forward):
        #   row spec + gather table + heads -> ring::producer_fused, the rows
        #                                      and the heads in one node;
        #   row spec (static / prefix / chunked) + heads
        #                                   -> the row producer, then a
        #                                      head-only producer_fused node;
        #   heads only (no row spec)        -> the head-only node.
        # Either way the row task is published before the head task: the
        # spec (metadata) order.  _head_arena is the node's int32 weight
        # arena, _head_acc the shared fp32 [rows, 16] multi-layer
        # accumulator, and _head_wbf16 the arena's weight dtype (a launch
        # constant).
        self._head_hook_type: Optional[int] = None
        self._head_hook_id: Optional[int] = None
        self._head_table: Optional[torch.Tensor] = None
        self._head_arena: Optional[torch.Tensor] = None
        self._head_acc: Optional[torch.Tensor] = None
        self._head_slots: int = 0
        self._head_wbf16: int = 1

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
            ht = self._ring_hook_type
            if ht is not None:
                hook_label = f"{HOOK_TYPE_TO_SHORT_NAME.get(ht, ht)}@L{self._ring_hook_id}"
            else:
                hook_label = "uninstalled"
            with _nvtx_range(f"TL::Hook[{hook_label}:{dir}]"):
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

        # Capture-schedule gate: the adapter driver disarms the transport for
        # steps the schedule refused. Checking before the CUDA probe both
        # skips the work and keeps the refusal testable without a GPU.
        from ..transport import ring as _rt
        transport = _rt._active_transport
        if transport is not None and not transport.capture_step:
            return x

        row_on = self._ring_hook_type is not None
        head_on = self._head_hook_type is not None
        if not (row_on or head_on) or not x.is_cuda:
            return x

        x_cont = x.contiguous()

        # Eager safety-net path: when the active transport asks for it,
        # dispatch dynamically based on whether the hook's padded transport
        # size fits. v0 has no upper-bound reclaim, so stripped producers use
        # CPU-direct instead of reserving the full input and writing less.
        # Strip-mode dispatch.  Three branches signaled by attrs:
        #   _strip_tensor is None                    -> producer (static)
        #   _strip_tensor set, _strip_row_bytes > 0  -> producer_prefix
        #   _strip_tensor set, _strip_row_bytes == 0 -> producer_chunked
        # The branch is predictable per HookPoint (set once at attach;
        # never changes); torch.compile traces only the live branch.
        strip_t = self._strip_tensor
        strip_rb = self._strip_row_bytes
        gather_t = self._gather_table if row_on else None
        ring_payload = self._ring_payload

        from ..transport import ring as _rt
        transport = _rt._active_transport
        if (transport is not None and transport.force_eager
                and transport._ring_engine is not None):
            # The eager safety net publishes exactly what the fast path
            # publishes, in the same order: the row task (when this
            # HookPoint has a row spec), then the head task.
            engine = transport._ring_engine
            if row_on:
                self._eager_row(transport, engine, ring_payload, x_cont,
                                strip_t, strip_rb, gather_t)
            if head_on:
                self._eager_head(transport, engine, ring_payload, x_cont)
            return x_cont

        # Fast path: torch.compile-serializable, CUDA-graph captureable.
        # C++ ring_producer_* impls no-op when g_active_engine is null
        # (monitoring inactive); otherwise launch the producer kernel.
        if head_on and gather_t is not None:
            # One node for the rows and the heads; each part is gated by its
            # own table row.
            torch.ops.ring.producer_fused(
                ring_payload, x_cont, gather_t, strip_rb, self._head_table,
                self._head_arena, self._head_acc, self._head_slots,
                self._head_wbf16, self._ring_hook_type, self._ring_hook_id,
                self._head_hook_type)
            return x_cont
        if row_on:
            dispatch_producer(ring_payload, x_cont, strip_t, strip_rb,
                              self._ring_hook_type, self._ring_hook_id, gather_t)
        if head_on:
            self._launch_head_only(ring_payload, x_cont)
        return x_cont

    def _launch_head_only(self, ring_payload, x_cont):
        """The head part alone: a producer_fused node without a row table,
        recorded and toggled under the head key only, so the row producer
        and the heads keep independent toggle states."""
        torch.ops.ring.producer_fused(
            ring_payload, x_cont, None, 0, self._head_table,
            self._head_arena, self._head_acc, self._head_slots,
            self._head_wbf16, self._head_hook_type, self._head_hook_id,
            self._head_hook_type)

    def _eager_row(self, transport, engine, ring_payload, x_cont,
                   strip_t, strip_rb, gather_t):
        """Eager safety net for the row producer (never captured)."""
        # Node toggle: the eager safety net fires producers outside any
        # graph, where cudaGraphNodeSetEnabled cannot gate them.  On a
        # gated step that fell back to eager (a step larger than the
        # ring), a disabled hook must be skipped here, or it fires while
        # metadata and reservation counted only the enabled subset.
        # This branch is never captured or compiled, so a host check is
        # safe; toggle_gate_active keeps the stock eager path untouched.
        if (getattr(transport, "toggle_gate_active", False)
                and (getattr(transport, "_gated_step", True)
                     or getattr(transport, "_prefill_table_gate", False))
                and not engine.is_hook_enabled(
                    self._ring_hook_type, self._ring_hook_id)):
            return
        nbytes = x_cont.nbytes
        transport_bytes = align_up_py(nbytes, 16)
        # Gather mode reserves the full tensor here: on an eager step
        # the integration writes the full-row table, so the producer
        # copies exactly nbytes and the reservation matches.
        if strip_t is not None and gather_t is None:
            # CPU-direct bypasses ring/staging accounting. Flush older
            # ring tasks first so the shared metadata FIFO stays in
            # producer order. The legacy consumer applies its normal
            # CPU-side request slicing to the full tensor.
            engine.flush_and_wait()
            transport.submit_cpu_direct(
                x_cont.cpu(),
                self._ring_hook_type, self._ring_hook_id)
            return
        # The ring's real per-step ceiling is min(payload, staging),
        # not payload alone. The drain assembles each flush batch per
        # WHOLE entry and breaks when the entry does not fit staging
        # (drain_thread.cpp), so a tensor larger than staging that is
        # admitted here is never drained: the capture is lost and its
        # payload reservation is never released, permanently shrinking
        # ring capacity, while flush_and_wait() still returns success.
        # Same ceiling the native prepare_step/reserve_record use, and
        # the one RingCapacities.effective_bytes publishes. Both caps
        # are engine-lifetime constants queried over pybind, so the
        # min is cached on the transport and shared by every hook --
        # querying the pair per hook would repeat it for the whole
        # active set on the first eager forward.
        # A free task entry is needed too: a step with more hooks
        # than task entries lands here with the payload ring nearly
        # empty, and reserving past task_cap makes a producer
        # overwrite an unread slot. A flush frees both.
        effective_cap = transport.effective_cap
        if (transport_bytes <= min(engine.available_capacity(),
                                   effective_cap)
                and engine.available_task_slots() > 0):
            engine.reserve_one(nbytes)
            dispatch_producer(ring_payload, x_cont, strip_t, strip_rb,
                              self._ring_hook_type, self._ring_hook_id,
                              gather_t)
        elif transport_bytes <= effective_cap:
            engine.flush_and_wait()
            engine.reserve_one(nbytes)
            dispatch_producer(ring_payload, x_cont, strip_t, strip_rb,
                              self._ring_hook_type, self._ring_hook_id,
                              gather_t)
        else:
            # Larger than the ring's effective capacity (payload OR
            # staging) -- bypass via cpu_direct.  Flush first so submit_cpu_direct consumes
            # the FIFO meta for THIS hook (prior ring entries finish
            # first; this hook's pre-pushed meta becomes the head).
            engine.flush_and_wait()
            transport.submit_cpu_direct(
                x_cont.cpu(),
                self._ring_hook_type, self._ring_hook_id)

    def _eager_head(self, transport, engine, ring_payload, x_cont):
        """Eager safety net for the head part (never captured).

        The same gate and reservation order as the row producer: a disabled
        head reserves nothing; otherwise the integration wrote the full-row
        head table on eager steps, so the heads write rows x K x 4 bytes.

        ``rows`` is every physical source row: the product of all dimensions
        before hidden, as the native launcher derives it (nbytes / (hidden x
        element size) of the contiguous source) and as the step planner
        budgets the head spec's ``[..., K]`` shape.  A batched ``[B, S, H]``
        HookPoint therefore has B x S rows, not B.
        """
        if engine is None:
            return
        if (getattr(transport, "toggle_gate_active", False)
                and (getattr(transport, "_gated_step", True)
                     or getattr(transport, "_prefill_table_gate", False))
                and not engine.is_hook_enabled(self._head_hook_type,
                                               self._head_hook_id)):
            return
        rows = _source_rows(x_cont)
        nbytes = rows * int(self._head_slots) * 4
        transport_bytes = align_up_py(nbytes, 16)
        # As for the rows: an entry larger than min(payload, staging) is never
        # drained and would wedge the ring.  Heads have no CPU-direct path
        # (the scores exist only on the GPU), so refuse before reserving.
        if transport_bytes > transport.effective_cap:
            raise RuntimeError(
                f"in-graph head output ({transport_bytes} bytes for "
                f"{rows} rows x {int(self._head_slots)} slots) "
                f"exceeds the ring's effective capacity "
                f"({transport.effective_cap} bytes)")
        if (transport_bytes > engine.available_capacity()
                or engine.available_task_slots() == 0):
            engine.flush_and_wait()
        engine.reserve_one(nbytes)
        self._launch_head_only(ring_payload, x_cont)

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
