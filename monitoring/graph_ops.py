import os
import threading
from typing import Any, Optional

import torch
from torch.utils.cpp_extension import load as load_extension

_OPS_MODULE: Optional[Any] = None
_OPS_LOCK = threading.Lock()
_LOWERING_REGISTERED = False


def _register_record_lowering() -> None:
    """Register custom Inductor lowering for record op.

    Adds the input tensor to ``V.graph.never_reuse_buffers`` so that
    Inductor's memory planner will not recycle its buffer after the last
    downstream consumer finishes.  This is the surgical fix for Path 2
    (FreeIfNotReused) buffer reuse — the ``Tensor(a!)`` schema annotation
    already blocks Path 1 (inplace reuse).
    """
    global _LOWERING_REGISTERED
    if _LOWERING_REGISTERED:
        return
    try:
        from torch._inductor.lowering import register_lowering
        from torch._inductor import ir
        from torch._inductor.virtualized import V
    except ImportError:
        return  # Inductor not available

    @register_lowering(
        torch.ops.graphmonitor_ops.record.default,
        type_promotion_kind=None,
    )
    def record_lowering(tensor, buffer, slot_id):
        tensor.realize()
        buffer.realize()
        # Prevent Inductor from recycling tensor's memory after record()
        # captures its data_ptr for later D2H copy.
        V.graph.never_reuse_buffers.add(tensor.data.get_name())
        ir.FallbackKernel.create(
            torch.ops.graphmonitor_ops.record.default,
            tensor,
            buffer,
            slot_id,
        )
        return ()

    _LOWERING_REGISTERED = True


def load_graph_monitor_ops(verbose: bool = False) -> Any:
    """JIT-compile and load the graph monitor CUDA ops."""
    global _OPS_MODULE
    if _OPS_MODULE is not None:
        return _OPS_MODULE

    with _OPS_LOCK:
        if _OPS_MODULE is not None:
            return _OPS_MODULE
        base_dir = os.path.dirname(__file__)
        source_path = os.path.join(base_dir, "csrc", "graph_monitor_ops.cu")
        build_dir = os.path.join(base_dir, ".torch_extensions")
        os.makedirs(build_dir, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        load_extension(
            name="graph_monitor_ops",
            sources=[source_path],
            build_directory=build_dir,
            verbose=verbose,
        )
        _OPS_MODULE = torch.ops.graphmonitor_ops
        _register_record_lowering()
    return _OPS_MODULE
