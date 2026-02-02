import os
import threading
from typing import Any, Optional

import torch
from torch.utils.cpp_extension import load as load_extension

_OPS_MODULE: Optional[Any] = None
_OPS_LOCK = threading.Lock()


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
    return _OPS_MODULE
