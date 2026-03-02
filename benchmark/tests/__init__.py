"""Benchmark test implementations."""

__all__ = [
    "BatchInferenceBenchmark",
    "HookImpactBenchmark",
    "CacheComparisonBenchmark",
]


def __getattr__(name: str):
    if name == "BatchInferenceBenchmark":
        from .batch_inference import BatchInferenceBenchmark

        return BatchInferenceBenchmark
    if name == "HookImpactBenchmark":
        from .hook_impact import HookImpactBenchmark

        return HookImpactBenchmark
    if name == "CacheComparisonBenchmark":
        from .cache_comparison import CacheComparisonBenchmark

        return CacheComparisonBenchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
