"""Host-only telemetry of the legacy ring's existing waits and copies.

Every reader returns ``{"available": False, "reason": ...}`` instead of
inventing zeros when the native build or the adapter does not expose the
counters, and raises on a negative counter.  Reading does no CUDA work: the
native snapshot (``RingEngine.get_stats()``) holds host counters only.
"""
from __future__ import annotations

from typing import Any, Optional

# prepare_step outcomes and its host waits, and node-toggle pre-apply waits.
FIELDS = ('prepare_ok_calls', 'prepare_flushed_calls', 'prepare_oversized_calls',
          'prepare_sync_calls', 'prepare_sync_ns', 'prepare_sync_errors',
          'prepare_drain_calls', 'prepare_drain_ns', 'prepare_drain_errors',
          'toggle_wait_calls', 'toggle_wait_ns', 'toggle_wait_errors')

# Drain and P2P stages.
PIPELINE_FIELDS = ('staging_wait_calls', 'staging_wait_ns', 'staging_wait_errors',
                   'ready_commit_tasks', 'ready_commit_us', 'ready_commit_invalid',
                   'd2h_copy_errors', 'd2h_sync_errors',
                   'pageable_copy_calls', 'pageable_copy_ns', 'pageable_copy_bytes')

# Per-HookPoint row-table uploads an integration performs before a step
# (the integration's ``row_table_stats()`` mapping).
GATHER_FIELDS = ('hook_table_wait_calls', 'hook_table_wait_ns',
                 'hook_table_wait_errors', 'hook_table_uploads',
                 'hook_table_reuses')


def _native_counters(engine: Any, snapshot: Any, fields: tuple) -> dict:
    if engine is None and snapshot is None:
        return {'available': False, 'reason': 'no ring engine'}
    if snapshot is None:
        # A native build from before the snapshot API has no get_stats at
        # all: report it like missing fields.  Errors raised by an existing
        # get_stats propagate.
        getter = getattr(engine, 'get_stats', None)
        if not callable(getter):
            return {'available': False,
                    'reason': 'native telemetry unavailable (RingEngine has '
                              'no get_stats); rebuild required',
                    'missing_fields': list(fields)}
        stats = getter()
    else:
        stats = snapshot
    missing = [name for name in fields if not hasattr(stats, name)]
    if missing:
        return {'available': False,
                'reason': 'native telemetry unavailable; rebuild required',
                'missing_fields': missing}
    counters = {name: int(getattr(stats, name)) for name in fields}
    if any(value < 0 for value in counters.values()):
        raise ValueError('negative native counter')
    return {'available': True, 'schema_version': 1, 'cumulative': counters}


def synchronization_stats(engine: Any, *, snapshot: Optional[Any] = None) -> dict:
    """prepare_step outcomes, its stream synchronizations and forced flushes,
    and node-toggle pre-apply waits (host nanoseconds)."""
    out = _native_counters(engine, snapshot, FIELDS)
    if out['available']:
        out.update(clock='host_steady_clock_ns',
                   scope='existing host waits only; not device execution time')
    return out


def pipeline_stats(engine: Any, *, snapshot: Optional[Any] = None) -> dict:
    """Drain staging waits, per-task ready-to-commit latency (summed
    microseconds), D2H failures and staging-to-pageable copies."""
    out = _native_counters(engine, snapshot, PIPELINE_FIELDS)
    if out['available']:
        out.update(clock='host_steady_clock',
                   scope='host staging waits and memcpy ns; summed per-task '
                         'ready-to-commit us; not device time')
    return out


def gather_sync_stats(source: Any) -> dict:
    """An integration's per-HookPoint row-table upload counters: uploads,
    reuses of an unchanged table, and the host event waits before
    rewriting a pinned table.  ``source.row_table_stats()`` returns them as
    a mapping keyed by ``GATHER_FIELDS``.  Never queries a CUDA event."""
    read = getattr(source, 'row_table_stats', None)
    if not callable(read):
        return {'available': False,
                'reason': 'integration row-table telemetry unavailable',
                'missing_fields': list(GATHER_FIELDS)}
    stats = read()
    missing = [name for name in GATHER_FIELDS if name not in stats]
    if missing:
        return {'available': False,
                'reason': 'integration row-table telemetry unavailable',
                'missing_fields': missing}
    counters = {name: stats[name] for name in GATHER_FIELDS}
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError('invalid row-table synchronization counter')
    return {'available': True, 'schema_version': 1, 'cumulative': counters,
            'clock': 'host_perf_counter_ns',
            'scope': 'row-table reuse/uploads and existing host event waits; '
                     'not GPU copy time'}


__all__ = ['FIELDS', 'GATHER_FIELDS', 'PIPELINE_FIELDS', 'gather_sync_stats',
           'pipeline_stats', 'synchronization_stats']
