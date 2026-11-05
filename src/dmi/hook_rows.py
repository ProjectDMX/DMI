"""Per-HookPoint request-row plans for the gather producer.

A plan says which rows of a step's packed activation tensor one HookPoint
copies (``segments``: merged ``(source_row, rows)`` runs, the producer's row
table) and how the consumer attributes the packed result to requests
(``req_ids``, ``token_ranges``, destination ``offsets``, ``orig_positions``).
Plans are immutable and carry no GPU or framework dependency.

* :func:`plan_hook_rows` plans one HookPoint from an explicit request
  selection and optional absolute token windows.
* :func:`plan_hook_groups` and :class:`HookRowPlanner` plan many HookPoints
  at once, sharing one plan between HookPoints whose selections agree.
* :class:`PackedDemandPlanner` plans declared per-request demand with array
  operations over rules compiled once per demand change.
* :func:`full_row_plan` is the static full-row baseline: the whole physical
  payload as one run, physical offsets.

A plan whose segments exceed the producer's table capacity falls back to the
un-compacted full payload while keeping the consumer-side selection
(``capacity_fallback``).
"""
from bisect import bisect_right
from dataclasses import dataclass, replace
from itertools import chain as _chain, islice, repeat as _repeat
from operator import itemgetter
from types import MappingProxyType

import numpy as _np


# Row-table capacity of the gather producer (GATHER_MAX_SEGS in
# native/csrc/ring/producer.cuh): a table holds at most this many segments.
GATHER_MAX_SEGS = 256

_window_start = itemgetter(0)


def canonical_token_windows(windows):
    """Validate and union half-open absolute token intervals at admission.

    ``None`` as an interval's end means unbounded. An empty list is an empty
    selection, not the default (all positions). No sorting/validation is needed
    in the per-step planner after this conversion.
    """
    parsed = []
    if not isinstance(windows, (list, tuple)):
        raise ValueError("token windows must be a list of (start, end) pairs")
    for interval in windows:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("token windows must be (start, end) pairs")
        start, end = interval
        if (type(start) is not int or start < 0 or
                (end is not None and (type(end) is not int or end <= start))):
            raise ValueError("token windows require 0 <= start < end (or end=None)")
        parsed.append((start, end))
    merged = []
    for start, end in sorted(parsed, key=lambda pair: pair[0]):
        if merged and (merged[-1][1] is None or start <= merged[-1][1]):
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, None if previous_end is None or end is None
                          else max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class TokenWindowIndex:
    """Hook-indexed immutable filters, shared when their request maps agree."""
    by_hook: object
    long_hooks: frozenset = frozenset()


def compile_token_windows(windows):
    """Transpose and intern on selection changes, not on every serving step.

    Input intervals must already be canonical. Values are owned immutable maps;
    caller mutation cannot alter a snapshot while a forward is using it.
    """
    by_hook, long_hooks = {}, set()
    for rid, hooks in windows.items():
        for hook, spans in hooks.items():
            by_hook.setdefault(hook, {})[rid] = spans
            if len(spans) > 8:
                long_hooks.add(hook)
    interned = {}
    for hook, by_request in by_hook.items():
        signature = tuple(sorted(by_request.items()))
        if signature not in interned:
            interned[signature] = MappingProxyType(by_request)
        by_hook[hook] = interned[signature]
    return TokenWindowIndex(MappingProxyType(by_hook), frozenset(long_hooks))


@dataclass(frozen=True)
class HookRows:
    segments: tuple
    row_count: int
    req_ids: tuple
    token_ranges: tuple
    offsets: tuple
    source_offsets: tuple
    orig_positions: tuple
    capacity_fallback: bool = False

    def full_payload(self, total_rows):
        """Keep consumer selection, but describe an un-compacted fallback."""
        return replace(self, segments=((0, total_rows),), row_count=total_rows,
                       offsets=self.source_offsets)


def plan_hook_rows(req_ids, token_ranges, offsets, orig_positions, selected,
                   total_rows, max_segments, token_windows=None):
    """Select scheduled request spans or their declared token windows.

    `selected=None` means all supplied requests, not graph-padding rows.
    Metadata offsets address the compacted destination, never the source.
    On table overflow, capture full actual rows but preserve consumer filtering.
    ``token_windows`` maps request IDs to canonical absolute intervals. Adjacent
    source ranges can merge for copying even when they belong to different
    requests; metadata continues to identify each interval separately.
    """
    scheduled = _checked_layout(req_ids, token_ranges, offsets, orig_positions, total_rows)
    chosen = (row for row in scheduled if selected is None or row[0] in selected)
    long_windows = bool(token_windows) and any(len(w) > 8 for w in token_windows.values())
    return _plan_selected_rows(chosen, total_rows, max_segments, token_windows, long_windows)


def _checked_layout(req_ids, token_ranges, offsets, orig_positions, total_rows):
    """Validate all scheduled rows, including rows no monitor selects."""
    if not (len(req_ids) == len(token_ranges) == len(offsets) == len(orig_positions)):
        raise ValueError("request row metadata lengths differ")
    scheduled = tuple(zip(req_ids, token_ranges, offsets, orig_positions))
    previous_end = 0
    for _, tr, off, _ in scheduled:
        length = tr[1] - tr[0]
        if length <= 0 or off < previous_end or off + length > total_rows:
            raise ValueError("invalid or overlapping scheduled request rows")
        previous_end = off + length
    return scheduled


def _plan_selected_rows(chosen, total_rows, max_segments, token_windows, long_windows=False):
    """Build one profile from a checked layout; no repeated batch validation."""
    segs, ids, ranges, src, dst, positions = [], [], [], [], [], []
    count = 0
    for rid, tr, off, pos in chosen:
        windows = token_windows.get(rid) if token_windows else None
        if long_windows and windows and len(windows) > 8:
            if (windows[0][0] >= tr[1] or
                    (windows[-1][1] is not None and windows[-1][1] <= tr[0])):
                continue
            # Seek to the current chunk instead of rescanning all earlier
            # sampled positions on every prefill/decode step. No cursor is
            # retained, so reordering or replaying an earlier chunk is safe.
            first = max(0, bisect_right(windows, tr[0], key=_window_start) - 1)
            windows = islice(windows, first, None)
        for start, end in (tr,) if windows is None else windows:
            # Canonical windows are sorted and disjoint. In particular, a
            # prompt-only window must not leak into subsequent decode rows.
            if windows is not None:
                if start >= tr[1]:
                    break
                if start < tr[0]:
                    start = tr[0]
                if end is None or end > tr[1]:
                    end = tr[1]
                if end <= start:
                    continue
            source = off + start - tr[0]
            length = end - start
            if segs and segs[-1][0] + segs[-1][1] == source:
                segs[-1] = (segs[-1][0], segs[-1][1] + length)
            else:
                segs.append((source, length))
            ids.append(rid)
            ranges.append((start, end))
            src.append(source)
            dst.append(count)
            positions.append(pos)
            count += length
    plan = HookRows(tuple(segs), count, tuple(ids), tuple(ranges), tuple(dst),
                    tuple(src), tuple(positions))
    if len(segs) > max_segments:
        plan = replace(plan.full_payload(total_rows), capacity_fallback=True)
    return plan


def _sparse_rows(scheduled, bits):
    """Visit selected request indices in packed order without a full scan."""
    while bits:
        lowest = bits & -bits
        yield scheduled[lowest.bit_length() - 1]
        bits ^= lowest


class HookRowPlanner:
    """Own a selection snapshot and cache only its request-membership groups.

    Graph row indices and absolute positions are deliberately NOT cached.
    A cold request order uses one-step grouping; reusable indices are built
    only if that order repeats. Changed selections/windows install a new
    planner. Only one ordering is retained, bounding memory.
    The serving thread owns plan(), just as it owns the current batch layout.
    """

    def __init__(self, keys, selection, default=None, token_windows=None):
        self.keys = tuple(keys)
        self.selection = {rid: None if hooks is None else frozenset(hooks)
                          for rid, hooks in selection.items()}
        self.default = None if default is None else frozenset(default)
        self.windows = (token_windows if isinstance(token_windows, TokenWindowIndex)
                        else compile_token_windows(token_windows) if token_windows else None)
        self._request_order = None
        self._groups = None
        self.group_builds = 0
        self.group_reuses = 0

    def _group(self, req_ids):
        profiles = {}
        for i, rid in enumerate(req_ids):
            hooks = self.selection.get(rid, self.default)
            profiles[hooks] = profiles.get(hooks, 0) | (1 << i)
        masks = dict.fromkeys(self.keys, 0)
        for hooks, bits in profiles.items():
            for key in self.keys if hooks is None else hooks:
                if key in masks:
                    masks[key] |= bits
        filters = self.windows.by_hook if self.windows else {}
        long_hooks = self.windows.long_hooks if self.windows else frozenset()
        groups = {}
        all_rows = (1 << len(req_ids)) - 1
        for key, bits in masks.items():
            windows = filters.get(key)
            identity = (bits, id(windows)) if windows else bits
            if identity not in groups:
                # Compute dense and sparse membership just once, then visit
                # only selected rows on all subsequent steps in this batch.
                indices = (None if bits == all_rows else
                           tuple(_sparse_rows(tuple(range(len(req_ids))), bits)))
                groups[identity] = ([], indices, windows, key in long_hooks)
            groups[identity][0].append(key)
        return tuple((tuple(keys), indices, windows, long)
                     for keys, indices, windows, long in groups.values())

    def plan(self, req_ids, token_ranges, offsets, orig_positions, total_rows, max_segments):
        order = tuple(req_ids)
        if order != self._request_order:
            # A one-off layout needs no persistent membership-index cache.
            # Validate and finish planning before replacing the previous cache.
            plans = plan_hook_groups(self.keys, order, token_ranges, offsets,
                                     orig_positions, self.selection, self.default,
                                     total_rows, max_segments, self.windows)
            self._groups = None
            self._request_order = order
            self.group_builds += 1
            return plans
        scheduled = _checked_layout(order, token_ranges, offsets, orig_positions, total_rows)
        if self._groups is None:
            self._groups = self._group(order)
        self.group_reuses += 1
        plans = {}
        for keys, indices, windows, long_windows in self._groups:
            chosen = scheduled if indices is None else map(scheduled.__getitem__, indices)
            rows = _plan_selected_rows(chosen, total_rows, max_segments, windows, long_windows)
            plans.update(dict.fromkeys(keys, rows))
        return plans


def plan_hook_groups(keys, req_ids, token_ranges, offsets, orig_positions,
                     selection, default, total_rows, max_segments, token_windows=None):
    """Intern equal request selections within ONE batch layout.

    A bit identifies a scheduled request, not a token row. Requests sharing a
    profile are grouped first, then transposed to HookPoint request bitsets.
    Absolute token positions belong to this step and are never cached across
    steps, even when the physical row intervals are unchanged.
    """
    scheduled = _checked_layout(req_ids, token_ranges, offsets, orig_positions, total_rows)
    profiles = {}
    resolved = [selection.get(rid, default) for rid in req_ids]
    for i, hooks in enumerate(resolved):
        profiles[hooks] = profiles.get(hooks, 0) | (1 << i)
    masks = dict.fromkeys(keys, 0)
    for hooks, bits in profiles.items():
        for key in keys if hooks is None else hooks:
            if key in masks:
                masks[key] |= bits
    if token_windows and not isinstance(token_windows, TokenWindowIndex):
        token_windows = compile_token_windows(token_windows)
    filters = token_windows.by_hook if token_windows else {}
    long_hooks = token_windows.long_hooks if token_windows else frozenset()
    interned, plans = {}, {}
    all_rows = (1 << len(scheduled)) - 1
    for key, bits in masks.items():
        # Keep the whole-span fast path allocation-free. With position filters,
        # two hooks share a plan only when both membership and windows agree.
        windows = filters.get(key)
        group = (bits, id(windows)) if windows else bits
        if group not in interned:
            # Membership is already resolved above. Do not construct a set of
            # IDs, then scan and validate the same batch again for each group.
            if not bits:
                chosen = ()
            elif bits == all_rows:
                chosen = scheduled
            elif bits.bit_count() * 4 < len(scheduled):
                chosen = _sparse_rows(scheduled, bits)
            else:
                chosen = (row for row, hooks in zip(scheduled, resolved)
                          if hooks is None or key in hooks)
            interned[group] = _plan_selected_rows(chosen, total_rows, max_segments, windows, key in long_hooks)
        plans[key] = interned[group]
    return plans


# ---------------------------------------------------------------------------
# Packed (array) planner for declared monitor demand.
#
# A request's demand is fixed while its demand entry is unchanged, so its
# per-HookPoint row rule (HookPoint index + absolute window pieces) is
# compiled once and reused on every later step. Each step then derives the
# rows of every HookPoint for the whole batch with vectorised NumPy over the
# concatenated rules: clip to the scheduled token range, drop empty pieces,
# compact destination offsets and merge adjacent source runs. The plans
# expose exactly HookRows' fields, values and order; tuples are built only if
# a consumer reads them, and the native metadata rows of all HookPoints are
# built in one pass per step. HookRowPlanner/plan_hook_groups remain the
# reference planner for explicit selections.
# ---------------------------------------------------------------------------

_UNBOUNDED = 1 << 62
_EMPTY_I64 = _np.zeros(0, dtype=_np.int64)
EMPTY_HOOK_ROWS = HookRows((), 0, (), (), (), (), ())


def row_signature(rows):
    """Value identity of a plan's source segments.

    Two plans with equal signatures produce the same row table, so an
    integration can skip rewriting (and re-uploading) a table whose
    signature did not change."""
    sig = getattr(rows, "segment_signature", None)
    return rows.segments if sig is None else sig


def segment_array(rows):
    """A plan's ``(source row, rows)`` segments as an int64 ``[n, 2]`` array."""
    arr = getattr(rows, "segment_array", None)
    if arr is not None:
        return arr
    return _np.asarray(rows.segments, dtype=_np.int64).reshape(-1, 2)


_hook_signature = row_signature
_segment_array = segment_array


class _PackedStep:
    """One step's selected pieces for all HookPoints, sorted by HookPoint."""
    __slots__ = ("ids", "b", "lo", "hi", "dst", "src", "pos", "_rows")

    def __init__(self, ids, b, lo, hi, dst, src, pos):
        self.ids, self.b, self.lo, self.hi = ids, b, lo, hi
        self.dst, self.src, self.pos = dst, src, pos
        self._rows = None

    def rows(self):
        """Native metadata rows ``(rid, start, end, dst, 0, orig_position)``."""
        if self._rows is None:
            ids = self.ids
            self._rows = list(zip(map(ids.__getitem__, self.b.tolist()), self.lo.tolist(),
                                  self.hi.tolist(), self.dst.tolist(), _repeat(0), self.pos.tolist()))
        return self._rows


class PackedHookRows:
    """One HookPoint's rows for one step: a slice of a shared _PackedStep.

    Field semantics are exactly HookRows'. ``full`` (total rows) marks the
    un-compacted payload of a table-capacity or eager fallback: segments
    cover all rows, offsets become source offsets.
    """
    __slots__ = ("_step", "_r0", "_r1", "_seg", "row_count", "capacity_fallback", "_full",
                 "segment_signature", "_cache")

    def __init__(self, step, r0, r1, seg, row_count, capacity_fallback=False, full=None):
        self._step, self._r0, self._r1, self._seg = step, r0, r1, seg
        self.capacity_fallback = bool(capacity_fallback)
        self._full = None if full is None else int(full)
        self.row_count = int(row_count) if full is None else self._full
        self.segment_signature = self.segment_array.tobytes()
        self._cache = None

    @property
    def segment_array(self):
        if self._full is not None:
            return _np.array([[0, self._full]], dtype=_np.int64)
        return self._seg

    def _field(self, name):
        cache = self._cache
        if cache is None:
            cache = self._cache = {}
        value = cache.get(name)
        if value is None:
            s, r0, r1 = self._step, self._r0, self._r1
            if name == "segments":
                value = tuple(map(tuple, self.segment_array.tolist()))
            elif name == "req_ids":
                value = tuple(map(s.ids.__getitem__, s.b[r0:r1].tolist()))
            elif name == "token_ranges":
                value = tuple(zip(s.lo[r0:r1].tolist(), s.hi[r0:r1].tolist()))
            elif name == "offsets":
                value = tuple((s.src if self._full is not None else s.dst)[r0:r1].tolist())
            elif name == "source_offsets":
                value = tuple(s.src[r0:r1].tolist())
            elif name == "orig_positions":
                value = tuple(s.pos[r0:r1].tolist())
            elif name == "layout":
                if self._full is None:
                    value = s.rows()[r0:r1]
                else:
                    value = list(zip(self.req_ids, s.lo[r0:r1].tolist(), s.hi[r0:r1].tolist(),
                                     s.src[r0:r1].tolist(), _repeat(0), s.pos[r0:r1].tolist()))
            cache[name] = value
        return value

    segments = property(lambda self: self._field("segments"))
    req_ids = property(lambda self: self._field("req_ids"))
    token_ranges = property(lambda self: self._field("token_ranges"))
    offsets = property(lambda self: self._field("offsets"))
    source_offsets = property(lambda self: self._field("source_offsets"))
    orig_positions = property(lambda self: self._field("orig_positions"))

    def layout(self):
        """This HookPoint's native metadata rows (shared, do not mutate)."""
        return self._field("layout")

    def layout_arrays(self):
        """The same rows as int64 arrays for native ingestion:
        (index into the step's request list, start, end, offset, orig_position)."""
        s, r0, r1 = self._step, self._r0, self._r1
        return (s.b[r0:r1], s.lo[r0:r1], s.hi[r0:r1],
                (s.src if self._full is not None else s.dst)[r0:r1], s.pos[r0:r1])

    def full_payload(self, total_rows):
        """Keep consumer selection, but describe an un-compacted fallback."""
        return PackedHookRows(self._step, self._r0, self._r1, self._seg, 0,
                              self.capacity_fallback, total_rows)

    def as_hook_rows(self):
        return HookRows(self.segments, self.row_count, self.req_ids, self.token_ranges,
                        self.offsets, self.source_offsets, self.orig_positions,
                        self.capacity_fallback)

    def __eq__(self, other):
        if isinstance(other, PackedHookRows):
            other = other.as_hook_rows()
        if not isinstance(other, HookRows):
            return NotImplemented
        return self.as_hook_rows() == other

    def __hash__(self):
        return hash(self.as_hook_rows())

    def __repr__(self):
        return f"Packed{self.as_hook_rows()!r}"


def compile_request_rule(hooks, windows, key_index, validate_hook):
    """Admission-time rule of one request: sorted (HookPoint index, start, end).

    ``hooks`` are the request's selected HookPoints (None = every gather key);
    ``windows`` maps a HookPoint to raw intervals (absent = all positions,
    [] = none). Validation and canonicalization match the reference install;
    windows never enable an unselected HookPoint.
    """
    selected = validate_hook(hooks)
    if selected is None:
        selected = frozenset(key_index)
    parsed = {}
    for hook, intervals in (windows or {}).items():
        (parsed_hook,) = validate_hook([hook])
        parsed[parsed_hook] = canonical_token_windows(intervals)
    h, ws, we = [], [], []
    for hook in sorted(selected, key=key_index.__getitem__):
        k = key_index[hook]
        spans = parsed.get(hook)
        for start, end in ((0, None),) if spans is None else spans:
            h.append(k)
            ws.append(start)
            we.append(_UNBOUNDED if end is None else end)
    return (_np.asarray(h, dtype=_np.int64), _np.asarray(ws, dtype=_np.int64),
            _np.asarray(we, dtype=_np.int64))


class PackedDemandPlanner:
    """Per-step HookPoint row planning for declared (per-request) demand.

    Rules are cached per request by the identity of its demand entries, so a
    request is compiled once per demand change (admission, terminal position),
    not once per step. The concatenated, HookPoint-sorted batch rule is cached
    by (request order, demand object); per step only the position clip,
    compaction and segment merge run, as array operations.
    The serving thread owns plan(), just as it owns the current batch layout.
    """

    def __init__(self, keys, validate_hook):
        self.keys = tuple(keys)
        self.key_index = {key: i for i, key in enumerate(self.keys)}
        if len(self.keys) >= 1 << 15:
            raise ValueError("too many HookPoints for the packed planner")
        self._validate = validate_hook
        # rid -> (hooks entry, windows entry, pool start, piece count). Rules
        # live in append-only pools, compacted when mostly dead.
        self._rules = {}
        self._pool = _np.zeros((3, 1024), dtype=_np.int64)   # rows: HookPoint index, start, end
        self._pool_used = 0
        self._live = 0
        self._batch_key = None
        self._batch = None
        self.present_keys = []
        self.group_builds = 0
        self.group_reuses = 0
        self.rule_compiles = 0

    def _compile(self, rid, hooks, windows):
        h, ws, we = compile_request_rule(hooks, windows, self.key_index, self._validate)
        self.rule_compiles += 1
        n = len(h)
        if self._pool_used + n > self._pool.shape[1]:
            # Growing keeps every existing position valid.
            grown = _np.zeros((3, max(2 * self._pool.shape[1], self._pool_used + n)), dtype=_np.int64)
            grown[:, :self._pool_used] = self._pool[:, :self._pool_used]
            self._pool = grown
        start = self._pool_used
        self._pool[0, start:start + n] = h
        self._pool[1, start:start + n] = ws
        self._pool[2, start:start + n] = we
        self._pool_used = start + n
        old = self._rules.get(rid)
        self._live += n - (old[3] if old is not None else 0)
        entry = self._rules[rid] = (hooks, windows, start, n)
        return entry

    def _maintain(self, req_ids):
        """Between batches: drop rules of departed requests, compact the pool."""
        rules = self._rules
        if len(rules) > 1024 and len(rules) > 4 * len(req_ids):
            keep = {rid: rules[rid] for rid in req_ids if rid in rules}
            rules.clear()
            rules.update(keep)
            self._live = sum(entry[3] for entry in keep.values())
        if self._pool_used > 4096 and self._pool_used > 2 * self._live:
            old, used = self._pool, 0
            pool = _np.zeros((3, max(1024, 2 * self._live)), dtype=_np.int64)
            for rid, (hooks, windows, start, n) in list(rules.items()):
                pool[:, used:used + n] = old[:, start:start + n]
                rules[rid] = (hooks, windows, used, n)
                used += n
            self._pool, self._pool_used = pool, used

    def _build_batch(self, req_ids, rows, windows, default):
        self._maintain(req_ids)
        rules, compile_ = self._rules, self._compile
        get_rows, get_windows = rows.get, windows.get
        starts, counts = [], []
        for rid in req_ids:
            hooks, spans = get_rows(rid, default), get_windows(rid)
            entry = rules.get(rid)
            if entry is None or entry[0] is not hooks or entry[1] is not spans:
                entry = compile_(rid, hooks, spans)
            starts.append(entry[2])
            counts.append(entry[3])
        n = len(starts)
        count = _np.fromiter(counts, dtype=_np.int64, count=n)
        total = int(count.sum()) if n else 0
        if not total:
            return _EMPTY_I64, _EMPTY_I64, _EMPTY_I64, _EMPTY_I64
        first = _np.fromiter(starts, dtype=_np.int64, count=n)
        excl = _np.cumsum(count) - count
        index = _np.repeat(first - excl, count) + _np.arange(total, dtype=_np.int64)
        b = _np.repeat(_np.arange(n, dtype=_np.int64), count)
        pool_h, pool_ws, pool_we = self._pool
        # Stable (radix on int16): within a HookPoint, batch order then window order.
        order = _np.argsort(pool_h[index].astype(_np.int16), kind="stable")
        index = index[order]
        return pool_h[index], b[order], pool_ws[index], pool_we[index]

    def plan(self, req_ids, token_ranges, offsets, orig_positions, total_rows, max_segments,
             rows, windows, default=frozenset(), demand_identity=None):
        n = len(req_ids)
        if not (n == len(token_ranges) == len(offsets) == len(orig_positions)):
            raise ValueError("request row metadata lengths differ")
        ids = req_ids if isinstance(req_ids, tuple) else tuple(req_ids)
        if n:
            tr = _np.fromiter(_chain.from_iterable(token_ranges), dtype=_np.int64,
                              count=2 * n).reshape(n, 2)
            tr_lo, tr_hi = tr[:, 0], tr[:, 1]
            off = _np.fromiter(offsets, dtype=_np.int64, count=n)
            length = tr_hi - tr_lo
            prev_end = _np.empty(n, dtype=_np.int64)
            prev_end[0] = 0
            _np.add(off[:-1], length[:-1], out=prev_end[1:])
            if ((length <= 0) | (off < prev_end) | (off + length > total_rows)).any():
                raise ValueError("invalid or overlapping scheduled request rows")
        key = None if demand_identity is None else (ids, demand_identity)
        if key is not None and key == self._batch_key:
            self.group_reuses += 1
            h, b, ws, we = self._batch
        else:
            h, b, ws, we = self._build_batch(ids, rows, windows, default)
            self._batch_key, self._batch = key, (h, b, ws, we)
            self.group_builds += 1
        plans = dict.fromkeys(self.keys, EMPTY_HOOK_ROWS)
        # Keys given a non-empty plan this step (every other key maps to
        # EMPTY_HOOK_ROWS): callers derive the enabled set in O(present).
        self.present_keys = present = []
        if not len(h):
            return plans
        lo = _np.maximum(ws, tr_lo[b])
        hi = _np.minimum(we, tr_hi[b])
        keep = _np.flatnonzero(lo < hi)
        if not len(keep):
            return plans
        if len(keep) != len(h):
            h, b, lo, hi = h[keep], b[keep], lo[keep], hi[keep]
        size = hi - lo
        src = off[b] + (lo - tr_lo[b])
        pos = _np.fromiter(orig_positions, dtype=_np.int64, count=n)[b]
        m = len(h)
        excl = _np.cumsum(size)
        excl -= size
        hook_change = _np.empty(m, dtype=bool)
        hook_change[0] = True
        _np.not_equal(h[1:], h[:-1], out=hook_change[1:])
        first = _np.flatnonzero(hook_change)            # first piece of each present HookPoint
        bounds = _np.append(first, m)
        dst = excl - _np.repeat(excl[first], _np.diff(bounds))
        new_seg = hook_change.copy()
        new_seg[1:] |= src[1:] != (src[:-1] + size[:-1])
        seg_idx = _np.flatnonzero(new_seg)
        seg = _np.empty((len(seg_idx), 2), dtype=_np.int64)
        seg[:, 0] = src[seg_idx]
        seg[:, 1] = _np.add.reduceat(size, seg_idx)
        seg_bounds = _np.searchsorted(seg_idx, bounds).tolist()
        totals = _np.add.reduceat(size, first).tolist()
        step = _PackedStep(ids, b, lo, hi, dst, src, pos)
        bl = bounds.tolist()
        keys = self.keys
        coarse, interned, made = {}, {}, {}
        for j, k in enumerate(h[first].tolist()):
            r0, r1, s0, s1 = bl[j], bl[j + 1], seg_bounds[j], seg_bounds[j + 1]
            # Equal (request, interval) pieces imply equal offsets and segments:
            # such HookPoints share one plan. Compare contents only when the
            # piece/row/segment counts already agree.
            plan, shape = None, (r1 - r0, totals[j], s1 - s0)
            candidates = coarse.get(shape)
            if candidates is not None:
                sig = (b[r0:r1].tobytes(), lo[r0:r1].tobytes(), hi[r0:r1].tobytes())
                for other in candidates:
                    if other not in interned:
                        o0, o1 = bl[other], bl[other + 1]
                        interned[other] = (b[o0:o1].tobytes(), lo[o0:o1].tobytes(), hi[o0:o1].tobytes())
                    if interned[other] == sig:
                        plan = made[other]
                        break
            if plan is None:
                if s1 - s0 > max_segments:
                    plan = PackedHookRows(step, r0, r1, seg[s0:s1], 0, True, total_rows)
                else:
                    plan = PackedHookRows(step, r0, r1, seg[s0:s1], totals[j])
                coarse.setdefault(shape, []).append(j)
                made[j] = plan
            plans[keys[k]] = plan
            present.append(keys[k])
        return plans


def full_row_plan(req_ids, token_ranges, offsets, orig_positions, total_rows):
    """Static full-row baseline: the ungathered producer's payload.

    The plan copies the whole physical tensor, ``((0, total_rows),)``, so a
    gather producer fed this plan writes exactly what an ungathered producer
    writes -- including leading, interior and trailing padding rows -- and
    the destination offsets are the step's physical ``offsets``: each
    request's rows are ``payload[off : off + (hi - lo)]``, the same rows the
    ungathered consumer attributes.  Used to compare request-row gather
    against plain full-row capture without changing the consumer.

    A step with no scheduled request (or no rows) is the empty plan: no
    segments, no rows and nothing to attribute.
    """
    scheduled = _checked_layout(req_ids, token_ranges, offsets, orig_positions, total_rows)
    plan = _plan_selected_rows(scheduled, total_rows, max_segments=1 << 30,
                               token_windows=None)
    if not plan.segments:
        return plan
    return plan.full_payload(total_rows)


# HookPoint feature state armed by configure_hook_gather/configure_hook_heads.
# It belongs to one attachment: install_ring_hooks and uninstall_ring_hooks
# clear it (clear_hook_row_features), so a detached or re-attached HookPoint
# never runs a previous attachment's gather table or heads.
_GATHER_SAVED_STRIP_ROW_BYTES = "_gather_saved_strip_row_bytes"
_HEAD_DEFAULTS = (
    ("_head_hook_type", None),
    ("_head_hook_id", None),
    ("_head_table", None),
    ("_head_arena", None),
    ("_head_acc", None),
    ("_head_slots", 0),
    ("_head_wbf16", 1),
)


def clear_hook_gather(hook_point):
    """Disarm a HookPoint's gather producer and restore its strip stride.

    ``configure_hook_gather`` overwrites ``_strip_row_bytes`` with the gather
    row stride (a chunked strip keeps 0 there); the value it replaced is
    restored so the HookPoint returns to its own static/prefix/chunked mode.
    Releases the table reference.  Idempotent."""
    saved = getattr(hook_point, _GATHER_SAVED_STRIP_ROW_BYTES, None)
    if saved is not None:
        hook_point._strip_row_bytes = saved
        try:
            delattr(hook_point, _GATHER_SAVED_STRIP_ROW_BYTES)
        except AttributeError:  # pragma: no cover - exotic attribute store
            setattr(hook_point, _GATHER_SAVED_STRIP_ROW_BYTES, None)
    hook_point._gather_table = None


def clear_hook_heads(hook_point):
    """Disarm a HookPoint's in-graph heads, releasing the table, arena and
    accumulator references.  Idempotent."""
    for name, default in _HEAD_DEFAULTS:
        setattr(hook_point, name, default)


def clear_hook_row_features(hook_point):
    """Clear every gather/head feature of a HookPoint's attachment."""
    clear_hook_heads(hook_point)
    clear_hook_gather(hook_point)


def configure_hook_gather(hook_point, table, row_bytes):
    """Arm an installed HookPoint's request-row gather producer.

    ``table`` is the HookPoint's stable int64 row-table view (the layout
    above, ``1 + 2 * max_segments`` words) and ``row_bytes`` the stride of
    one packed row.  Configure before graph capture: a captured producer
    keeps the table's address and the stride.  ``table=None`` disarms the
    gather producer (``clear_hook_gather``).  The configuration lasts for
    the current attachment: ``install_ring_hooks`` and
    ``uninstall_ring_hooks`` clear it."""
    import torch
    if table is None:
        clear_hook_gather(hook_point)
        return
    if table.dtype != torch.int64 or table.dim() != 1 or not table.is_contiguous():
        raise ValueError("a row table is a contiguous 1-D int64 tensor")
    if table.numel() < 1:
        raise ValueError("a row table holds at least its segment count")
    if int(row_bytes) <= 0:
        raise ValueError("gather needs a positive row stride")
    if getattr(hook_point, _GATHER_SAVED_STRIP_ROW_BYTES, None) is None:
        setattr(hook_point, _GATHER_SAVED_STRIP_ROW_BYTES,
                int(getattr(hook_point, "_strip_row_bytes", 0)))
    hook_point._gather_table = table
    hook_point._strip_row_bytes = int(row_bytes)


def configure_hook_heads(hook_point, *, head_type, layer_no, table, arena, acc,
                         slots, weight_bf16):
    """Arm in-graph projection heads on an installed HookPoint.

    The heads score the rows selected by ``table`` under their own toggle key
    ``(head_type, layer_no)``.  ``arena`` is the node's int32 weight arena
    (``head_node_words`` words), ``acc`` the shared fp32 ``[rows, 16]``
    multi-layer accumulator, ``slots`` the head count ``K`` and
    ``weight_bf16`` the arena's weight dtype.

    Dispatch: with a gather table and a row spec the rows and the heads run
    as one fused node; otherwise the HookPoint's own row producer (static,
    prefix or chunked, when it has a row spec) launches first and a
    head-only node follows, so the published tasks always match the spec
    order (row, then head).  The configuration lasts for the current
    attachment (see ``configure_hook_gather``)."""
    import torch
    if not 0 < int(slots) <= 16:
        raise ValueError("in-graph heads take 1..16 slots")
    if arena.dtype != torch.int32 or acc.dtype != torch.float32:
        raise ValueError("head arenas are int32 words and accumulators fp32")
    if (table is None or table.dtype != torch.int64 or table.dim() != 1
            or not table.is_contiguous() or table.numel() < 1):
        raise ValueError("a head row table is a contiguous 1-D int64 tensor "
                         "holding at least its segment count")
    hook_point._head_hook_type = int(head_type)
    hook_point._head_hook_id = int(layer_no)
    hook_point._head_table = table
    hook_point._head_arena = arena
    hook_point._head_acc = acc
    hook_point._head_slots = int(slots)
    hook_point._head_wbf16 = int(bool(weight_bf16))


def encode_row_table(rows, max_segments):
    """The producer row table of one plan: ``[n, src0, len0, src1, len1,
    ...]`` padded to ``1 + 2 * max_segments`` int64 words.  ``rows=None``
    disables the producer for the step (``n = -1``)."""
    width = 1 + 2 * max_segments
    table = _np.zeros(width, dtype=_np.int64)
    if rows is None:
        table[0] = -1
        return table
    segments = _segment_array(rows)
    if len(segments) > max_segments:
        raise ValueError("plan has more segments than the row table holds")
    table[0] = len(segments)
    table[1:1 + 2 * len(segments)] = segments.reshape(-1)
    return table


__all__ = [
    "EMPTY_HOOK_ROWS",
    "GATHER_MAX_SEGS",
    "HookRowPlanner",
    "HookRows",
    "PackedDemandPlanner",
    "PackedHookRows",
    "TokenWindowIndex",
    "canonical_token_windows",
    "clear_hook_gather",
    "clear_hook_heads",
    "clear_hook_row_features",
    "compile_request_rule",
    "compile_token_windows",
    "configure_hook_gather",
    "configure_hook_heads",
    "encode_row_table",
    "full_row_plan",
    "plan_hook_groups",
    "plan_hook_rows",
    "row_signature",
    "segment_array",
]
