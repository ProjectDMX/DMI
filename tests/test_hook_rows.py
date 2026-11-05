"""Request-row planning for the gather producer (dmi.hook_rows).

The reference planners (plan_hook_rows, plan_hook_groups, HookRowPlanner)
are checked against independent enumerations of the selected token
positions; PackedDemandPlanner is checked against the reference planners;
the helpers that feed the producer (full_row_plan, encode_row_table,
segment_array, configure_hook_gather, configure_hook_heads) against their
contracts.  Pure Python: no GPU or native extension.
"""
import random

import numpy as np
import pytest

from dmi.hook_rows import (
    EMPTY_HOOK_ROWS,
    GATHER_MAX_SEGS,
    HookRowPlanner,
    HookRows,
    PackedDemandPlanner,
    canonical_token_windows,
    compile_token_windows,
    configure_hook_gather,
    configure_hook_heads,
    encode_row_table,
    full_row_plan,
    plan_hook_groups,
    plan_hook_rows,
    row_signature,
    segment_array,
)

pytestmark = pytest.mark.cpu

H18, H20, M20 = (0, 18), (0, 20), (13, 20)


# ---------------------------------------------------------------------------
# Reference planners: cached membership never caches positions or rows


def test_cold_churn_does_not_construct_reusable_indices(monkeypatch):
    p = HookRowPlanner((H18, H20), {"A": [H18], "B": [H20]}, [])
    original = p._group
    calls = []
    def record(ids):
        calls.append(ids)
        return original(ids)
    monkeypatch.setattr(p, "_group", record)
    for ids in (("A", "B"), ("B", "A"), ("A", "B")):
        result = p.plan(ids, [(3, 4)] * 2, [0, 1], [0, 1], 2, 8)
        assert result[H18].source_offsets == (ids.index("A"),)
    assert not calls
    result = p.plan(("A", "B"), [(4, 5)] * 2, [0, 1], [0, 1], 2, 8)
    assert calls == [("A", "B")]
    assert result[H20].token_ranges == ((4, 5),)
    p.plan(("A", "B"), [(5, 6)] * 2, [0, 1], [0, 1], 2, 8)
    assert len(calls) == 1


def test_invalid_layout_does_not_publish_a_new_cache():
    p = HookRowPlanner((H18,), {"A": [H18]}, [])
    p.plan(["A"], [(0, 1)], [0], [0], 1, 8)
    with pytest.raises(ValueError):
        p.plan(["B", "A"], [(0, 1)] * 2, [0, 0], [0, 1], 2, 8)
    assert p._request_order == ("A",) and p.group_builds == 1
    result = p.plan(["A"], [(1, 2)], [0], [0], 1, 8)
    assert result[H18].token_ranges == ((1, 2),)


def test_same_membership_advancing_positions_and_changed_chunk_size():
    p = HookRowPlanner((H18, H20), {"A": [H18], "B": [H20]}, [])
    first = p.plan(["A", "B"], [(0, 63), (8, 9)], [0, 63], [0, 1], 64, 16)
    second = p.plan(["A", "B"], [(63, 64), (9, 10)], [0, 1], [0, 1], 2, 16)
    assert (p.group_builds, p.group_reuses) == (1, 1)
    assert second[H18].token_ranges == ((63, 64),)
    assert second[H20].segments == ((1, 1),)
    assert second[H20].token_ranges == ((9, 10),)
    assert first[H20].segments == ((63, 1),)  # previous queued metadata is immutable


def test_expired_prefill_window_does_not_leak_into_decode_on_cache_hit():
    windows = compile_token_windows({"A": {H18: ((61, 64),)}})
    p = HookRowPlanner((H18,), {"A": [H18]}, [], windows)
    first = p.plan(["A"], [(32, 63)], [0], [0], 31, 16)
    tail = p.plan(["A"], [(63, 64)], [0], [0], 1, 16)
    decode = p.plan(["A"], [(64, 65)], [0], [0], 1, 16)
    assert first[H18].segments == ((29, 2),)
    assert tail[H18].segments == ((0, 1),)
    assert not decode[H18].segments and decode[H18].row_count == 0
    assert p.group_reuses == 2


def test_reordering_arrival_and_snapshot_ownership():
    rows = {"A": [H18], "B": []}
    p = HookRowPlanner((H18, H20), rows, [H20])
    rows["A"].append(H20)
    for ids in (["A", "B"], ["B", "A"], ["new", "A", "B"], ["A"]):
        n = len(ids)
        plans = p.plan(ids, [(9, 10)] * n, list(range(n)), list(range(n)), n, 16)
        assert plans[H18].req_ids == ("A",)
        assert plans[H18].source_offsets == (ids.index("A"),)
        assert plans[H20].req_ids == (("new",) if "new" in ids else ())
    assert p.group_builds == 4


def test_cache_hit_still_honors_table_capacity_and_current_offsets():
    p = HookRowPlanner((H18, H20), {"A": [H18, H20], "C": [H18, H20]}, [])
    args = (["A", "B", "C"], [(9, 10)] * 3, [0, 1, 2], [0, 1, 2], 3)
    exact = p.plan(*args, 2)
    fallback = p.plan(*args, 1)
    assert exact[H18] is exact[H20]
    assert not exact[H18].capacity_fallback
    assert fallback[H18].capacity_fallback
    assert fallback[H18].segments == ((0, 3),)
    assert fallback[H18].offsets == (0, 2)
    assert fallback[H18].req_ids == ("A", "C")


def test_generated_layouts_against_independent_position_enumeration():
    rng = random.Random(20260912)
    keys = (H18, H20, M20)
    ids = [f"r{i}" for i in range(9)]
    for _ in range(30):
        selection = {rid: frozenset(k for k in keys if rng.randrange(2)) for rid in ids[:-1]}
        windows = {rid: {key: ((3, 5), (7, None)) for key in keys if rng.randrange(2)}
                   for rid in ids if rng.randrange(2)}
        compiled = compile_token_windows(windows)
        p = HookRowPlanner(keys, selection, None, compiled)
        order = list(ids)
        for step in range(8):
            if step % 3 == 0:
                rng.shuffle(order)
            lengths = [rng.randrange(1, 5) for rid in order]
            offsets, total = [], 0
            for n in lengths:
                offsets.append(total)
                total += n
            ranges = [(step, step + n) for n in lengths]
            pos = list(range(len(order)))
            actual = p.plan(order, ranges, offsets, pos, total, 64)
            assert actual == plan_hook_groups(keys, order, ranges, offsets, pos,
                                             selection, None, total, 64, compiled)
            for key in keys:
                expected = []
                for rid, (lo, hi), off in zip(order, ranges, offsets):
                    if key not in selection.get(rid, keys):
                        continue
                    spans = windows.get(rid, {}).get(key)
                    expected.extend((rid, t, off + t - lo) for t in range(lo, hi)
                                    if spans is None or any(a <= t and (b is None or t < b)
                                                            for a, b in spans))
                rows = actual[key]
                got = [(rid, t, off + t - lo) for rid, (lo, hi), off
                       in zip(rows.req_ids, rows.token_ranges, rows.source_offsets)
                       for t in range(lo, hi)]
                assert got == expected


# ---------------------------------------------------------------------------
# Token windows


def test_union_is_canonical_and_preserves_empty():
    assert canonical_token_windows([(8, 11), (3, 7), (6, 8), (20, None)]) == ((3, 11), (20, None))
    assert canonical_token_windows([(5, None), (7, 9), (0, 5)]) == ((0, None),)
    assert canonical_token_windows([]) == ()


def test_compiled_window_profiles_share_storage_and_own_snapshot():
    windows = {"A": {H18: ((61, 64),), H20: ((61, 64),)}}
    compiled = compile_token_windows(windows)
    assert compiled.by_hook[H18] is compiled.by_hook[H20]
    windows["A"][H18] = ()
    assert compiled.by_hook[H18]["A"] == ((61, 64),)
    with pytest.raises(TypeError):
        compiled.by_hook[H18]["A"] = ()


@pytest.mark.parametrize("start,length", [(0, 3), (3, 4), (509, 8), (960, 1),
                                          (1021, 9), (2048, 64)])
@pytest.mark.parametrize("unbounded", [False, True])
def test_long_window_list_seeks_current_chunk(start, length, unbounded):
    spans = [(i * 4, i * 4 + 1) for i in range(256)]
    if unbounded:
        spans.append((1100, None))
    windows = {"A": canonical_token_windows(spans)}
    rows = plan_hook_rows(["A"], [(start, start + length)], [0], [3], {"A"},
                          length, 128, windows)
    expected = [token for token in range(start, start + length)
                if any(lo <= token and (hi is None or token < hi) for lo, hi in spans)]
    actual = [t for lo, hi in rows.token_ranges for t in range(lo, hi)]
    assert actual == expected
    assert [row for off, n in rows.segments for row in range(off, off + n)] == [t - start for t in expected]
    # A later call may go backwards (e.g. a recomputed prefill chunk). A seek
    # optimization must not carry a monotone cursor into the next call.
    earlier = plan_hook_rows(["A"], [(0, 1)], [0], [3], {"A"}, 1, 128, windows)
    assert earlier.token_ranges == ((0, 1),)


@pytest.mark.parametrize("selected", [(), (0,), (127,), (0, 63, 127), tuple(range(128))])
def test_sparse_membership_keeps_packed_order_and_positions(selected):
    ids = tuple(f"r{i}" for i in range(128))
    ranges = tuple((i, i + 1) for i in range(128))
    offsets = tuple(range(128))
    selection = {rid: frozenset([H18]) if i in selected else frozenset()
                 for i, rid in enumerate(ids)}
    plans = plan_hook_groups((H18, H20), ids, ranges, offsets, offsets,
                             selection, frozenset(), 128, 128)
    assert plans[H18].req_ids == tuple(ids[i] for i in selected)
    assert plans[H18].token_ranges == tuple(ranges[i] for i in selected)
    assert plans[H18].source_offsets == selected
    assert plans[H20].row_count == 0


def test_seeded_window_oracle():
    rng = random.Random(20260912)
    for _ in range(250):
        ids = [f"r{i}" for i in range(rng.randint(1, 12))]
        rng.shuffle(ids)
        ranges, offsets, windows, expected, total = [], [], {}, [], 0
        selected = set(rng.sample(ids, rng.randint(0, len(ids))))
        for rid in ids:
            start, length = rng.randrange(12), rng.randint(1, 16)
            ranges.append((start, start + length))
            offsets.append(total)
            raw = [(rng.randrange(25), rng.randrange(25, 40)) for _ in range(rng.randrange(4))]
            windows[rid] = canonical_token_windows(raw)
            # Deliberately enumerate the original, un-reduced window union.
            expected.extend((rid, token, total + token - start)
                            for token in range(start, start + length)
                            if rid in selected and any(lo <= token < hi for lo, hi in raw))
            total += length
        rows = plan_hook_rows(ids, ranges, offsets, list(range(len(ids))),
                              selected, total, 256, windows)
        source_rows = [row for off, length in rows.segments for row in range(off, off + length)]
        got = [(rid, token, source_rows[off + token - start])
               for rid, (start, end), off in zip(rows.req_ids, rows.token_ranges, rows.offsets)
               for token in range(start, end)]
        assert got == expected


def test_group_planner_checks_layout_once(monkeypatch):
    import dmi.hook_rows as module
    checked = []
    original = module._checked_layout
    def check(*args):
        checked.append(1)
        return original(*args)
    monkeypatch.setattr(module, "_checked_layout", check)
    selection = {"A": frozenset([H18]), "B": frozenset([H20])}
    plans = plan_hook_groups([H18, H20], ["A", "B"], [(0, 64), (7, 8)],
                             [0, 64], [0, 1], selection, frozenset(), 65, 8)
    assert len(checked) == 1
    assert plans[H18].row_count == 64 and plans[H20].row_count == 1


@pytest.mark.parametrize("keys", [[], [H18, H20]])
def test_group_planner_rejects_invalid_even_if_no_selected_rows(keys):
    with pytest.raises(ValueError, match="overlapping"):
        plan_hook_groups(keys, ["A", "B"], [(0, 3), (0, 2)], [0, 2], [0, 1],
                         {}, frozenset(), 5, 8)
    with pytest.raises(ValueError, match="lengths"):
        plan_hook_groups(keys, ["A"], [], [], [], {}, frozenset(), 0, 8)


def test_grouped_window_oracle_with_dynamic_replacement():
    """Random profiles, holes, chunk positions and segment fallback.

    Enumerate source rows independently instead of calling the row planner
    whose assembly helper is shared with the grouped implementation.
    """
    rng = random.Random(20260913)
    keys = (H18, H20, (1, 18), (1, 20))
    for _ in range(300):
        ids = [f"r{i}" for i in range(rng.randint(1, 12))]
        rng.shuffle(ids)
        ranges, offsets, total = [], [], 0
        selection, windows = {}, {}
        default = rng.choice([None, frozenset(), frozenset([H18])])
        for rid in ids:
            total += rng.randrange(3)  # padding gaps must not be captured
            length, start = rng.randint(1, 64), rng.randrange(100)
            ranges.append((start, start + length))
            offsets.append(total)
            total += length
            if rng.randrange(3):
                selection[rid] = rng.choice([None, frozenset(rng.sample(keys, rng.randrange(5)))])
            windows[rid] = {}
            for key in keys:
                if rng.randrange(3):
                    windows[rid][key] = canonical_token_windows([
                        (rng.randrange(120), rng.choice([None, 180]))
                        for _ in range(rng.randrange(4))])
        maximum = rng.choice([1, 4, 128])
        plans = plan_hook_groups(keys, ids, ranges, offsets, list(range(len(ids))),
                                 selection, default, total, maximum,
                                 compile_token_windows(windows))
        for key, plan in plans.items():
            expected = []
            for pos, (rid, (start, end), off) in enumerate(zip(ids, ranges, offsets)):
                selected = selection.get(rid, default)
                intervals = windows[rid].get(key)
                if selected is None or key in selected:
                    expected.extend((rid, t, off + t - start, pos)
                                    for t in range(start, end)
                                    if intervals is None or any(lo <= t and (hi is None or t < hi)
                                                                for lo, hi in intervals))
            copied = [i for off, count in plan.segments for i in range(off, off + count)]
            actual = [(rid, t, copied[off + t - start], pos)
                      for rid, (start, end), off, pos in zip(plan.req_ids, plan.token_ranges,
                                                           plan.offsets, plan.orig_positions)
                      for t in range(start, end)]
            assert actual == expected
            assert plan.row_count == (total if plan.capacity_fallback else len(expected))



# ---------------------------------------------------------------------------
# PackedDemandPlanner matches the reference planner


def _validate(keys):
    def validate(hooks):
        if hooks is None:
            return None
        hooks = frozenset(tuple(h) for h in hooks)
        unknown = hooks - set(keys)
        if unknown:
            raise ValueError(f"unknown HookPoint {sorted(unknown)}")
        return hooks
    return validate


def _random_step(rng, ids):
    ranges, offsets, total = [], [], 0
    for _ in ids:
        total += rng.randrange(3)             # graph padding between requests
        length, start = rng.randint(1, 40), rng.randrange(80)
        ranges.append((start, start + length))
        offsets.append(total)
        total += length
    return ranges, offsets, total


def test_packed_planner_matches_reference_planner_on_random_demand():
    rng = random.Random(20261103)
    keys = (H18, H20, M20, (1, 18))
    for _ in range(200):
        ids = [f"r{i}" for i in range(rng.randint(1, 10))]
        rows, windows = {}, {}
        default = rng.choice([None, frozenset(), frozenset([H18])])
        for rid in ids:
            if rng.randrange(3):
                rows[rid] = rng.choice(
                    [None, frozenset(rng.sample(keys, rng.randrange(len(keys) + 1)))])
            if rng.randrange(2):
                windows[rid] = {key: [(rng.randrange(60), rng.choice([None, 70, 120]))
                                      for _ in range(rng.randrange(3))]
                                for key in keys if rng.randrange(2)}
        canonical = {rid: {key: canonical_token_windows(spans)
                           for key, spans in per.items()}
                     for rid, per in windows.items()}
        compiled = compile_token_windows(canonical)
        planner = PackedDemandPlanner(keys, _validate(keys))
        for step in range(4):
            rng.shuffle(ids)
            ranges, offsets, total = _random_step(rng, ids)
            positions = list(range(len(ids)))
            maximum = rng.choice([1, 3, 256])
            got = planner.plan(ids, ranges, offsets, positions, total, maximum,
                               rows, windows, default)
            want = plan_hook_groups(keys, ids, ranges, offsets, positions,
                                    rows, default, total, maximum, compiled)
            for key in keys:
                assert got[key] == want[key], (step, key)
                assert got[key].row_count == want[key].row_count
            assert set(planner.present_keys) == {
                key for key in keys if got[key] is not EMPTY_HOOK_ROWS}


def test_packed_planner_compiles_rules_once_per_demand_change():
    keys = (H18, H20)
    planner = PackedDemandPlanner(keys, _validate(keys))
    rows = {"A": frozenset([H18]), "B": None}
    windows = {"A": {H18: [(2, None)]}}
    identity = object()
    for step in range(5):
        plans = planner.plan(["A", "B"], [(step, step + 4), (10 + step, 11 + step)],
                             [0, 4], [0, 1], 5, 8, rows, windows,
                             demand_identity=identity)
        assert plans[H20].req_ids == ("B",)
    assert planner.rule_compiles == 2          # one rule per request
    assert (planner.group_builds, planner.group_reuses) == (1, 4)
    rows = dict(rows, A=frozenset([H18, H20]))  # a demand change recompiles A
    planner.plan(["A", "B"], [(5, 9), (15, 16)], [0, 4], [0, 1], 5, 8, rows,
                 windows, demand_identity=object())
    assert planner.rule_compiles == 3


def test_packed_planner_shares_equal_plans_and_exposes_native_layouts():
    keys = (H18, H20, M20)
    planner = PackedDemandPlanner(keys, _validate(keys))
    plans = planner.plan(["A", "B"], [(0, 3), (7, 9)], [0, 3], [0, 1], 5, 8,
                         {"A": None, "B": frozenset([H18, H20])}, {})
    assert plans[H18] is plans[H20]
    assert plans[M20].req_ids == ("A",)
    assert plans[H18].layout() == [("A", 0, 3, 0, 0, 0), ("B", 7, 9, 3, 0, 1)]
    index, lo, hi, offset, pos = plans[M20].layout_arrays()
    assert index.tolist() == [0] and lo.tolist() == [0] and hi.tolist() == [3]
    assert offset.tolist() == [0] and pos.tolist() == [0]
    fallback = plans[H18].full_payload(5)
    assert fallback.segments == ((0, 5),) and fallback.offsets == (0, 3)


def test_packed_planner_validates_layout_and_hooks():
    keys = (H18,)
    planner = PackedDemandPlanner(keys, _validate(keys))
    with pytest.raises(ValueError, match="overlapping"):
        planner.plan(["A", "B"], [(0, 3), (0, 2)], [0, 2], [0, 1], 5, 8, {}, {})
    with pytest.raises(ValueError, match="lengths"):
        planner.plan(["A"], [], [], [], 0, 8, {}, {})
    with pytest.raises(ValueError, match="unknown HookPoint"):
        planner.plan(["A"], [(0, 1)], [0], [0], 1, 8, {"A": [H20]}, {})


# ---------------------------------------------------------------------------
# Static full-row baseline and the producer row table


def test_full_row_plan_selects_every_scheduled_row_with_identity_offsets():
    ids, ranges, offsets = ["A", "B"], [(5, 8), (0, 2)], [0, 4]   # row 3 is padding
    plan = full_row_plan(ids, ranges, offsets, [0, 1], 6)
    assert plan.segments == ((0, 6),)
    assert plan.row_count == 6
    assert plan.offsets == plan.source_offsets == (0, 4)
    assert plan.req_ids == ("A", "B") and plan.token_ranges == ((5, 8), (0, 2))
    # Same attribution as a reference plan selecting every request.
    reference = plan_hook_rows(ids, ranges, offsets, [0, 1], None, 6, 8)
    assert reference.req_ids == plan.req_ids
    assert reference.token_ranges == plan.token_ranges
    empty = full_row_plan([], [], [], [], 0)
    assert empty.segments == () and empty.row_count == 0


def _full_row_slices(plan, source):
    """Each request's rows as the consumer slices them from the payload the
    gather producer copies for ``plan``, next to the rows the ungathered
    producer's payload (the source itself) attributes to that request."""
    payload = [source[i] for start, count in plan.segments
               for i in range(start, start + count)]
    actual = [payload[off:off + hi - lo]
              for (lo, hi), off in zip(plan.token_ranges, plan.offsets)]
    expected = [source[off:off + hi - lo]
                for (lo, hi), off in zip(plan.token_ranges, plan.source_offsets)]
    return payload, actual, expected


@pytest.mark.parametrize("ids,ranges,offsets,total", [
    (["A", "B"], [(0, 2), (0, 1)], [2, 4], 8),          # leading + trailing padding
    (["A", "B"], [(0, 2), (3, 5)], [0, 5], 7),          # interior padding
    (["A", "B"], [(0, 2), (3, 5)], [0, 2], 6),          # trailing padding only
    (["B"], [(7, 9)], [3], 5),                          # filtered: first offset nonzero
    (["A", "B", "C"], [(0, 1), (2, 4), (9, 10)], [1, 3, 6], 7),
])
def test_full_row_plan_copies_the_ungathered_payload_with_physical_offsets(
        ids, ranges, offsets, total):
    plan = full_row_plan(ids, ranges, offsets, list(range(len(ids))), total)
    assert plan.segments == ((0, total),) and plan.row_count == total
    assert plan.offsets == plan.source_offsets == tuple(offsets)
    assert not plan.capacity_fallback
    payload, actual, expected = _full_row_slices(plan, list(range(100, 100 + total)))
    assert payload == list(range(100, 100 + total))     # the whole physical tensor
    assert actual == expected


def test_full_row_plan_without_requests_is_empty():
    for total in (0, 4):
        plan = full_row_plan([], [], [], [], total)
        assert plan.segments == () and plan.row_count == 0 and plan.offsets == ()
        assert encode_row_table(plan, 4).tolist()[0] == 0
    with pytest.raises(ValueError):
        full_row_plan(["A"], [(0, 2)], [3], [0], 4)        # rows past the tensor


def test_encode_row_table_matches_the_producer_contract():
    plan = plan_hook_rows(["A", "B"], [(0, 2), (4, 5)], [0, 3], [0, 1], None, 4, 8)
    table = encode_row_table(plan, 4)
    assert table.dtype == np.int64 and table.shape == (9,)
    assert table.tolist() == [2, 0, 2, 3, 1, 0, 0, 0, 0]
    assert encode_row_table(None, 2).tolist() == [-1, 0, 0, 0, 0]
    assert encode_row_table(EMPTY_HOOK_ROWS, 1).tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="more segments"):
        encode_row_table(plan, 1)


def test_row_signature_and_segment_array_agree_across_planners():
    ids, ranges, offsets, pos = ["A", "B", "C"], [(0, 2), (4, 6), (9, 10)], [0, 2, 4], [0, 1, 2]
    reference = plan_hook_rows(ids, ranges, offsets, pos, {"A", "C"}, 5, 8)
    planner = PackedDemandPlanner((H18,), _validate((H18,)))
    packed = planner.plan(ids, ranges, offsets, pos, 5, 8,
                          {"A": frozenset([H18]), "C": frozenset([H18])}, {})[H18]
    assert segment_array(reference).tolist() == [[0, 2], [4, 1]]
    assert segment_array(packed).tolist() == segment_array(reference).tolist()
    assert row_signature(packed) == row_signature(packed)
    other = plan_hook_rows(ids, ranges, offsets, pos, {"B"}, 5, 8)
    assert row_signature(other) != row_signature(reference)
    assert segment_array(EMPTY_HOOK_ROWS).shape == (0, 2)
    assert GATHER_MAX_SEGS == 256


def test_configure_hook_gather_and_heads_validate_their_inputs():
    import torch
    from types import SimpleNamespace

    hook = SimpleNamespace(_gather_table=None, _strip_row_bytes=0)
    table = torch.zeros(1 + 2 * 4, dtype=torch.int64)
    configure_hook_gather(hook, table, 64)
    assert hook._gather_table is table and hook._strip_row_bytes == 64
    with pytest.raises(ValueError, match="int64"):
        configure_hook_gather(hook, table.to(torch.int32), 64)
    with pytest.raises(ValueError, match="int64"):
        configure_hook_gather(hook, table.reshape(3, 3), 64)
    with pytest.raises(ValueError, match="positive row stride"):
        configure_hook_gather(hook, table, 0)
    configure_hook_gather(hook, None, 0)
    assert hook._gather_table is None

    arena = torch.zeros(80, dtype=torch.int32)
    acc = torch.zeros(4, 16)
    configure_hook_heads(hook, head_type=24, layer_no=3, table=table, arena=arena,
                         acc=acc, slots=2, weight_bf16=True)
    assert (hook._head_hook_type, hook._head_hook_id, hook._head_slots,
            hook._head_wbf16) == (24, 3, 2, 1)
    assert hook._head_arena is arena and hook._head_acc is acc
    with pytest.raises(ValueError, match="1..16"):
        configure_hook_heads(hook, head_type=24, layer_no=3, table=table,
                             arena=arena, acc=acc, slots=17, weight_bf16=False)
    with pytest.raises(ValueError, match="int32"):
        configure_hook_heads(hook, head_type=24, layer_no=3, table=table,
                             arena=arena.float(), acc=acc, slots=2, weight_bf16=False)
