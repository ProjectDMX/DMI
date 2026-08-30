from __future__ import annotations

from dataclasses import fields, replace

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import CaptureQuery, CaptureSelection


pytestmark = pytest.mark.cpu


# One alternate value per filter field. The completeness assertion below fails
# when a filter is added to CaptureQuery without being covered here, which is
# the case that would silently drop a field out of ``filter_hash``.
_FILTER_ALTERNATES = {
    "tenant_id": "tenant-z",
    "experiment_id": "experiment-z",
    "run_id": "run-z",
    "session_id": "session-z",
    "model_id": "model-z",
    "hook_names": ("hook-z",),
    "layer_numbers": (99,),
    "captured_after_ns": 1_700_000_000_000_000_000,
    "captured_before_ns": 1_800_000_000_000_000_000,
}
_NON_FILTER_FIELDS = {"cursor", "limit"}


def _query(**overrides) -> CaptureQuery:
    base = {
        "tenant_id": "tenant-a",
        "experiment_id": "experiment-a",
        "run_id": "run-a",
        "hook_names": ("hook-a",),
        "layer_numbers": (3,),
    }
    return CaptureQuery(**{**base, **overrides})


def test_filter_alternates_cover_every_filter_field():
    declared = {item.name for item in fields(CaptureQuery)}
    assert set(_FILTER_ALTERNATES) == declared - _NON_FILTER_FIELDS


def test_filter_hash_is_stable_across_pages():
    first = _query()
    second = _query(cursor="ZW5jb2RlZC1jdXJzb3I")

    assert first.filter_hash == second.filter_hash


def test_filter_hash_is_stable_across_page_sizes():
    assert _query(limit=10).filter_hash == _query(limit=5_000).filter_hash


@pytest.mark.parametrize("name,value", sorted(_FILTER_ALTERNATES.items()))
def test_filter_hash_changes_with_every_filter(name: str, value):
    assert _query().filter_hash != _query(**{name: value}).filter_hash


def test_query_hash_still_separates_pages():
    # Preserved for backward compatibility: query_hash covers the full request,
    # cursor and limit included. Cursor binding uses filter_hash instead.
    first = _query()
    second = _query(cursor="ZW5jb2RlZC1jdXJzb3I")

    assert first.query_hash != second.query_hash


def test_cursor_accepts_a_full_keyset_payload():
    cursor = "c" * 2048

    assert CaptureQuery(cursor=cursor).cursor == cursor


def test_cursor_rejects_payloads_beyond_its_limit():
    with pytest.raises(ValueError, match="cursor"):
        CaptureQuery(cursor="c" * 2049)


def test_cursor_rejects_empty_and_non_string_values():
    with pytest.raises(ValueError, match="cursor"):
        CaptureQuery(cursor="")
    with pytest.raises(ValueError, match="cursor"):
        CaptureQuery(cursor=b"bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name", ("tenant_id", "experiment_id", "run_id", "session_id", "model_id")
)
def test_filter_text_fields_keep_the_shared_limit(name: str):
    assert CaptureQuery(**{name: "x" * 512})

    with pytest.raises(ValueError, match=name):
        CaptureQuery(**{name: "x" * 513})


def test_selection_binds_to_the_filter_hash():
    descriptors = synthetic_descriptors(2)
    query = _query()

    selection = CaptureSelection.create(
        descriptors, catalog_watermark="w-1", filter_hash=query.filter_hash
    )

    assert selection.filter_hash == query.filter_hash


def test_selection_identity_separates_different_filters():
    descriptors = synthetic_descriptors(2)
    same_filters = tuple(
        CaptureSelection.create(
            descriptors, catalog_watermark="w-1", filter_hash=_query(**overrides).filter_hash
        )
        for overrides in ({}, {"cursor": "cGFnZS10d28"}, {"limit": 25})
    )
    other_filters = CaptureSelection.create(
        descriptors, catalog_watermark="w-1", filter_hash=_query(run_id="run-z").filter_hash
    )

    # Paging through one query yields one selection identity; changing a filter
    # yields a different one.
    assert len({item.selection_id for item in same_filters}) == 1
    assert other_filters.selection_id != same_filters[0].selection_id


def test_selection_identity_separates_watermarks():
    descriptors = synthetic_descriptors(2)
    filter_hash = _query().filter_hash

    first = CaptureSelection.create(
        descriptors, catalog_watermark="w-1", filter_hash=filter_hash
    )
    second = CaptureSelection.create(
        descriptors, catalog_watermark="w-2", filter_hash=filter_hash
    )

    assert first.selection_id != second.selection_id


def test_selection_carries_the_tenant_of_its_descriptors():
    descriptors = synthetic_descriptors(2)

    selection = CaptureSelection.create(
        descriptors, catalog_watermark="w-1", filter_hash=_query().filter_hash
    )

    assert selection.tenant_id == descriptors[0].metadata.tenant_id


def test_selection_identity_separates_tenants():
    descriptors = synthetic_descriptors(2)
    retenanted = tuple(
        replace(item, metadata=replace(item.metadata, tenant_id="tenant-z"))
        for item in descriptors
    )
    filter_hash = _query().filter_hash

    first = CaptureSelection.create(
        descriptors, catalog_watermark="w-1", filter_hash=filter_hash
    )
    second = CaptureSelection.create(
        retenanted, catalog_watermark="w-1", filter_hash=filter_hash
    )

    # Identity v3: the tenant participates in selection_id, so the same
    # capture ids under another tenant are a different selection.
    assert first.selection_id != second.selection_id


def test_selection_refuses_descriptors_spanning_tenants():
    first, second = synthetic_descriptors(2)
    foreign = replace(second, metadata=replace(second.metadata, tenant_id="tenant-z"))

    with pytest.raises(ValueError, match="spans multiple tenants"):
        CaptureSelection.create(
            (first, foreign), catalog_watermark="w-1", filter_hash=_query().filter_hash
        )


def test_selection_refuses_an_empty_descriptor_set():
    # No descriptors means no tenant to bind the lookup to.
    with pytest.raises(ValueError, match="at least one capture"):
        CaptureSelection.create(
            (), catalog_watermark="w-1", filter_hash=_query().filter_hash
        )


def test_selection_validates_its_tenant_like_any_text_field():
    with pytest.raises(ValueError, match="tenant_id"):
        CaptureSelection(
            selection_id="s" * 64,
            capture_ids=("capture-0",),
            catalog_watermark="w-1",
            filter_hash="f" * 64,
            tenant_id="",
        )
