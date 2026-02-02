import pytest

from monitoring.graph_consumer import GraphSlotConsumer
from monitoring.graph_engine import GraphSlotResult


def make_slot(step_id: int, slot_id: int = 0) -> GraphSlotResult:
    return GraphSlotResult(
        slot_id=slot_id,
        name=f"layer_{slot_id}",
        step_id=step_id,
        data_ptr=slot_id + 1,
        shape=(2, 4),
        stride=(4, 1),
        ndim=2,
        dtype_id=5,
        device_index=0,
    )


def test_consumer_no_delay_releases_immediately():
    consumer = GraphSlotConsumer(delay_steps=0)
    slots = [make_slot(step_id=1, slot_id=0), make_slot(step_id=1, slot_id=1)]
    ready = consumer.consume_graph_slots(slots)
    assert ready == slots


def test_consumer_with_delay_requires_future_step():
    consumer = GraphSlotConsumer(delay_steps=1)
    slots_step1 = [make_slot(step_id=1, slot_id=0)]
    slots_step2 = [make_slot(step_id=2, slot_id=0)]

    ready = consumer.consume_graph_slots(slots_step1)
    assert ready == []

    ready = consumer.consume_graph_slots(slots_step2)
    assert ready == slots_step1

    remaining = consumer.flush_ready()
    assert remaining == slots_step2


def test_consumer_empty_submission_is_noop():
    consumer = GraphSlotConsumer(delay_steps=2)
    assert consumer.consume_graph_slots([]) == []


def test_consumer_raises_on_multiple_steps():
    consumer = GraphSlotConsumer()
    slots = [make_slot(step_id=1, slot_id=0), make_slot(step_id=2, slot_id=1)]
    with pytest.raises(ValueError):
        consumer.consume_graph_slots(slots)
