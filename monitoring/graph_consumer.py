from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence

from .graph_engine import GraphSlotResult


class GraphSlotConsumer:
    """Delay-aware aggregator for GraphSafeEngine slot metadata."""

    def __init__(self, *, delay_steps: int = 0) -> None:
        self._delay_steps = max(0, int(delay_steps))
        self._pending: Dict[int, List[GraphSlotResult]] = {}
        self._order: Deque[int] = deque()

    def consume_graph_slots(self, slots: Sequence[GraphSlotResult]) -> List[GraphSlotResult]:
        """Submit slot metadata for a step and return any ready results."""

        if not slots:
            return []
        step_id = slots[0].step_id
        for result in slots:
            if result.step_id != step_id:
                raise ValueError("consume_graph_slots expects a single step per call.")

        bucket = self._pending.setdefault(step_id, [])
        bucket.extend(slots)
        if not self._order or self._order[-1] != step_id:
            self._order.append(step_id)
        return self._pop_ready(force=False)

    def flush_ready(self) -> List[GraphSlotResult]:
        """Return all remaining results regardless of delay."""

        return self._pop_ready(force=True)

    def _pop_ready(self, *, force: bool) -> List[GraphSlotResult]:
        ready: List[GraphSlotResult] = []
        while self._order:
            if not force and len(self._order) <= self._delay_steps:
                break
            step_id = self._order.popleft()
            ready.extend(self._pending.pop(step_id, []))
        return ready
