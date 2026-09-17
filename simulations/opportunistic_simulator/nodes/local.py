"""No-collaboration baseline."""

from __future__ import annotations

from opportunistic_simulator.data import ContactEvent, ContactAttempt
from .base import Node


class LocalOnlyNode(Node):
    """
    Baseline that observes contacts but never transfers usable knowledge.
    """
    def local_task(self) -> dict[str, float] | None:
        return self.local_train()

    def begin_contacts(self, nearby_events: list[ContactEvent], time: int)  -> list[str]:
        pass

    def perform_contacts(self, nodes: list[Node], time: int) -> list[ContactAttempt]:
        pass

    def end_contacts(self, time: int) -> None:
        pass
