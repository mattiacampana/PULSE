"""Indiscriminate full-head exchange baseline."""

from __future__ import annotations

from .base import Node
from opportunistic_simulator.data import ContactEvent, ContactAttempt
from opportunistic_simulator.utils import models as model_utils


class OppAVG(Node):
    """All-to-all averaging of prediction heads at each contact instant."""

    def local_task(self) -> dict[str, float] | None:
        """Perform the node's scheduled local optimization step."""
        return self.local_train()

    def begin_contacts(self, nearby_events: list[ContactEvent], time: int)  -> list[str]:
        """
        Snapshot the local classifier before the contacts at ``time``.

        The snapshot is used by all participants in the current contact
        instant, ensuring that aggregation is synchronous and independent
        of the order in which nodes execute ``perform_contacts``.

        Returns:
            Identifiers of the peers encountered by this node at ``time``.
        """
        self.last_model = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.classifier.state_dict().items()
        }

        self.nearby_events = nearby_events
        nodes_to_contact = list(nearby_events)

        if self.node_id in nodes_to_contact:
            raise ValueError(f"Self-contact detected for node {self.node_id}.")

        return nodes_to_contact


    def perform_contacts(self, nodes: list[Node], time: int) -> list[ContactAttempt]:
        """
        Fuse the local and nearby peers' pre-contact classifier states.

        Each classifier contributes equally to the aggregate. The local
        classifier is updated only after all states have been collected.

        Returns:
            One successful contact attempt for each peer included in the
            aggregation.
        """
        if not nodes:
            return []

        states = [self.last_model] + [node.last_model for node in nodes]

        averaged_state = model_utils.fuse_model_states(
            states,
            device=self.device,
        )

        self.classifier.load_state_dict(averaged_state, strict=True)

        return [
            ContactAttempt(
                peer_id=node.node_id,
                event=self.nearby_events[node.node_id],
            )
            for node in nodes
        ]

    def end_contacts(self, time: int) -> None:
        pass