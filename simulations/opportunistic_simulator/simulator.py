"""Single-timeline event-driven simulator."""

from __future__ import annotations

import time as wall_time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from .data import ContactEvent
from .metrics import classification_metrics

from opportunistic_simulator.nodes.base import Node
from opportunistic_simulator.utils.config import Config
from opportunistic_simulator.utils.logger import SimulationLogger


class Simulator:
    """
    Simulator class.
    """
    def __init__(
        self,
        nodes: dict[str, Node],
        contact_events: list[ContactEvent],
        config: Config,
        node_assignment: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> None:
        
        if config.training.interval <= 0 or config.evaluation.interval <= 0:
            raise ValueError("Training and evaluation intervals must be positive.")

        self.nodes = nodes
        self.config = config

        # time -> node -> nearby peer -> physical contact event
        # This structure allows for efficient lookup of contact events at each time step.
        self.nearby_events_by_time: dict[
            int,
            dict[str, dict[str, ContactEvent]],
        ] = defaultdict(lambda: defaultdict(dict))
        for event in contact_events:
            self.nearby_events_by_time[event.time][event.node_a][event.node_b] = event
            self.nearby_events_by_time[event.time][event.node_b][event.node_a] = event

        self.node_assignment = node_assignment or {}
        
        self.num_contact_events = len(contact_events)
        self.final_mean_metrics: dict[str, float] = {}

        # Logging setup
        self.output_dir = Path(config.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = SimulationLogger(output_dir=self.output_dir, seed=seed)

        for node in self.nodes.values():
            node.set_logger(self.logger)


    def run(self) -> None:
        """
        Use one timeline: periodic common training plus asynchronous contacts.
        """
        max_time = max(self.nearby_events_by_time, default=0)
        timeline = sorted(set(self.nearby_events_by_time) | set(range(0, max_time + 1, self.config.training.interval)) | set(range(0, max_time + 1, self.config.evaluation.interval)))
        started_at = wall_time.perf_counter()

        print(
            f"[simulation] t=0..{max_time} | {len(timeline):,} timeline instants | "
            f"{sum(len(events) for events in self.nearby_events_by_time.values()):,} contact events",
            flush=True,
        )

        # Main timeline loop: train, contact, evaluate, log progress.
        for time in timeline:

            losses: list[float] = []

            # Every training.interval steps, all nodes perform their local task.
            if time % self.config.training.interval == 0:

                # Local task ------------------------------------------------------------------------------
                for node in self.nodes.values():
                    result = node.local_task()

                    if result is not None:
                        self.logger.log("training", {
                            "time": time,
                            "node": node.node_id,
                            **result,
                        })
                        losses.append(float(result["loss"]))

            # Contacts -------------------------------------------------------------------------------------------
            # Perform contacts at this instant, if any.
            nearby_events_by_node = self.nearby_events_by_time.get(time, {})

            # First, notify every involved node of all currently available links.
            nodes_to_contact = defaultdict(list)
            for node_id, nearby_events in nearby_events_by_node.items():
                nodes_to_contact_ids = self.nodes[node_id].begin_contacts(nearby_events=nearby_events, time=time)
                if nodes_to_contact_ids != None:
                    nodes_to_contact[node_id] = [self.nodes[other_node_id] for other_node_id in nodes_to_contact_ids]
                else:
                    nodes_to_contact[node_id] = []

            for node_id in nearby_events_by_node.keys():
                attempts = self.nodes[node_id].perform_contacts(nodes=nodes_to_contact[node_id], time=time)

                if attempts:
                    for attempt in attempts:
                        if attempt.details is not None and not isinstance(attempt.details, Mapping):
                            raise TypeError(
                                f"{type(self.nodes[node_id]).__name__}.perform_contacts() returned "
                                f"invalid details for contact {node_id!r} -> {attempt.peer_id!r}: "
                                f"expected a mapping or None, got {type(attempt.details).__name__}."
                            )
                        self.logger.log("contacts", {
                            "time": time,
                            "requester_id": node_id,
                            "peer_id": attempt.peer_id,
                            "completed": attempt.completed,
                            "details": attempt.details
                        })

            for node_id in nearby_events_by_node.keys():
                self.nodes[node_id].end_contacts(time=time)

            # Evaluation -------------------------------------------------------------------------------------------
            # Regardless of whether any training or contacts occurred, evaluate all nodes at the scheduled interval.
            if time % self.config.evaluation.interval == 0 or time == max_time:

                rows: list[dict[str, float]] = []

                for node in self.nodes.values():
                    predictions, labels = node.evaluate()
                    node_metrics = classification_metrics(predictions, labels, self.config.model.num_classes)

                    self.logger.log("metrics", {
                        "time": time,
                        "node": node.node_id,
                        "split": "test",
                        **node_metrics,
                    })

                    rows.append(node_metrics)

                mean_metrics = {
                    key: sum(row[key] for row in rows) / len(rows)
                    for key in ("accuracy", "macro_f1")
                }

                self.final_mean_metrics = mean_metrics

            if (time == 0 or time == max_time or time % self.config.logging.progress_interval == 0):
                self._print_progress(
                    time=time,
                    max_time=max_time,
                    mean_loss=sum(losses) / len(losses) if losses else None,
                    metrics=mean_metrics,
                    started_at=started_at,
                )

        self.logger.close()

        final_text = (
            ", ".join(
            f"{key}={value:.4f}"
            for key, value in self.final_mean_metrics.items()
        ) or "no evaluation")

        print(
            f"[done] Saved results to {self.output_dir} | "
            f"final mean {final_text} | "
            f"elapsed {wall_time.perf_counter() - started_at:.1f}s",
            flush=True,
        )


    def _print_progress(
        self,
        time: int,
        max_time: int,
        mean_loss: float | None,
        metrics: dict[str, float] | None,
        started_at: float,
    ) -> None:
        parts = [f"[t={time}/{max_time}]"]

        if mean_loss is not None:
            parts.append(f"local train mean loss={mean_loss:.4f}")

        if metrics is not None:
            parts.append(
                "test mean "
                + ", ".join(
                    f"{name}={value:.4f}"
                    for name, value in metrics.items()
                )
            )

        if len(parts) == 1:
            parts.append("no scheduled training or evaluation")

        parts.append(
            f"elapsed={wall_time.perf_counter() - started_at:.1f}s"
        )
        print(" | ".join(parts), flush=True)
