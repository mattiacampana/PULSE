"""Opportunistic Federated Learning baseline (Lee et al., PerCom 2021)."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch

from opportunistic_simulator.data import ContactAttempt, ContactEvent, dataset_labels

from .base import Node


class OpportunisticFLNode(Node):
    """Perform encounter-driven, similarity-weighted gradient aggregation.

    At an encounter, a node asks each compatible peer to evaluate a gradient
    of *its current head* on one of the peer's private mini-batches.  The
    returned gradients are combined with a local gradient.  ``momentum``
    retains the most recent gradient received for every observed label set,
    whereas ``greedy`` uses only the currently encountered peer.
    """

    def __init__(
        self,
        *args: Any,
        similarity_threshold: float = 0.0,
        lambda_weight: float = 1.0,
        encounter_rounds: int = 1,
        aggregation: str = "momentum",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1].")
        if lambda_weight < 0.0:
            raise ValueError("lambda_weight must be non-negative.")
        if encounter_rounds <= 0:
            raise ValueError("encounter_rounds must be positive.")
        if aggregation not in {"greedy", "momentum"}:
            raise ValueError("aggregation must be either 'greedy' or 'momentum'.")

        self.similarity_threshold = similarity_threshold
        self.lambda_weight = lambda_weight
        self.encounter_rounds = encounter_rounds
        self.aggregation = aggregation
        self.label_distribution = self._label_distribution()
        # Adapt OppFL's goal profile to naturally heterogeneous per-user
        # data: G_i emphasises the activities under-represented in i's own
        # training set, without modifying the original data split.
        self.goal_distribution = self._minority_goal_distribution(
            self.label_distribution
        )
        self.gradient_memory: dict[
            tuple[int, ...], tuple[torch.Tensor, dict[str, torch.Tensor]]
        ] = {}
        self.nearby_events: Mapping[str, ContactEvent] = {}

    def local_task(self) -> dict[str, float] | None:
        """
        OpportunisticFL does not perform continuous local training.
        """
        return None

    def begin_contacts(
        self,
        nearby_events: Mapping[str, ContactEvent],
        time: int,
    ) -> list[str]:
        """Record the peers currently reachable by this node.

        No head parameters are exchanged in OppFL: the encounter protocol
        requests gradients computed on the peer's private data.  Keeping the
        contact map nevertheless lets ``perform_contacts`` return an accurate
        ``ContactAttempt`` for each accepted peer.
        """
        self.nearby_events = nearby_events
        if self.node_id in nearby_events:
            raise ValueError(f"Self-contact detected for node {self.node_id}.")
        if not nearby_events:
            return []

        # OppFL processes every neighbour available in the current contact
        # window.  The deterministic order makes simultaneous encounters
        # reproducible without arbitrarily discarding possible contacts.
        return sorted(nearby_events)

    def perform_contacts(
        self,
        nodes: list[Node],
        time: int,
    ) -> list[ContactAttempt]:
        attempts: list[ContactAttempt] = []
        local_losses: list[float] = []
        initial_sgd_steps = self.sgd_steps

        for peer in sorted(nodes, key=lambda node: node.node_id):
            if not isinstance(peer, OpportunisticFLNode):
                continue
            if peer.node_id not in self.nearby_events:
                continue

            goal_similarity = self._similarity(
                self.goal_distribution,
                peer.label_distribution,
            )

            if goal_similarity <= self.similarity_threshold:
                attempts.append(
                    ContactAttempt(
                        peer_id=peer.node_id,
                        event=self.nearby_events[peer.node_id],
                        completed=False,
                        details={"similarity": goal_similarity},
                    )
                )
                continue

            for _ in range(self.encounter_rounds):
                state = self._classifier_state()

                remote_gradient, _ = peer._gradient_on_private_batch(state)
                local_gradient, local_loss = self._gradient_on_private_batch(state)

                peer_key = tuple(
                    torch.nonzero(
                        peer.label_distribution > 0,
                        as_tuple=False,
                    ).flatten().tolist()
                )
                self.gradient_memory[peer_key] = (
                    peer.label_distribution.detach().cpu().clone(),
                    remote_gradient,
                )

                self._apply_aggregated_gradient(
                    local_gradient=local_gradient,
                    peer_distribution=peer.label_distribution,
                    peer_gradient=remote_gradient,
                )
                self.sgd_steps += 1
                local_losses.append(local_loss)

            attempts.append(
                ContactAttempt(
                    peer_id=peer.node_id,
                    event=self.nearby_events[peer.node_id],
                    completed=True,
                    details={
                        "similarity": goal_similarity,
                        "sgd_steps": self.encounter_rounds,
                    },
                )
            )

        executed_steps = self.sgd_steps - initial_sgd_steps
        if executed_steps > 0:
            self.logger.log(
                "training",
                {
                    "time": time,
                    "node": self.node_id,
                    "training_mode": "contact",
                    "loss": sum(local_losses) / len(local_losses),
                    "sgd_steps": executed_steps,
                    "total_sgd_steps": self.sgd_steps,
                    "accepted_peers": sum(attempt.completed for attempt in attempts),
                },
            )

        return attempts

    def end_contacts(self, time: int) -> None:
        """End the contact window; no deferred update is required."""
        self.nearby_events = {}

    def _label_distribution(self) -> torch.Tensor:
        """Return the empirical class distribution of the private data."""
        labels = dataset_labels(self.train_data)
        num_classes = int(self.classifier.output_layer.out_features)
        if len(labels) and int(labels.max()) >= num_classes:
            raise ValueError(
                f"Node {self.node_id} contains label {int(labels.max())}, but the "
                f"classifier exposes only {num_classes} classes."
            )
        return torch.bincount(labels, minlength=num_classes).float() / max(len(labels), 1)

    @staticmethod
    def _minority_goal_distribution(
        local_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """Build a goal distribution that favours locally scarce classes.

        Each class below the uniform share receives a mass proportional to its
        deficit.  Classes already above that share are not actively sought.
        A uniform local distribution has no deficits, so the uniform profile
        is used as the well-defined fallback.
        """
        num_classes = local_distribution.numel()
        if num_classes == 0:
            raise ValueError("A goal distribution requires at least one class.")
        uniform_share = 1.0 / num_classes
        deficits = (uniform_share - local_distribution.float()).clamp_min(0.0)
        total_deficit = deficits.sum()
        if total_deficit <= torch.finfo(deficits.dtype).eps:
            return torch.full_like(deficits, uniform_share)
        return deficits / total_deficit

    @staticmethod
    def _similarity(first: torch.Tensor, second: torch.Tensor) -> float:
        """Return Jensen--Shannon goal similarity in ``[0, 1]``.

        The paper measures distribution compatibility with Jensen--Shannon
        divergence.  We expose its complement as a similarity, so that the
        configured threshold retains the intuitive rule: communicate iff
        ``similarity >= similarity_threshold``.  Logarithms are base two,
        for which the divergence is bounded by one.
        """
        first = first.detach().float().cpu()
        second = second.detach().float().cpu()
        first = first / first.sum().clamp_min(torch.finfo(first.dtype).eps)
        second = second / second.sum().clamp_min(torch.finfo(second.dtype).eps)
        midpoint = 0.5 * (first + second)

        def _kl(distribution: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            mask = distribution > 0
            return torch.sum(
                distribution[mask]
                * torch.log2(distribution[mask] / reference[mask])
            )

        js_divergence = 0.5 * (_kl(first, midpoint) + _kl(second, midpoint))
        return float((1.0 - js_divergence).clamp(0.0, 1.0).item())

    def _weight(self, distribution: torch.Tensor) -> float:
        """Compute the exponentially decayed contribution weight."""
        similarity = self._similarity(self.goal_distribution, distribution)
        return math.exp(-self.lambda_weight * (1.0 - similarity))

    def _classifier_state(self) -> dict[str, torch.Tensor]:
        """Copy the head state so gradient evaluation cannot mutate it."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.classifier.state_dict().items()
        }

    def _gradient_on_private_batch(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], float]:
        temporary = copy.deepcopy(self.classifier).to(self.device)
        temporary.load_state_dict(
            {name: value.to(self.device) for name, value in state.items()},
            strict=True,
        )
        temporary.train()

        x, y = self._next_train_batch()
        with torch.no_grad():
            embeddings = self._encode(x)

        loss = self.loss_fn(temporary(embeddings), y)
        gradients = torch.autograd.grad(loss, tuple(temporary.parameters()))

        return (
            {
                name: gradient.detach().cpu().clone()
                for (name, _), gradient in zip(
                    temporary.named_parameters(), gradients
                )
            },
            float(loss.detach().cpu()),
        )

    def _apply_aggregated_gradient(
        self,
        *,
        local_gradient: Mapping[str, torch.Tensor],
        peer_distribution: torch.Tensor,
        peer_gradient: Mapping[str, torch.Tensor],
    ) -> None:
        """Apply one manual SGD step using the selected OppFL sources."""
        sources = [(self.label_distribution, local_gradient)]
        if self.aggregation == "greedy":
            sources.append((peer_distribution, peer_gradient))
        else:
            sources.extend(self.gradient_memory.values())

        normalizer = sum(self._weight(distribution) for distribution, _ in sources)
        with torch.no_grad():
            for name, parameter in self.classifier.named_parameters():
                aggregate = sum(
                    self._weight(distribution)
                    * gradient[name].to(self.device, dtype=parameter.dtype)
                    for distribution, gradient in sources
                ) / normalizer
                parameter.add_(aggregate, alpha=-self.config.learning_rate)
