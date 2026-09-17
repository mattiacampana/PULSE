"""OPTIMIST baseline: sequential independent-subnetwork training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from opportunistic_simulator.data import ContactAttempt, ContactEvent

from .base import Node


class OPTIMISTNode(Node):
    """Exchange an active subnetwork and train only the received subnetwork.

    This implementation targets an ``MLPClassifier`` with exactly one hidden
    linear layer.  All participating nodes must start from the same full-head
    initialisation: a subnetwork is identified by neuron indices, hence the
    indices have a shared meaning only under a common initial model.
    """

    def __init__(
        self,
        *args: Any,
        num_subnetworks: int,
        subnet_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        hidden_layers = [
            layer
            for layer in self.classifier.feature_layers
            if isinstance(layer, torch.nn.Linear)
        ]
        if len(hidden_layers) != 1:
            raise ValueError("OPTIMIST requires a classifier with one hidden linear layer.")
        if not 0 <= subnet_id < num_subnetworks:
            raise ValueError("subnet_id must be in [0, num_subnetworks).")

        self.hidden_layer = hidden_layers[0]
        self.hidden_size = self.hidden_layer.out_features
        if not 0 < num_subnetworks <= self.hidden_size:
            raise ValueError("num_subnetworks must be in [1, hidden width].")

        self.num_subnetworks = num_subnetworks
        self.subnet_id = subnet_id
        self.active_indices = self._indices(subnet_id)
        self.nearby_events: Mapping[str, ContactEvent] = {}
        self._snapshot_subnet_id: int | None = None
        self._snapshot_subnet_state: dict[str, torch.Tensor] | None = None

    def local_task(self) -> None:
        """Do not train outside an accepted encounter.

        OPTIMIST trains a received subnetwork after a match; consequently a
        node with no accepted peer performs no scheduled local update.
        """
        return None

    def begin_contacts(
        self,
        nearby_events: Mapping[str, ContactEvent],
        time: int,
    ) -> list[str]:
        """Snapshot the active subnetwork before all contacts at ``time``."""
        self.nearby_events = nearby_events
        if self.node_id in nearby_events:
            raise ValueError(f"Self-contact detected for node {self.node_id}.")
        self._snapshot_subnet_id = self.subnet_id
        self._snapshot_subnet_state = self._subnet_state()
        return list(nearby_events)

    def perform_contacts(
        self,
        nodes: list[Node],
        time: int,
    ) -> list[ContactAttempt]:
        """Swap one mutually selected pre-contact subnetwork and train it.

        A node can process one subnetwork per contact instant.  The mutual
        deterministic choice prevents two simultaneous transfers from
        overwriting one another and makes the result independent of the order
        in which the simulator invokes the nodes.
        """
        peer = self._matched_peer(nodes)
        if peer is None or peer._snapshot_subnet_state is None:
            return []
        if peer._snapshot_subnet_id is None:
            return []
        if not self._load_subnet(
            peer._snapshot_subnet_id,
            peer._snapshot_subnet_state,
        ):
            return []

        training_result = self._train_active_subnet()

        self.logger.log(
            "training",
            {
                "time": time,
                "node": self.node_id,
                "training_mode": "contact",
                "loss": training_result["loss"],
                "sgd_steps": training_result["sgd_steps"],
                "total_sgd_steps": training_result["total_sgd_steps"],
                "peer_id": peer.node_id,
                "received_subnet_id": self.subnet_id,
                "active_neurons": int(self.active_indices.numel()),
            },
        )

        return [
            ContactAttempt(
                peer_id=peer.node_id,
                event=self.nearby_events[peer.node_id],
            )
        ]

    def end_contacts(self, time: int) -> None:
        """Discard the transient pre-contact snapshot."""
        self.nearby_events = {}
        self._snapshot_subnet_id = None
        self._snapshot_subnet_state = None

    def _matched_peer(self, nodes: list[Node]) -> "OPTIMISTNode | None":
        """Return the deterministic mutual partner, if one exists."""
        candidates = sorted(
            (node for node in nodes if isinstance(node, OPTIMISTNode)),
            key=lambda node: node.node_id,
        )
        if not candidates:
            return None
        peer = candidates[0]
        peer_candidates = sorted(peer.nearby_events)
        return peer if peer_candidates and peer_candidates[0] == self.node_id else None

    def _indices(self, subnet_id: int) -> torch.Tensor:
        """Return the hidden-neuron indices assigned to ``subnet_id``."""
        chunks = torch.tensor_split(
            torch.arange(self.hidden_size),
            self.num_subnetworks,
        )
        return chunks[subnet_id].to(self.device)

    def _subnet_state(self) -> dict[str, torch.Tensor]:
        """Serialize exactly the parameters belonging to the active subnet."""
        indices = self.active_indices
        state = {
            "hidden_weight": self.hidden_layer.weight[indices].detach().cpu().clone(),
            "hidden_bias": self.hidden_layer.bias[indices].detach().cpu().clone(),
            "output_weight": self.classifier.output_layer.weight[:, indices].detach().cpu().clone(),
        }
        for layer in self.classifier.feature_layers:
            if isinstance(layer, torch.nn.BatchNorm1d):
                state.update({
                    "bn_weight": layer.weight[indices].detach().cpu().clone(),
                    "bn_bias": layer.bias[indices].detach().cpu().clone(),
                    "bn_mean": layer.running_mean[indices].detach().cpu().clone(),
                    "bn_var": layer.running_var[indices].detach().cpu().clone(),
                })
        return state

    def _load_subnet(
        self,
        subnet_id: int,
        state: Mapping[str, torch.Tensor],
    ) -> bool:
        """Install a compatible received subnetwork and make it active."""
        indices = self._indices(subnet_id)
        required = {"hidden_weight", "hidden_bias", "output_weight"}
        if not required.issubset(state):
            return False
        if any(not isinstance(state[name], torch.Tensor) for name in required):
            return False
        if state["hidden_weight"].shape != self.hidden_layer.weight[indices].shape:
            return False
        if state["hidden_bias"].shape != self.hidden_layer.bias[indices].shape:
            return False
        if state["output_weight"].shape != self.classifier.output_layer.weight[:, indices].shape:
            return False

        batch_norm_layers = [
            layer
            for layer in self.classifier.feature_layers
            if isinstance(layer, torch.nn.BatchNorm1d)
        ]
        batch_norm_keys = {"bn_weight", "bn_bias", "bn_mean", "bn_var"}
        if batch_norm_layers and not batch_norm_keys.issubset(state):
            return False

        with torch.no_grad():
            self.hidden_layer.weight[indices] = state["hidden_weight"].to(self.device)
            self.hidden_layer.bias[indices] = state["hidden_bias"].to(self.device)
            self.classifier.output_layer.weight[:, indices] = state["output_weight"].to(self.device)
            for layer in batch_norm_layers:
                layer.weight[indices] = state["bn_weight"].to(self.device)
                layer.bias[indices] = state["bn_bias"].to(self.device)
                layer.running_mean[indices] = state["bn_mean"].to(self.device)
                layer.running_var[indices] = state["bn_var"].to(self.device)

        self.subnet_id = subnet_id
        self.active_indices = indices
        return True

    def _train_active_subnet(self) -> dict[str, float | int]:
        """Train the received active subnet and return update statistics."""
        self.classifier.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.config.local_steps):
            x, y = self._next_train_batch()

            with torch.no_grad():
                embeddings = self._encode(x)

            loss = self.loss_fn(self.classifier(embeddings), y)

            self.optimizer.zero_grad()
            loss.backward()
            self._mask_inactive_gradients()
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            steps += 1
            self.sgd_steps += 1

        return {
            "loss": total_loss / steps,
            "sgd_steps": steps,
            "total_sgd_steps": self.sgd_steps,
        }

    def _mask_inactive_gradients(self) -> None:
        """Zero gradients and BatchNorm-buffer updates outside active neurons."""
        indices = self.active_indices
        hidden_mask = torch.zeros_like(self.hidden_layer.weight.grad)
        hidden_mask[indices] = 1
        self.hidden_layer.weight.grad.mul_(hidden_mask)
        bias_mask = torch.zeros_like(self.hidden_layer.bias.grad)
        bias_mask[indices] = 1
        self.hidden_layer.bias.grad.mul_(bias_mask)
        output_mask = torch.zeros_like(self.classifier.output_layer.weight.grad)
        output_mask[:, indices] = 1
        self.classifier.output_layer.weight.grad.mul_(output_mask)
        if self.classifier.output_layer.bias.grad is not None:
            self.classifier.output_layer.bias.grad.zero_()
        for layer in self.classifier.feature_layers:
            if isinstance(layer, torch.nn.BatchNorm1d) and layer.weight.grad is not None:
                mask = torch.zeros_like(layer.weight.grad)
                mask[indices] = 1
                layer.weight.grad.mul_(mask)
                layer.bias.grad.mul_(mask)
