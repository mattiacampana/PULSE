"""PULSE: Personalized Utility-guided Learning via Selective Encounters.

This module implements the encounter-driven update described in the PULSE
paper.  Each :class:`PULSENode` owns a personalized prediction head ``h_i``
and uses a shared, frozen encoder ``E`` inherited from :class:`.base.Node`.
Consequently, the exchanged information and every gradient in this file refer
only to the prediction head, never to the encoder or to raw local samples.

At a peer-discovery event, node ``i`` selects either one reachable node ``j``
or the no-contact action ``None`` (the implementation counterpart of
``varnothing`` in the paper).  When a peer is selected, ``i`` sends a
temporary copy of its current head.  Peer ``j`` evaluates that copy on one
private minibatch and returns only the resulting gradient.  Node ``i`` then
computes its own gradient on the same head parameters, measures their cosine
agreement, norm-matches the remote gradient, and applies the mixing update.

Choosing no-contact never stops local learning.  It skips the message exchange
and applies the same mixing rule with ``alpha = 0``, which is exactly a local
gradient step.  This distinction is important: PULSE decides whether to
collaborate, not whether to learn.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch

from opportunistic_simulator.data import ContactAttempt, ContactEvent

from .base import Node


class PULSENode(Node):
    """PULSE node implementing utility-guided, head-only collaboration.

    The state maintained for each peer ``j`` directly mirrors the peer
    selection policy in the paper:

    * ``peer_utility[j]`` is :math:`\\mathcal{U}_i[j]`, the exponentially
      smoothed signed gradient agreement;
    * ``peer_visits[j]`` is :math:`n_{ij}`, the number of completed exchanges
      with that peer; and
    * ``total_selection_opportunities`` is :math:`s_i`, the number of events
      at which this node had at least one reachable peer and could choose
      between a peer and no-contact.

    These quantities are local to node ``i``: utility is neither transmitted
    nor assumed to be symmetric between two nodes.
    """
    def __init__(
        self,
        *args: Any,
        alpha_max: float = 0.5,
        utility_momentum: float = 0.2,
        initial_utility: float = 0.5,
        ucb_exploration: float = 0.25,
        norm_epsilon: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not 0.0 <= alpha_max <= 1.0:
            raise ValueError("alpha_max must be in [0, 1].")
        if not 0.0 < utility_momentum <= 1.0:
            raise ValueError("utility_momentum must be in (0, 1].")
        if not -1.0 <= initial_utility <= 1.0:
            raise ValueError("initial_utility must be in [-1, 1].")
        if ucb_exploration < 0.0:
            raise ValueError("ucb_exploration must be non-negative.")
        if norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be positive.")

        self.alpha_max = alpha_max
        self.utility_momentum = utility_momentum
        self.initial_utility = initial_utility
        self.ucb_exploration = ucb_exploration
        self.norm_epsilon = norm_epsilon

        # Per-peer state of Eq. 8; entries are created lazily.
        self.peer_utility: dict[str, float] = {}
        self.peer_visits: dict[str, int] = {}
        # s_i: incremented once per actual selection opportunity, including
        # selections of no-contact, but never when no peer is reachable.
        self.total_selection_opportunities = 0
        self.nearby_events: Mapping[str, ContactEvent] = {}


    def local_task(self) -> dict[str, float] | None:
        """Run the simulator's periodic local-training task.

        This is the independent ``R``-step local-training mechanism.  It is
        separate from the single update performed in response to a discovery
        event by :meth:`perform_contacts`, including the no-contact case.
        """
        return self.local_train()


    def begin_contacts(
        self,
        nearby_events: Mapping[str, ContactEvent],
        time: int,
    ) -> list[str]:
        """Store the reachable set :math:`\\mathcal{N}_i(t)` for this event.

        The simulator calls this method before :meth:`perform_contacts` and
        supplies the corresponding contact metadata.  PULSE subsequently
        selects at most one member of this set, or no-contact.
        """
        self.nearby_events = nearby_events
        if self.node_id in nearby_events:
            raise ValueError(f"Self-contact detected for node {self.node_id}.")
        
        return list(nearby_events)


    def perform_contacts(
        self,
        nodes: list[Node],
        time: int,
    ) -> list[ContactAttempt]:
        """Perform PULSE's one-step update for the current discovery event.

        If at least one PULSE peer is reachable, this method evaluates the UCB
        policy over those peers and the implicit no-contact action.  A selected
        peer provides a remote gradient, after which the local and remote
        gradients are agreement-calibrated and mixed.  If no-contact wins, no
        message is sent and the same update reduces to a local step
        (``alpha = 0``).

        Returns:
            A singleton list describing the initiated exchange, or an empty
            list when no peer was reachable or no-contact was selected.
        """
        candidates = [
            node
            for node in nodes
            if isinstance(node, PULSENode) and node.node_id in self.nearby_events
        ]
        if not candidates:
            return []

        peer = self._select_peer(candidates)

        # Increment s_i once for this decision, irrespective of its outcome.
        self.total_selection_opportunities += 1

        if peer is None:
            state = self._classifier_state()
            local_gradient = self._gradient_on_private_batch(state)

            self._apply_agreement_calibrated_gradient(
                local_gradient=local_gradient,
                remote_gradient=local_gradient,  # Has zero weight because alpha = 0.
                alpha=0.0,
            )
            self.sgd_steps += 1
            return []

        state = self._classifier_state()

        # SEND h_i to j; RECEIVE grad ell_j(h_i) on j's private minibatch.
        remote_gradient = peer._gradient_on_private_batch(state)
        local_gradient = self._gradient_on_private_batch(state)

        agreement = self._cosine_agreement(local_gradient, remote_gradient)
        utility_before = self.peer_utility.get(
            peer.node_id,
            self.initial_utility,
        )

        alpha = (
            self.alpha_max
            * max(utility_before, 0.0)
            * max(agreement, 0.0)
        )

        self._apply_agreement_calibrated_gradient(
            local_gradient=local_gradient,
            remote_gradient=remote_gradient,
            alpha=alpha,
        )
        self._update_peer_statistics(peer.node_id, agreement)
        self.sgd_steps += 1

        return [
            ContactAttempt(
                peer_id=peer.node_id,
                event=self.nearby_events[peer.node_id],
                details={"agreement": agreement}
            )
        ]


    def end_contacts(self, time: int) -> None:
        """End the contact window; PULSE has no deferred communication."""
        self.nearby_events = {}


    def _select_peer(self, candidates: list[PULSENode]) -> PULSENode:
        """Return the best UCB-indexed peer, or the no-contact action.

        For each candidate ``j``, the index is

        .. math::

           \\mathcal{U}_i[j] + \\beta
           \\sqrt{\\log(2+s_i)/(1+n_{ij})}.

        The no-contact action has index zero.  A fresh peer is therefore
        explored through the uncertainty bonus, whereas a peer with a
        sufficiently negative estimated utility may be skipped.  Exact ties
        among the best peers are broken randomly to avoid deterministic bias.
        """
        log_term = math.log(
            2.0 + float(self.total_selection_opportunities)
        )

        indices: list[float] = []
        for peer in candidates:
            visits = self.peer_visits.get(peer.node_id, 0)
            utility = self.peer_utility.get(
                peer.node_id,
                self.initial_utility,
            )
            bonus = self.ucb_exploration * math.sqrt(
                log_term / (1.0 + visits)
            )
            indices.append(utility + bonus)

        best_index = max(indices)

        # The no-contact action has index 0.  A zero tie is resolved in favour
        # of no-contact, making the decision conservative and reproducible.
        if best_index <= 0.0:
            return None

        tied = [
            peer
            for peer, index in zip(candidates, indices)
            if abs(index - best_index) <= 1e-12
        ]

        return tied[int(torch.randint(len(tied), (1,)).item())]
    

    def _update_peer_statistics(self, peer_id: str, agreement: float) -> None:
        """Update :math:`\\mathcal{U}_i[j]` and :math:`n_{ij}` after contact.

        The utility is an exponential moving average of the signed cosine
        agreement.  Positive values indicate historically compatible remote
        gradients; negative values indicate conflict.  Neither quantity is
        updated for no-contact, because no remote gradient is observed.
        """
        previous = self.peer_utility.get(peer_id, self.initial_utility)

        self.peer_utility[peer_id] = (
            (1.0 - self.utility_momentum) * previous
            + self.utility_momentum * agreement
        )
        self.peer_visits[peer_id] = (
            self.peer_visits.get(peer_id, 0) + 1
        )


    def _classifier_state(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU copy of the current personalized head ``h_i``.

        Sending a state copy makes the protocol explicit: a peer evaluates
        node ``i``'s head but cannot mutate the sender's parameters.
        """
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.classifier.state_dict().items()
        }


    def _gradient_on_private_batch(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute a gradient of a supplied head on one local private minibatch.

        The head is recreated temporarily from ``state``.  The encoder is used
        under ``no_grad`` because it is frozen; only the temporary classifier
        receives gradients.  The returned tensors are detached CPU copies, so
        the caller obtains a gradient message rather than a live computation
        graph or any training examples.
        """
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
        return {
            name: gradient.detach().cpu().clone()
            for (name, _), gradient in zip(temporary.named_parameters(), gradients)
        }


    def _cosine_agreement(
        self,
        first: Mapping[str, torch.Tensor],
        second: Mapping[str, torch.Tensor],
    ) -> float:
        """Return signed cosine agreement :math:`a_{ij}` between two gradients.

        A value near ``1`` denotes aligned update directions, a value near
        ``-1`` denotes conflicting directions, and ``0`` is also returned
        when either gradient has negligible norm.  The latter avoids an
        unstable division and makes the remote contribution vanish.
        """
        first_vector = torch.cat([gradient.detach().reshape(-1).float() for gradient in first.values()])
        second_vector = torch.cat([gradient.detach().reshape(-1).float() for gradient in second.values()])
        denominator = first_vector.norm() * second_vector.norm()
        if float(denominator.item()) <= self.norm_epsilon:
            return 0.0
        
        return float(torch.clamp(torch.dot(first_vector, second_vector) / denominator, -1.0, 1.0).item())


    def _apply_agreement_calibrated_gradient(
        self,
        *,
        local_gradient: Mapping[str, torch.Tensor],
        remote_gradient: Mapping[str, torch.Tensor],
        alpha: float,
    ) -> None:
        """Apply the paper's norm-matched gradient-mixing update.

        The remote gradient is first rescaled to the norm of the local one,
        then the update direction is ``(1-alpha) * g_i + alpha * g_j``.
        ``alpha`` is supplied by :meth:`perform_contacts` and is nonzero only
        for a useful peer exchange.  With ``alpha = 0`` this method is exactly
        a standard local head update, which implements no-contact.
        """
        local_norm_sq = sum(float(gradient.float().pow(2).sum().item()) for gradient in local_gradient.values())
        remote_norm_sq = sum(float(gradient.float().pow(2).sum().item()) for gradient in remote_gradient.values())
        scale = math.sqrt(local_norm_sq / max(remote_norm_sq, self.norm_epsilon))

        with torch.no_grad():
            for name, parameter in self.classifier.named_parameters():
                local = local_gradient[name].to(self.device, dtype=parameter.dtype)
                remote = remote_gradient[name].to(self.device, dtype=parameter.dtype) * scale
                aggregate = (1.0 - alpha) * local + alpha * remote
                parameter.add_(aggregate, alpha=-self.config.learning_rate)
