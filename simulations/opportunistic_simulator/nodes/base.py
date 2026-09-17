"""Base abstraction for every algorithm simulated by the framework."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from opportunistic_simulator.data import ContactEvent, ContactAttempt
from opportunistic_simulator.utils.config import TrainingConfig
from opportunistic_simulator.utils.logger import SimulationLogger


class Node(ABC):
    """
    Common node lifecycle independent of the collaboration algorithm.

    Subclasses only decide which compact message to exchange and how to apply
    it.  The simulator calls ``contact`` on both nodes using messages built
    from the same pre-contact snapshot, so contacts are bidirectional and do
    not depend on arbitrary call ordering.
    """
    def __init__(
        self,
        node_id: str,
        encoder: nn.Module,
        classifier: nn.Module,
        train_data: Dataset,
        val_data: Dataset,
        test_data: Dataset,
        config: TrainingConfig,
        device: torch.device,
        seed: int
    ) -> None:
        
        self.node_id = str(node_id)

        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)

        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data

        self.config, self.device, self.seed = config, device, seed

        trainable_parameters = list(self.classifier.parameters()) + [
            parameter for parameter in self.encoder.parameters() if parameter.requires_grad
        ]

        self.optimizer = torch.optim.SGD(trainable_parameters, lr=config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        has_batch_norm = any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in self.classifier.modules())
        if has_batch_norm and len(train_data) < 2:
            raise ValueError("A classifier with BatchNorm requires at least two local training samples per node.")
        
        # BatchNorm cannot train on a one-sample tail batch.  Discarding only
        # that tail makes OPTIMIST robust while leaving every complete batch
        # and all non-BatchNorm baselines untouched.
        drop_last = has_batch_norm and len(train_data) >= config.batch_size
        
        pin_memory = torch.device(device).type == "cuda"
        if config.evaluation_batch_size <= 0:
            raise ValueError("training.evaluation_batch_size must be positive.")

        # Create DataLoaders. Evaluation batching does not affect predictions,
        # and is deliberately independent from the SGD batch size.
        self._train_loader = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True, drop_last=drop_last,
            pin_memory=pin_memory,
        )
        self._train_iterator = iter(self._train_loader)

        self._val_loader = DataLoader(
            val_data, batch_size=self.config.evaluation_batch_size, shuffle=False,
            pin_memory=pin_memory)

        self._test_loader = DataLoader(
            test_data, batch_size=self.config.evaluation_batch_size, shuffle=False,
            pin_memory=pin_memory)

        self.contacts_seen = 0
        self.sgd_steps = 0


    def set_logger(self, logger: SimulationLogger) -> None:
        """Attach the experiment logger to the node."""
        self.logger = logger


    # Training and evaluation methods ------------------------------------------------------
    def _move_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.to(self.device, non_blocking=True) for name, value in inputs.items()}

    def _encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if set(inputs) == {"embedding"}:
            return inputs["embedding"].to(self.device, non_blocking=True)
        return self.encoder(**self._move_inputs(inputs))

    def _next_train_batch(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        try:
            x, y = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self._train_loader)
            x, y = next(self._train_iterator)
        return self._move_inputs(x), y.to(self.device, non_blocking=True)


    def local_train(self) -> dict[str, float] | None:
        """
        Train at most the remaining per-node optimizer-step budget.
        """
        self.classifier.train()
        total_loss = 0.0

        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.train()
        else:
            self.encoder.eval()

        for _ in range(self.config.local_steps):
            x, y = self._next_train_batch()
            # In the standard setting the encoder is frozen. This branch also
            # permits later ablations with an adaptable encoder.
            if set(x) == {"embedding"}:
                z = x["embedding"]
            elif any(parameter.requires_grad for parameter in self.encoder.parameters()):
                z = self.encoder(**x)
            else:
                with torch.no_grad():
                    z = self.encoder(**x)

            logits = self.classifier(z)
            loss = self.loss_fn(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.sgd_steps += 1

            total_loss += float(loss.detach().cpu())

        return {"loss": total_loss / self.config.local_steps, "sgd_steps": self.config.local_steps, "total_sgd_steps": self.sgd_steps}

    def validation_loss(self) -> float:
        """
        Compute a deterministic local validation loss for transfer gating.
        """
        self.encoder.eval()
        self.classifier.eval()
        total, count = 0.0, 0

        with torch.no_grad():
            for x, y in self._val_loader:
                logits = self.classifier(self._encode(x))
                batch_size = len(y)
                total += float(self.loss_fn(logits, y.to(self.device)).cpu()) * batch_size
                count += batch_size

        return total / max(count, 1)

    def evaluate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return predictions and labels; metrics are deliberately external.
        """
        self.encoder.eval()
        self.classifier.eval()
        predictions, labels = [], []

        with torch.no_grad():
            for x, y in self._test_loader:
                logits = self.classifier(self._encode(x))
                predictions.append(logits.argmax(dim=1).cpu())
                labels.append(y.cpu())

        return torch.cat(predictions), torch.cat(labels)


    # Abstract methods for algorithm-specific contact behavior ----------------------------------------------------
    @abstractmethod
    def local_task(self) -> dict[str, float | int] | None:
        """
        Return, e.g., {"loss": 0.42, "local_sgd_steps": 5}.
        """
        raise NotImplementedError

    @abstractmethod
    def begin_contacts(self, nearby_events: list[ContactEvent], time: int) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def perform_contacts(self, nodes: list[Node], time: int) -> list[ContactAttempt]:
        """
        Return one record for each directional contact actually initiated.
        """
        raise NotImplementedError

    @abstractmethod
    def end_contacts(self, time: int) -> dict[str, int]:
        """
        Return, e.g., {"contact_updates_applied": 2}.
        """
        raise NotImplementedError
