"""Controlled image benchmarks with disjoint pretraining and node data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from datasets.config import DatasetConfig
from datasets.core import Sample
from datasets.reporting import REPORTER

from .base import DatasetAdapter, register


class _ImageBenchmarkAdapter(DatasetAdapter):
    """Shared preparation logic for torchvision classification benchmarks.

    Unlike wearable datasets, CIFAR-10 and Fashion-MNIST do not identify
    people.  We therefore create simulated nodes through a label-skewed sample
    partition, while reserving disjoint, class-balanced examples for backbone
    pretraining and its validation set.
    """

    dataset_class: str
    channels: int
    image_size: int
    class_names: list[str]

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        try:
            from torchvision import datasets  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Image benchmarks require torchvision. Install the optional "
                "dependencies with: python -m pip install -e '.[vision]'"
            ) from error

        raw_dir.mkdir(parents=True, exist_ok=True)
        dataset_type = self._torchvision_dataset()
        try:
            dataset_type(root=str(raw_dir), train=True, download=True)
            dataset_type(root=str(raw_dir), train=False, download=True)
        except Exception as error:
            raise RuntimeError(
                f"Unable to download {self.name} into {raw_dir}. "
                "Check network access, or place the torchvision dataset files "
                "there and rerun the command."
            ) from error
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        options = self._options(config)
        train_images, train_labels = self._load_split(source, train=True)
        test_images, test_labels = self._load_split(source, train=False)
        rng = np.random.default_rng(config.split.seed)

        pretrain_indices, validation_indices, simulation_indices = self._reserve_training_pools(
            train_labels,
            options["pretrain_samples_per_class"],
            options["validation_samples_per_class"],
            rng,
        )
        train_by_node = self._dirichlet_partition(
            train_labels,
            simulation_indices,
            options["simulation_nodes"],
            options["dirichlet_alpha"],
            options["min_train_samples_per_node"],
            rng,
        )
        test_by_node = self._dirichlet_partition(
            test_labels,
            np.arange(len(test_labels)),
            options["simulation_nodes"],
            options["dirichlet_alpha"],
            options["min_test_samples_per_node"],
            rng,
        )

        samples: list[Sample] = []
        assignments: list[str] = []
        samples.extend(self._samples_from_indices(train_images, train_labels, pretrain_indices, "pretraining_pool", "pretrain", "train"))
        assignments.extend(["pretrain"] * len(pretrain_indices))
        samples.extend(self._samples_from_indices(train_images, train_labels, validation_indices, "pretraining_validation", "validation", "train"))
        assignments.extend(["validation"] * len(validation_indices))

        for node, indices in enumerate(train_by_node):
            subject_id = f"node_{node:02d}"
            samples.extend(self._samples_from_indices(train_images, train_labels, indices, subject_id, "simulation", "train"))
            assignments.extend(["simulation"] * len(indices))
        for node, indices in enumerate(test_by_node):
            subject_id = f"node_{node:02d}"
            samples.extend(self._samples_from_indices(test_images, test_labels, indices, subject_id, "simulation", "test"))
            assignments.extend(["simulation"] * len(indices))

        self._explicit_splits = assignments
        self._split_metadata = {
            "selection_unit": "example",
            "partition": {
                "strategy": "classwise_dirichlet",
                "alpha": options["dirichlet_alpha"],
                "simulation_nodes": options["simulation_nodes"],
                "minimum_train_samples_per_node": options["min_train_samples_per_node"],
                "minimum_test_samples_per_node": options["min_test_samples_per_node"],
            },
            "pretraining": {
                "source": "official_train",
                "samples_per_class": options["pretrain_samples_per_class"],
                "validation_samples_per_class": options["validation_samples_per_class"],
            },
            "simulation": {
                "train_source": "official_train",
                "test_source": "official_test",
                "node_ids": [f"node_{node:02d}" for node in range(options["simulation_nodes"])],
            },
        }
        REPORTER.info(
            self.name,
            "PARTITION",
            "Reserved disjoint train examples for pretraining/validation; "
            f"partitioned the remaining training data and official test data across {options['simulation_nodes']} nodes (Dirichlet alpha={options['dirichlet_alpha']:g}).",
        )
        return samples

    def _configuration_summary(self, config: DatasetConfig) -> str:
        options = self._options(config)
        return (
            "mode=disjoint example pools; "
            f"pretrain={options['pretrain_samples_per_class']}/class; "
            f"validation={options['validation_samples_per_class']}/class; "
            f"simulation_nodes={options['simulation_nodes']}; "
            f"Dirichlet alpha={options['dirichlet_alpha']:g}; seed={config.split.seed}"
        )

    def _preprocessing_summary(self, config: DatasetConfig) -> str:
        return (
            "Loading official images, scaling pixels to [0, 1], creating "
            "disjoint pretraining/validation pools, and applying a class-wise "
            "Dirichlet partition to simulation nodes."
        )

    def _options(self, config: DatasetConfig) -> dict[str, int | float]:
        defaults: dict[str, int | float] = {
            "pretrain_samples_per_class": 1000,
            "validation_samples_per_class": 250,
            "simulation_nodes": 10,
            "dirichlet_alpha": 0.1,
            "min_train_samples_per_node": 200,
            "min_test_samples_per_node": 50,
        }
        unexpected = sorted(set(config.options) - set(defaults))
        if unexpected:
            raise ValueError(f"Unsupported {self.name} options: {unexpected}")
        values = {key: config.options.get(key, default) for key, default in defaults.items()}
        for key in ("pretrain_samples_per_class", "validation_samples_per_class", "simulation_nodes", "min_train_samples_per_node", "min_test_samples_per_node"):
            values[key] = int(values[key])
            if values[key] <= 0:
                raise ValueError(f"{self.name}: {key} must be a positive integer.")
        values["dirichlet_alpha"] = float(values["dirichlet_alpha"])
        if values["dirichlet_alpha"] <= 0:
            raise ValueError(f"{self.name}: dirichlet_alpha must be positive.")
        return values

    def _torchvision_dataset(self) -> Any:
        from torchvision import datasets

        return getattr(datasets, self.dataset_class)

    def _load_split(self, root: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
        dataset = self._torchvision_dataset()(root=str(root), train=train, download=False)
        images = np.asarray(dataset.data)
        labels = np.asarray(dataset.targets, dtype=np.int64)
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]
        elif images.ndim == 4:
            images = np.moveaxis(images, -1, 1)
        else:
            raise ValueError(f"Unexpected {self.name} image shape: {images.shape}")
        return images.astype(np.float32) / 255.0, labels

    @staticmethod
    def _reserve_training_pools(
        labels: np.ndarray,
        pretrain_per_class: int,
        validation_per_class: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pretrain: list[np.ndarray] = []
        validation: list[np.ndarray] = []
        simulation: list[np.ndarray] = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            shuffled = rng.permutation(indices)
            boundary = pretrain_per_class + validation_per_class
            if boundary >= len(shuffled):
                raise ValueError(
                    f"Requested {boundary} reserved samples for class {label}, "
                    f"but the official training split has only {len(shuffled)}."
                )
            pretrain.append(shuffled[:pretrain_per_class])
            validation.append(shuffled[pretrain_per_class:boundary])
            simulation.append(shuffled[boundary:])
        return np.concatenate(pretrain), np.concatenate(validation), np.concatenate(simulation)

    @staticmethod
    def _dirichlet_partition(
        labels: np.ndarray,
        indices: np.ndarray,
        nodes: int,
        alpha: float,
        minimum_samples: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Assign every provided example once using a class-wise Dirichlet draw."""
        for _ in range(1_000):
            buckets = [[] for _ in range(nodes)]
            for label in np.unique(labels[indices]):
                class_indices = rng.permutation(indices[labels[indices] == label])
                proportions = rng.dirichlet(np.full(nodes, alpha))
                cut_points = np.cumsum(proportions[:-1] * len(class_indices)).astype(int)
                for node, chunk in enumerate(np.split(class_indices, cut_points)):
                    buckets[node].append(chunk)
            result = [np.concatenate(parts) if parts else np.empty(0, dtype=int) for parts in buckets]
            if min(map(len, result)) >= minimum_samples:
                return result
        raise RuntimeError(
            f"Could not create a Dirichlet(alpha={alpha:g}) partition with at least "
            f"{minimum_samples} samples per node after 1,000 attempts. Reduce the minimum or increase alpha."
        )

    def _samples_from_indices(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        subject_id: str,
        role: str,
        official_split: str,
    ) -> list[Sample]:
        return [
            Sample(
                inputs={"image": images[index]},
                label=int(labels[index]),
                subject_id=subject_id,
                session_id=official_split,
                device_id="image_benchmark",
                timestamp=float(index),
                metadata={"official_split": official_split, "original_index": int(index), "role": role},
            )
            for index in indices
        ]


@register
class Cifar10Adapter(_ImageBenchmarkAdapter):
    name = "cifar10"
    dataset_class = "CIFAR10"
    channels = 3
    image_size = 32
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    descriptor = {
        "task_type": "multiclass",
        "class_names": class_names,
        "encoder": "image_encoder_tiny_cnn",
        "channels": ["red", "green", "blue"],
        "image_size": [32, 32],
        "normalization": "unit_interval",
        "embedding_dim": 64,
    }


@register
class FashionMnistAdapter(_ImageBenchmarkAdapter):
    name = "fashion_mnist"
    dataset_class = "FashionMNIST"
    channels = 1
    image_size = 28
    class_names = ["t-shirt_top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
    descriptor = {
        "task_type": "multiclass",
        "class_names": class_names,
        "encoder": "image_encoder_tiny_cnn",
        "channels": ["gray"],
        "image_size": [28, 28],
        "normalization": "unit_interval",
        "embedding_dim": 64,
    }
