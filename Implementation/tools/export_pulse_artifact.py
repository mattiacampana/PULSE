#!/usr/bin/env python3
"""Export exact PULSE weights and replay data to a versioned NPZ artifact.

The encoder always comes from a supplied PyTorch checkpoint.  The head either
comes from a supplied state dictionary or, when explicitly requested, is the
canonical deterministic ``common`` initial head created by the supplied PULSE
source tree for a declared simulation seed.  No implicit/random fallback exists.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
	import numpy as np
except ImportError as exc:  # pragma: no cover - dependency diagnostic
	raise SystemExit("export_pulse_artifact.py requires NumPy") from exc

try:
	import torch
except ImportError as exc:  # pragma: no cover - dependency diagnostic
	raise SystemExit("export_pulse_artifact.py requires PyTorch") from exc

try:
	import yaml
except ImportError as exc:  # pragma: no cover - dependency diagnostic
	raise SystemExit("export_pulse_artifact.py requires PyYAML") from exc


SCHEMA_VERSION = "senswear.pulse.artifact.v1"
BATCH_COUNT = 4
BATCH_SIZE = 16
CHANNEL_COUNT = 3
WINDOW_LENGTH = 128
CLASS_COUNT = 6
NORMALIZATION_EPSILON = np.float32(1.0e-6)
NORMALIZATION_VERIFY_TOLERANCE = np.float32(2.0e-4)
DEPLOY_FOLD_ATOL = 2.0e-5
DEPLOY_FOLD_RTOL = 1.0e-5
PULSE_COMMON_INIT_ID = "common"
PULSE_EMBEDDING_DIM = 64
PULSE_CLASS_COUNT = 6
PULSE_CLASSIFIER_HIDDEN_DIMS = (64,)

# (canonical PyTorch state-dict name, exact shape, artifact key)
ENCODER_SPEC = (
	("features.0.weight", (32, 3, 5), "encoder.features.0.weight"),
	("features.0.bias", (32,), "encoder.features.0.bias"),
	("features.3.weight", (64, 32, 5), "encoder.features.3.weight"),
	("features.3.bias", (64,), "encoder.features.3.bias"),
	("projection.weight", (64, 64), "encoder.projection.weight"),
	("projection.bias", (64,), "encoder.projection.bias"),
)

HEAD_SPEC = (
	("feature_layers.0.weight", (64, 64), "head.feature_layers.0.weight"),
	("feature_layers.0.bias", (64,), "head.feature_layers.0.bias"),
	("output_layer.weight", (6, 64), "head.output_layer.weight"),
	("output_layer.bias", (6,), "head.output_layer.bias"),
)

HEAD_FLAT_ORDER = (
	"head.feature_layers.0.weight",
	"head.feature_layers.0.bias",
	"head.output_layer.weight",
	"head.output_layer.bias",
)

REPLAY_ROLES = (
	"scheduled_r_step_0",
	"scheduled_r_step_1",
	"initiator_local",
	"responder_remote",
)

DEPLOY_REQUIRED_FILES = (
	"data.h5",
	"manifest.csv",
	"hhar_device_setup.py",
	"uci_har_encoder.pt",
)

DEPLOY_MANIFEST_COLUMNS = (
	"sample_index",
	"subject_id",
	"session_id",
	"device_id",
	"timestamp",
	"label",
	"split",
	"metadata_json",
)

PAYLOAD_KEYS = tuple(spec[2] for spec in ENCODER_SPEC + HEAD_SPEC) + (
	"head.flat",
	"replay.x",
	"replay.y",
	"replay.source_indices",
	"replay.owner_ids",
)


class ArtifactError(ValueError):
	"""Raised when an input cannot represent the exact PULSE artifact schema."""


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as source:
		for chunk in iter(lambda: source.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _classifier_fingerprint_full(classifier: torch.nn.Module) -> str:
	"""Return the full digest underlying PULSE's 12-character fingerprint."""

	digest = hashlib.sha256()
	for name, parameter in classifier.state_dict().items():
		digest.update(name.encode("utf-8"))
		digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
	return digest.hexdigest()


def _derived_common_head_seed(simulation_seed: int) -> int:
	seed_material = f"{simulation_seed}:{PULSE_COMMON_INIT_ID}".encode("utf-8")
	return int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") % (
		2**63 - 1
	)


def _git_source_state(source_root: Path) -> dict[str, Any]:
	"""Return revision/dirty provenance without changing global Git configuration."""

	command_prefix = [
		"git",
		"-c",
		f"safe.directory={source_root.as_posix()}",
		"-C",
		str(source_root),
	]
	try:
		revision_result = subprocess.run(
			command_prefix + ["rev-parse", "HEAD"],
			check=False,
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			timeout=10,
		)
	except (OSError, subprocess.SubprocessError) as exc:
		return {
			"revision": "unavailable",
			"dirty": "unavailable",
			"availability": "unavailable",
			"detail": f"git invocation failed: {exc}",
		}
	if revision_result.returncode != 0:
		detail = revision_result.stderr.strip() or revision_result.stdout.strip()
		return {
			"revision": "unavailable",
			"dirty": "unavailable",
			"availability": "unavailable",
			"detail": detail.splitlines()[0] if detail else "not a Git work tree",
		}

	revision = revision_result.stdout.strip()
	try:
		status_result = subprocess.run(
			command_prefix + ["status", "--porcelain", "--untracked-files=normal"],
			check=False,
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			timeout=10,
		)
	except (OSError, subprocess.SubprocessError) as exc:
		return {
			"revision": revision,
			"dirty": "unavailable",
			"availability": "partial",
			"detail": f"git status failed: {exc}",
		}
	if status_result.returncode != 0:
		detail = status_result.stderr.strip() or status_result.stdout.strip()
		return {
			"revision": revision,
			"dirty": "unavailable",
			"availability": "partial",
			"detail": detail.splitlines()[0] if detail else "git status failed",
		}
	return {
		"revision": revision,
		"dirty": bool(status_result.stdout.strip()),
		"availability": "available",
	}


def _load_pulse_model_class(models_path: Path) -> type[torch.nn.Module]:
	"""Execute the supplied models.py and return its exact MLPClassifier class."""

	try:
		source = models_path.read_text(encoding="utf-8")
		code = compile(source, str(models_path), "exec")
		namespace: dict[str, Any] = {
			"__file__": str(models_path),
			"__name__": "_senswear_supplied_pulse_models",
		}
		exec(code, namespace)
	except Exception as exc:
		raise ArtifactError(f"could not load supplied PULSE models.py: {exc}") from exc
	classifier_class = namespace.get("MLPClassifier")
	if not isinstance(classifier_class, type) or not issubclass(classifier_class, torch.nn.Module):
		raise ArtifactError("supplied PULSE models.py does not define an nn.Module MLPClassifier")
	return classifier_class


def _load_pulse_common_functions(
	commons_path: Path,
	classifier_class: type[torch.nn.Module],
) -> tuple[Any, Any]:
	"""Load only the two canonical head helpers from the supplied commons.py."""

	try:
		source = commons_path.read_text(encoding="utf-8")
		tree = ast.parse(source, filename=str(commons_path))
	except Exception as exc:
		raise ArtifactError(f"could not parse supplied PULSE utils/commons.py: {exc}") from exc
	function_names = {"make_seeded_classifier", "classifier_fingerprint"}
	functions = [
		node for node in tree.body
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names
	]
	counts = {name: sum(node.name == name for node in functions) for name in function_names}
	if counts != {"make_seeded_classifier": 1, "classifier_fingerprint": 1}:
		raise ArtifactError(
			"supplied PULSE utils/commons.py must define exactly one "
			"make_seeded_classifier and classifier_fingerprint function"
		)
	if any(isinstance(node, ast.AsyncFunctionDef) for node in functions):
		raise ArtifactError("supplied PULSE common-head helpers must be synchronous functions")

	namespace: dict[str, Any] = {
		"hashlib": hashlib,
		"torch": torch,
		"MLPClassifier": classifier_class,
	}
	try:
		selected = ast.Module(body=functions, type_ignores=[])
		ast.fix_missing_locations(selected)
		exec(compile(selected, str(commons_path), "exec"), namespace)
	except Exception as exc:
		raise ArtifactError(f"could not load supplied PULSE common-head helpers: {exc}") from exc
	return namespace["make_seeded_classifier"], namespace["classifier_fingerprint"]


def _validate_common_head_architecture(classifier: torch.nn.Module) -> None:
	feature_layers = getattr(classifier, "feature_layers", None)
	output_layer = getattr(classifier, "output_layer", None)
	expected_modules = (
		"",
		"feature_layers",
		"feature_layers.0",
		"feature_layers.1",
		"output_layer",
	)
	if (
		not isinstance(feature_layers, torch.nn.Sequential)
		or len(feature_layers) != 2
		or not isinstance(feature_layers[0], torch.nn.Linear)
		or feature_layers[0].in_features != PULSE_EMBEDDING_DIM
		or feature_layers[0].out_features != PULSE_CLASSIFIER_HIDDEN_DIMS[0]
		or feature_layers[0].bias is None
		or not isinstance(feature_layers[1], torch.nn.ReLU)
		or not isinstance(output_layer, torch.nn.Linear)
		or output_layer.in_features != PULSE_CLASSIFIER_HIDDEN_DIMS[0]
		or output_layer.out_features != PULSE_CLASS_COUNT
		or output_layer.bias is None
		or tuple(name for name, _ in classifier.named_modules()) != expected_modules
		or tuple(classifier.named_buffers())
	):
		raise ArtifactError(
			"supplied PULSE MLPClassifier is not the required "
			"64 -> 64 (ReLU) -> 6 architecture with biases and no BatchNorm/buffers"
		)

	expected_names = tuple(item[0] for item in HEAD_SPEC)
	state = classifier.state_dict()
	if tuple(state) != expected_names:
		raise ArtifactError(
			f"supplied PULSE MLPClassifier state schema is {list(state)}, "
			f"expected {list(expected_names)}"
		)
	for name, shape, _ in HEAD_SPEC:
		if tuple(state[name].shape) != shape:
			raise ArtifactError(
				f"supplied PULSE MLPClassifier tensor {name} has shape "
				f"{tuple(state[name].shape)}, expected {shape}"
			)
	if sum(parameter.numel() for parameter in classifier.parameters()) != 4550:
		raise ArtifactError("supplied PULSE MLPClassifier does not have exactly 4550 parameters")


def _state_dicts_equal(
	first: Mapping[str, torch.Tensor],
	second: Mapping[str, torch.Tensor],
) -> bool:
	return tuple(first) == tuple(second) and all(
		torch.equal(first[name].detach().cpu(), second[name].detach().cpu()) for name in first
	)


def _config_value(mapping: Mapping[str, Any], section: str, key: str) -> Any:
	section_value = mapping.get(section)
	if not isinstance(section_value, Mapping) or key not in section_value:
		raise ArtifactError(f"PULSE config must define {section}.{key}")
	return section_value[key]


def _load_and_validate_pulse_config(
	config_path: Path,
	pulse_source_root: Path,
	simulation_seed: int,
	checkpoint_path: Path,
	allow_deploy_encoder_override: bool = False,
	deploy_train_fraction: float | None = None,
	deploy_val_fraction: float | None = None,
) -> dict[str, Any]:
	"""Bind seeded initialization to the exact controlled-evaluation YAML."""

	path = config_path.expanduser().resolve()
	root = pulse_source_root.expanduser().resolve()
	if not path.is_file():
		raise ArtifactError(f"PULSE config does not exist: {path}")
	try:
		path.relative_to(root)
	except ValueError as exc:
		raise ArtifactError(
			f"--pulse-config must be inside --pulse-source-root ({root}), got {path}"
		) from exc
	try:
		loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise ArtifactError(f"could not load PULSE config {path}: {exc}") from exc
	if not isinstance(loaded, Mapping):
		raise ArtifactError("PULSE config root must be a mapping")

	seeds = loaded.get("seeds")
	if (
		not isinstance(seeds, list)
		or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
		or simulation_seed not in seeds
	):
		raise ArtifactError(
			f"--simulation-seed {simulation_seed} must occur in the PULSE config seeds list"
		)

	expected = {
		("model", "freeze_encoder"): True,
		("model", "input_channels"): CHANNEL_COUNT,
		("model", "embedding_dim"): PULSE_EMBEDDING_DIM,
		("model", "num_classes"): PULSE_CLASS_COUNT,
		("model", "classifier_hidden_dims"): list(PULSE_CLASSIFIER_HIDDEN_DIMS),
		("training", "local_steps"): 2,
		("training", "batch_size"): BATCH_SIZE,
		("training", "learning_rate"): 0.01,
		("algorithm", "name"): "pulse",
		("pulse", "alpha_max"): 0.5,
		("pulse", "utility_momentum"): 0.2,
		("pulse", "initial_utility"): 0.0,
		("pulse", "ucb_exploration"): 0.25,
		("pulse", "norm_epsilon"): 1.0e-12,
	}
	validated: dict[str, dict[str, Any]] = {}
	for (section, key), expected_value in expected.items():
		actual = _config_value(loaded, section, key)
		if isinstance(expected_value, float):
			matches = (
				isinstance(actual, (int, float))
				and not isinstance(actual, bool)
				and float(actual) == expected_value
			)
		else:
			matches = actual == expected_value and type(actual) is type(expected_value)
		if not matches:
			raise ArtifactError(
				f"PULSE config {section}.{key} is {actual!r}, expected {expected_value!r}"
			)
		validated.setdefault(section, {})[key] = actual
	if allow_deploy_encoder_override:
		if deploy_train_fraction is None or deploy_val_fraction is None:
			raise ArtifactError("deploy config validation requires explicit split fractions")
		deploy_expected = {
			("data", "num_nodes"): 9,
			("data", "node_selection"): "random",
			("data", "train_fraction"): deploy_train_fraction,
			("data", "val_fraction"): deploy_val_fraction,
		}
		for (section, key), expected_value in deploy_expected.items():
			actual = _config_value(loaded, section, key)
			matches = (
				isinstance(actual, (int, float))
				and not isinstance(actual, bool)
				and float(actual) == float(expected_value)
				if isinstance(expected_value, float)
				else actual == expected_value and type(actual) is type(expected_value)
			)
			if not matches:
				raise ArtifactError(
					f"PULSE config {section}.{key} is {actual!r}, expected {expected_value!r}"
				)
			validated.setdefault(section, {})[key] = actual

	checkpoint_value = _config_value(loaded, "model", "checkpoint")
	if not isinstance(checkpoint_value, str) or not checkpoint_value:
		raise ArtifactError("PULSE config model.checkpoint must be a non-empty path string")
	declared_checkpoint = Path(checkpoint_value).expanduser()
	if not declared_checkpoint.is_absolute():
		declared_checkpoint = path.parent / declared_checkpoint
	declared_checkpoint = declared_checkpoint.resolve()
	declared_available = declared_checkpoint.is_file()
	if not declared_available and not allow_deploy_encoder_override:
		raise ArtifactError(
			"PULSE config's declared encoder checkpoint does not exist: "
			f"{declared_checkpoint}"
		)
	supplied_checkpoint = checkpoint_path.expanduser().resolve()
	if not supplied_checkpoint.is_file():
		raise ArtifactError(f"supplied encoder checkpoint does not exist: {supplied_checkpoint}")
	declared_hash = _sha256_file(declared_checkpoint) if declared_available else None
	supplied_hash = _sha256_file(supplied_checkpoint)
	checkpoint_match = declared_hash == supplied_hash if declared_available else False
	if not checkpoint_match and not allow_deploy_encoder_override:
		raise ArtifactError(
			"--checkpoint does not match the encoder checkpoint declared by "
			f"--pulse-config (SHA-256 {supplied_hash} != {declared_hash})"
		)

	return {
		"record": _source_record(path),
		"declared_seeds": seeds,
		"selected_simulation_seed": simulation_seed,
		"validated_fields": validated,
		"declared_checkpoint": (
			_source_record(declared_checkpoint) if declared_available else {
				"path": str(declared_checkpoint),
				"available": False,
			}
		),
		"supplied_checkpoint": _source_record(supplied_checkpoint),
		"supplied_checkpoint_sha256_match": checkpoint_match,
		"checkpoint_binding": (
			"yaml_declared_sha256_match"
			if checkpoint_match else "explicit_deploy_encoder_override"
		),
		"deploy_encoder_override": {
			"permitted": bool(allow_deploy_encoder_override),
			"used": bool(allow_deploy_encoder_override and not checkpoint_match),
			"reason": (
				"deploy bundle supplies a Conv1d-BatchNorm encoder that is folded for "
				"frozen evaluation; the YAML is used only to bind PULSE hyperparameters "
				"and canonical common-head initialization"
				if allow_deploy_encoder_override else None
			),
		},
		"deploy_data_override": {
			"binding": (
				"explicit_deploy_bundle_override" if allow_deploy_encoder_override else None
			),
			"declared_path": (
				str(_config_value(loaded, "data", "path"))
				if allow_deploy_encoder_override else None
			),
		},
	}


def _canonical_common_head(
	pulse_source_root: Path,
	simulation_seed: int,
	pulse_config: Path,
	checkpoint_path: Path,
	allow_deploy_encoder_override: bool = False,
	deploy_train_fraction: float | None = None,
	deploy_val_fraction: float | None = None,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
	"""Create and independently verify PULSE's canonical seeded ``common`` head."""

	root = pulse_source_root.expanduser().resolve()
	models_path = root / "models.py"
	commons_path = root / "utils" / "commons.py"
	if not root.is_dir():
		raise ArtifactError(f"PULSE source root is not a directory: {root}")
	if not models_path.is_file() or not commons_path.is_file():
		raise ArtifactError(
			"PULSE source root must contain models.py and utils/commons.py; "
			f"got {root}"
		)
	config_provenance = _load_and_validate_pulse_config(
		pulse_config,
		root,
		simulation_seed,
		checkpoint_path,
		allow_deploy_encoder_override,
		deploy_train_fraction,
		deploy_val_fraction,
	)

	classifier_class = _load_pulse_model_class(models_path)
	make_seeded_classifier, upstream_fingerprint = _load_pulse_common_functions(
		commons_path, classifier_class
	)
	derived_seed = _derived_common_head_seed(simulation_seed)
	try:
		classifier = make_seeded_classifier(
			PULSE_EMBEDDING_DIM,
			PULSE_CLASS_COUNT,
			list(PULSE_CLASSIFIER_HIDDEN_DIMS),
			simulation_seed,
			PULSE_COMMON_INIT_ID,
			batch_norm=False,
		)
		repeated = make_seeded_classifier(
			PULSE_EMBEDDING_DIM,
			PULSE_CLASS_COUNT,
			list(PULSE_CLASSIFIER_HIDDEN_DIMS),
			simulation_seed,
			PULSE_COMMON_INIT_ID,
			batch_norm=False,
		)
		with torch.random.fork_rng():
			torch.manual_seed(derived_seed)
			independent = classifier_class(
				PULSE_EMBEDDING_DIM,
				PULSE_CLASS_COUNT,
				list(PULSE_CLASSIFIER_HIDDEN_DIMS),
				batch_norm=False,
			)
	except Exception as exc:
		raise ArtifactError(f"supplied PULSE common-head initialization failed: {exc}") from exc
	for candidate in (classifier, repeated, independent):
		if not isinstance(candidate, torch.nn.Module):
			raise ArtifactError("supplied PULSE make_seeded_classifier did not return an nn.Module")
		_validate_common_head_architecture(candidate)
	if not _state_dicts_equal(classifier.state_dict(), repeated.state_dict()):
		raise ArtifactError("supplied PULSE common-head initialization is not deterministic")
	if not _state_dicts_equal(classifier.state_dict(), independent.state_dict()):
		raise ArtifactError(
			"supplied PULSE make_seeded_classifier does not match the canonical "
			"SHA256('<simulation_seed>:common') seed derivation"
		)

	full_fingerprint = _classifier_fingerprint_full(classifier)
	try:
		verified_short = upstream_fingerprint(classifier)
	except Exception as exc:
		raise ArtifactError(f"supplied PULSE classifier_fingerprint failed: {exc}") from exc
	if verified_short != full_fingerprint[:12]:
		raise ArtifactError(
			"supplied PULSE classifier_fingerprint does not match the canonical "
			"state-name/tensor-byte SHA-256 digest"
		)

	state = {
		name: parameter.detach().cpu().contiguous().clone()
		for name, parameter in classifier.state_dict().items()
	}
	provenance = {
		"kind": "pulse_seeded_common_initial_head",
		"simulation_seed": simulation_seed,
		"init_id": PULSE_COMMON_INIT_ID,
		"derived_torch_seed": derived_seed,
		"fingerprint_sha256_full": full_fingerprint,
		"fingerprint_sha256_12": full_fingerprint[:12],
		"upstream_classifier_fingerprint_12": verified_short,
		"torch_version": torch.__version__,
		"architecture": {
			"embedding_dim": PULSE_EMBEDDING_DIM,
			"num_classes": PULSE_CLASS_COUNT,
			"classifier_hidden_dims": list(PULSE_CLASSIFIER_HIDDEN_DIMS),
			"batch_norm": False,
			"parameter_count": 4550,
		},
		"source": {
			"root": str(root),
			"models_py": _source_record(models_path),
			"commons_py": _source_record(commons_path),
			"config": config_provenance,
			"git": _git_source_state(root),
		},
	}
	return state, provenance


def _resolve_deploy_bundle(bundle: Path) -> dict[str, Path]:
	root = bundle.expanduser().resolve()
	if not root.is_dir():
		raise ArtifactError(f"--deploy-bundle is not a directory: {root}")
	paths = {name: root / name for name in DEPLOY_REQUIRED_FILES}
	missing = [name for name, path in paths.items() if not path.is_file()]
	if missing:
		raise ArtifactError(
			f"deploy bundle {root} is missing required files: {missing}"
		)
	return {"root": root, **paths}


def _load_deploy_setup_definitions(
	setup_path: Path,
) -> tuple[type[torch.nn.Module], type[torch.nn.Module], Any]:
	"""Load only the encoder/head definitions from the supplied deploy setup script."""

	try:
		source = setup_path.read_text(encoding="utf-8")
		tree = ast.parse(source, filename=str(setup_path))
	except Exception as exc:
		raise ArtifactError(f"could not parse deploy setup script {setup_path}: {exc}") from exc
	wanted_classes = {"_ConvBranch", "IMUEncoderTiny", "MLPClassifier"}
	wanted_functions = {"_make_head"}
	selected = [
		node for node in tree.body
		if (
			isinstance(node, ast.ClassDef) and node.name in wanted_classes
		) or (
			isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
			and node.name in wanted_functions
		)
	]
	counts = {
		name: sum(getattr(node, "name", None) == name for node in selected)
		for name in wanted_classes | wanted_functions
	}
	if any(count != 1 for count in counts.values()):
		raise ArtifactError(
			"deploy hhar_device_setup.py must define exactly one _ConvBranch, "
			"IMUEncoderTiny, MLPClassifier, and _make_head"
		)
	if any(isinstance(node, ast.AsyncFunctionDef) for node in selected):
		raise ArtifactError("deploy _make_head must be synchronous")
	namespace: dict[str, Any] = {
		"__file__": str(setup_path),
		"__name__": "_senswear_supplied_hhar_device_setup",
		"torch": torch,
		"nn": torch.nn,
	}
	try:
		module = ast.Module(body=selected, type_ignores=[])
		ast.fix_missing_locations(module)
		exec(compile(module, str(setup_path), "exec"), namespace)
	except Exception as exc:
		raise ArtifactError(f"could not load deploy model definitions: {exc}") from exc
	encoder_class = namespace.get("IMUEncoderTiny")
	classifier_class = namespace.get("MLPClassifier")
	make_head = namespace.get("_make_head")
	if (
		not isinstance(encoder_class, type)
		or not issubclass(encoder_class, torch.nn.Module)
		or not isinstance(classifier_class, type)
		or not issubclass(classifier_class, torch.nn.Module)
		or not callable(make_head)
	):
		raise ArtifactError("deploy setup definitions are not the required PyTorch types")
	return encoder_class, classifier_class, make_head


def _validate_deploy_encoder_architecture(encoder: torch.nn.Module) -> tuple[float, float]:
	branch = getattr(encoder, "encoder", None)
	network = getattr(branch, "network", None)
	projection = getattr(branch, "projection", None)
	if not isinstance(network, torch.nn.Sequential) or len(network) != 8:
		raise ArtifactError("deploy encoder must contain the exact eight-layer network")
	conv1, bn1, relu1, pool, conv2, bn2, relu2, average = network
	convolutions = ((conv1, 3, 32), (conv2, 32, 64))
	for convolution, input_channels, output_channels in convolutions:
		if (
			not isinstance(convolution, torch.nn.Conv1d)
			or convolution.in_channels != input_channels
			or convolution.out_channels != output_channels
			or convolution.kernel_size != (5,)
			or convolution.stride != (1,)
			or convolution.padding != (2,)
			or convolution.dilation != (1,)
			or convolution.groups != 1
			or convolution.bias is not None
			or convolution.padding_mode != "zeros"
		):
			raise ArtifactError(
				"deploy encoder convolutions must be biasless 3->32->64 Conv1d "
				"layers with kernel 5 and padding 2"
			)
	for batch_norm, features in ((bn1, 32), (bn2, 64)):
		if (
			not isinstance(batch_norm, torch.nn.BatchNorm1d)
			or batch_norm.num_features != features
			or not batch_norm.affine
			or not batch_norm.track_running_stats
			or batch_norm.running_mean is None
			or batch_norm.running_var is None
			or not math.isfinite(float(batch_norm.eps))
			or float(batch_norm.eps) <= 0.0
		):
			raise ArtifactError("deploy encoder BatchNorm layers have an unsupported schema")
	if (
		not isinstance(relu1, torch.nn.ReLU)
		or not isinstance(relu2, torch.nn.ReLU)
		or not isinstance(pool, torch.nn.MaxPool1d)
		or pool.kernel_size != 2
		or pool.stride != 2
		or pool.padding != 0
		or pool.dilation != 1
		or pool.ceil_mode
		or not isinstance(average, torch.nn.AdaptiveAvgPool1d)
		or average.output_size != 1
		or not isinstance(projection, torch.nn.Linear)
		or projection.in_features != 64
		or projection.out_features != 64
		or projection.bias is None
	):
		raise ArtifactError("deploy encoder pooling, activation, or projection schema changed")
	return float(bn1.eps), float(bn2.eps)


def _deploy_head_from_setup(
	classifier_class: type[torch.nn.Module],
	make_head: Any,
	head_seed: int,
	setup_path: Path,
	initiator_node: str,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
	if head_seed < 0:
		raise ArtifactError("--deploy-head-seed must be non-negative")
	rng_before = torch.random.get_rng_state().clone()
	try:
		first = make_head(PULSE_EMBEDDING_DIM, PULSE_CLASS_COUNT, head_seed)
		repeated = make_head(PULSE_EMBEDDING_DIM, PULSE_CLASS_COUNT, head_seed)
		with torch.random.fork_rng(devices=[]):
			torch.manual_seed(head_seed)
			independent = classifier_class(
				PULSE_EMBEDDING_DIM,
				PULSE_CLASS_COUNT,
				hidden_dims=list(PULSE_CLASSIFIER_HIDDEN_DIMS),
			)
	except Exception as exc:
		raise ArtifactError(f"deploy setup head initialization failed: {exc}") from exc
	if not torch.equal(rng_before, torch.random.get_rng_state()):
		raise ArtifactError("deploy _make_head unexpectedly changed the global PyTorch RNG")
	for candidate in (first, repeated, independent):
		if not isinstance(candidate, torch.nn.Module):
			raise ArtifactError("deploy _make_head did not return an nn.Module")
		_validate_common_head_architecture(candidate)
	if not _state_dicts_equal(first.state_dict(), repeated.state_dict()):
		raise ArtifactError("deploy setup head initialization is not deterministic")
	if not _state_dicts_equal(first.state_dict(), independent.state_dict()):
		raise ArtifactError(
			"deploy _make_head does not match torch.manual_seed(head_seed) initialization"
		)
	state = {
		name: value.detach().cpu().contiguous().clone()
		for name, value in first.state_dict().items()
	}
	fingerprint = _classifier_fingerprint_full(first)
	return state, {
		"kind": "deploy_setup_node_initial_head",
		"initiator_node": initiator_node,
		"base_head_seed": head_seed,
		"initiator_position": 0,
		"effective_torch_seed": head_seed,
		"selected_node_order": [initiator_node],
		"fingerprint_sha256_full": fingerprint,
		"fingerprint_sha256_12": fingerprint[:12],
		"torch_version": torch.__version__,
		"architecture": {
			"embedding_dim": PULSE_EMBEDDING_DIM,
			"num_classes": PULSE_CLASS_COUNT,
			"classifier_hidden_dims": list(PULSE_CLASSIFIER_HIDDEN_DIMS),
			"batch_norm": False,
			"parameter_count": 4550,
		},
		"source": {
			"script": _source_record(setup_path),
			"function": "_make_head",
			"semantics": "head_seed + selected_node_position; initiator exported at position 0",
		},
	}


def _fold_deploy_encoder(
	encoder: torch.nn.Module,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	encoder.eval()
	network = encoder.encoder.network
	projection = encoder.encoder.projection
	arrays: dict[str, np.ndarray] = {}
	fold_records: list[dict[str, Any]] = []
	for convolution_index, batch_norm_index, artifact_weight, artifact_bias in (
		(0, 1, "encoder.features.0.weight", "encoder.features.0.bias"),
		(4, 5, "encoder.features.3.weight", "encoder.features.3.bias"),
	):
		convolution = network[convolution_index]
		batch_norm = network[batch_norm_index]
		try:
			weight, bias = torch.nn.utils.fusion.fuse_conv_bn_weights(
				convolution.weight,
				convolution.bias,
				batch_norm.running_mean,
				batch_norm.running_var,
				batch_norm.eps,
				batch_norm.weight,
				batch_norm.bias,
			)
		except Exception as exc:
			raise ArtifactError(f"could not fold deploy Conv1d/BatchNorm pair: {exc}") from exc
		arrays[artifact_weight] = weight.detach().cpu().to(torch.float32).contiguous().numpy().copy()
		arrays[artifact_bias] = bias.detach().cpu().to(torch.float32).contiguous().numpy().copy()
		fold_records.append({
			"convolution_module": f"encoder.network.{convolution_index}",
			"batch_norm_module": f"encoder.network.{batch_norm_index}",
			"batch_norm_epsilon": float(batch_norm.eps),
			"artifact_weight": artifact_weight,
			"artifact_bias": artifact_bias,
		})
	arrays["encoder.projection.weight"] = (
		projection.weight.detach().cpu().to(torch.float32).contiguous().numpy().copy()
	)
	arrays["encoder.projection.bias"] = (
		projection.bias.detach().cpu().to(torch.float32).contiguous().numpy().copy()
	)
	digest = hashlib.sha256()
	for canonical_name, _, artifact_key in ENCODER_SPEC:
		digest.update(canonical_name.encode("utf-8"))
		digest.update(np.ascontiguousarray(arrays[artifact_key], dtype=np.float32).tobytes())
	return arrays, {
		"kind": "torch_eval_conv_batch_norm_fusion",
		"implementation": "torch.nn.utils.fusion.fuse_conv_bn_weights",
		"formula": (
			"scale=gamma/sqrt(running_var+eps); "
			"W_fused=W*scale; b_fused=(b_conv-running_mean)*scale+beta"
		),
		"source_convolution_bias": None,
		"pairs": fold_records,
		"folded_named_tensor_sha256": digest.hexdigest(),
		"folded_named_tensor_sha256_contract": (
			"SHA256(canonical_name_utf8 || contiguous_cpu_float32_bytes) in ENCODER_SPEC order"
		),
	}


def _folded_encoder_forward(
	arrays: Mapping[str, np.ndarray],
	x: torch.Tensor,
) -> torch.Tensor:
	functional = torch.nn.functional
	value = functional.conv1d(
		x,
		torch.from_numpy(arrays["encoder.features.0.weight"]),
		torch.from_numpy(arrays["encoder.features.0.bias"]),
		padding=2,
	)
	value = functional.relu(value)
	value = functional.max_pool1d(value, 2)
	value = functional.conv1d(
		value,
		torch.from_numpy(arrays["encoder.features.3.weight"]),
		torch.from_numpy(arrays["encoder.features.3.bias"]),
		padding=2,
	)
	value = functional.relu(value)
	value = functional.adaptive_avg_pool1d(value, 1).squeeze(-1)
	return functional.linear(
		value,
		torch.from_numpy(arrays["encoder.projection.weight"]),
		torch.from_numpy(arrays["encoder.projection.bias"]),
	)


def _deploy_head_from_checkpoint(
	path: Path,
	classifier_class: type[torch.nn.Module],
	initiator_node: str,
	head_prefix: str | None,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
	mapping = _load_torch_mapping(path)
	root, root_name = _unwrap_state_dict(mapping, "head_checkpoint")
	nested, nested_name = _nested_role_mapping(
		root,
		(
			"classifier_state_dict",
			"head_state_dict",
			"common_classifier_state_dict",
			"common_head_state_dict",
			"classifier",
			"head",
			"common_classifier",
			"common_head",
		),
		root_name,
	)
	container = nested if nested is not None else root
	source = nested_name if nested_name is not None else root_name
	prefixes = _explicit_or_auto_prefix(
		head_prefix,
		("", "module.") if nested is not None else (
			"classifier.",
			"head.",
			"module.classifier.",
			"module.head.",
			"model.classifier.",
			"model.head.",
			"common_classifier.",
			"common_head.",
			"",
		),
	)
	arrays, resolution, _ = _extract_role(
		container,
		HEAD_SPEC,
		"head",
		source,
		prefixes,
		strict_namespace=True,
	)
	state = {
		name: torch.from_numpy(arrays[artifact_key]).clone()
		for name, _, artifact_key in HEAD_SPEC
	}
	classifier = classifier_class(
		PULSE_EMBEDDING_DIM,
		PULSE_CLASS_COUNT,
		hidden_dims=list(PULSE_CLASSIFIER_HIDDEN_DIMS),
	)
	try:
		classifier.load_state_dict(state, strict=True)
	except Exception as exc:
		raise ArtifactError(f"deploy head checkpoint is incompatible: {exc}") from exc
	_validate_common_head_architecture(classifier)
	fingerprint = _classifier_fingerprint_full(classifier)
	return state, {
		"kind": "supplied_deploy_head_checkpoint",
		"initiator_node": initiator_node,
		"fingerprint_sha256_full": fingerprint,
		"fingerprint_sha256_12": fingerprint[:12],
		"torch_version": torch.__version__,
		"architecture": {
			"embedding_dim": PULSE_EMBEDDING_DIM,
			"num_classes": PULSE_CLASS_COUNT,
			"classifier_hidden_dims": list(PULSE_CLASSIFIER_HIDDEN_DIMS),
			"batch_norm": False,
			"parameter_count": 4550,
		},
		"source": _source_record(path),
	}, resolution


def _load_deploy_models(
	paths: Mapping[str, Path],
	initiator_node: str,
	deploy_head_policy: str | None,
	deploy_head_seed: int | None,
	head_checkpoint: Path | None,
	head_prefix: str | None,
	pulse_source_root: Path | None,
	pulse_config: Path | None,
	simulation_seed: int,
	deploy_train_fraction: float,
	deploy_val_fraction: float,
) -> tuple[
	dict[str, np.ndarray],
	dict[str, Any],
	dict[str, Any],
	torch.nn.Module,
	type[torch.nn.Module],
	Mapping[str, torch.Tensor],
]:
	encoder_class, classifier_class, make_head = _load_deploy_setup_definitions(
		paths["hhar_device_setup.py"]
	)
	try:
		encoder = encoder_class(input_channels=CHANNEL_COUNT, embedding_dim=PULSE_EMBEDDING_DIM)
	except Exception as exc:
		raise ArtifactError(f"could not instantiate deploy encoder: {exc}") from exc
	if not isinstance(encoder, torch.nn.Module):
		raise ArtifactError("deploy IMUEncoderTiny did not construct an nn.Module")
	bn_epsilons = _validate_deploy_encoder_architecture(encoder)

	checkpoint = _load_torch_mapping(paths["uci_har_encoder.pt"])
	checkpoint_state, checkpoint_source = _unwrap_state_dict(checkpoint, "deploy_checkpoint")
	expected_state = encoder.state_dict()
	if set(checkpoint_state) != set(expected_state):
		missing = sorted(set(expected_state) - set(checkpoint_state))
		unexpected = sorted(set(checkpoint_state) - set(expected_state))
		raise ArtifactError(
			"deploy encoder checkpoint schema mismatch: "
			f"missing={missing}, unexpected={unexpected}"
		)
	for name, expected in expected_state.items():
		value = checkpoint_state[name]
		if not torch.is_tensor(value):
			raise ArtifactError(f"{checkpoint_source}.{name} is not a torch.Tensor")
		if tuple(value.shape) != tuple(expected.shape) or value.dtype != expected.dtype:
			raise ArtifactError(
				f"{checkpoint_source}.{name} has shape/dtype {tuple(value.shape)}/{value.dtype}, "
				f"expected {tuple(expected.shape)}/{expected.dtype}"
			)
		if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
			raise ArtifactError(f"{checkpoint_source}.{name} contains NaN or infinity")
	for name in ("encoder.network.1.running_var", "encoder.network.5.running_var"):
		if bool(torch.any(checkpoint_state[name] < 0).item()):
			raise ArtifactError(f"{checkpoint_source}.{name} contains a negative variance")
	try:
		encoder.load_state_dict(checkpoint_state, strict=True)
	except Exception as exc:
		raise ArtifactError(f"could not load deploy encoder checkpoint: {exc}") from exc
	encoder.eval()
	encoder_arrays, folding = _fold_deploy_encoder(encoder)

	if head_checkpoint is not None:
		head_state, head_initialization, head_resolution = _deploy_head_from_checkpoint(
			head_checkpoint,
			classifier_class,
			initiator_node,
			head_prefix,
		)
		head_resolution["origin"] = head_initialization["kind"]
	elif deploy_head_policy == "node-seeded":
		if deploy_head_seed is None:
			raise AssertionError("deploy node-head seed was not validated")
		head_state, head_initialization = _deploy_head_from_setup(
			classifier_class,
			make_head,
			deploy_head_seed,
			paths["hhar_device_setup.py"],
			initiator_node,
		)
		head_resolution = {
			"container": "deploy_hhar_device_setup._make_head",
			"prefix": "",
			"origin": head_initialization["kind"],
		}
	elif deploy_head_policy == "pulse-common":
		if pulse_source_root is None or pulse_config is None:
			raise AssertionError("deploy PULSE common-head sources were not validated")
		head_state, head_initialization = _canonical_common_head(
			pulse_source_root,
			simulation_seed,
			pulse_config,
			paths["uci_har_encoder.pt"],
			allow_deploy_encoder_override=True,
			deploy_train_fraction=deploy_train_fraction,
			deploy_val_fraction=deploy_val_fraction,
		)
		head_initialization["deploy_binding"] = {
			"initiator_node": initiator_node,
			"policy": "all participating nodes begin from init_id='common'",
		}
		head_resolution = {
			"container": "canonical_seeded_common_head",
			"prefix": "",
			"origin": head_initialization["kind"],
		}
	else:
		raise AssertionError("deploy head policy was not validated")

	head_arrays = {
		artifact_key: head_state[name].detach().cpu().to(torch.float32).contiguous().numpy().copy()
		for name, _, artifact_key in HEAD_SPEC
	}
	arrays = {**encoder_arrays, **head_arrays}
	arrays["head.flat"] = np.concatenate(
		[arrays[key].reshape(-1) for key in HEAD_FLAT_ORDER]
	).astype(np.float32, copy=False)
	resolution = {
		"encoder": {
			"container": checkpoint_source,
			"source_names": list(expected_state),
			"source_architecture": "deploy.IMUEncoderTiny Conv1d-BatchNorm-ReLU",
			"batch_norm_epsilons": list(bn_epsilons),
			"folding": folding,
		},
		"head": head_resolution,
	}
	return arrays, resolution, head_initialization, encoder, classifier_class, head_state


def _selected_window_fold_equivalence(
	source_encoder: torch.nn.Module,
	head_state: Mapping[str, torch.Tensor],
	model_arrays: Mapping[str, np.ndarray],
	replay_x: np.ndarray,
	source_indices: np.ndarray,
) -> dict[str, Any]:
	source_encoder.eval()
	inputs = torch.from_numpy(np.ascontiguousarray(replay_x.reshape(-1, 3, 128)))
	functional = torch.nn.functional

	def head_forward(embeddings: torch.Tensor) -> torch.Tensor:
		hidden = functional.relu(functional.linear(
			embeddings,
			head_state["feature_layers.0.weight"],
			head_state["feature_layers.0.bias"],
		))
		return functional.linear(
			hidden,
			head_state["output_layer.weight"],
			head_state["output_layer.bias"],
		)

	with torch.no_grad():
		source_embeddings = source_encoder(inputs)
		folded_embeddings = _folded_encoder_forward(model_arrays, inputs)
		source_logits = head_forward(source_embeddings)
		folded_logits = head_forward(folded_embeddings)
	if not torch.allclose(
		source_embeddings, folded_embeddings, rtol=DEPLOY_FOLD_RTOL, atol=DEPLOY_FOLD_ATOL
	):
		raise ArtifactError(
			"folded deploy encoder does not reproduce source embeddings on the selected replay"
		)
	if not torch.allclose(
		source_logits, folded_logits, rtol=DEPLOY_FOLD_RTOL, atol=DEPLOY_FOLD_ATOL
	):
		raise ArtifactError(
			"folded deploy encoder does not reproduce source logits on the selected replay"
		)
	source_predictions = source_logits.argmax(dim=1)
	folded_predictions = folded_logits.argmax(dim=1)
	if not torch.equal(source_predictions, folded_predictions):
		raise ArtifactError("source and folded deploy models disagree on selected predictions")

	def comparison(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
		difference = (first - second).abs()
		cosine = torch.nn.functional.cosine_similarity(
			first.reshape(1, -1), second.reshape(1, -1), dim=1
		)
		return {
			"max_abs_error": float(difference.max().item()),
			"mean_abs_error": float(difference.mean().item()),
			"cosine_similarity": float(cosine.item()),
			"source_fp32le_sha256": hashlib.sha256(
				first.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
			).hexdigest(),
			"folded_fp32le_sha256": hashlib.sha256(
				second.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
			).hexdigest(),
		}

	return {
		"passed": True,
		"reference": "supplied_hhar_device_setup_eval",
		"candidate": "firmware_canonical_encoder_from_folded_tensors",
		"head_evaluation": (
			"explicit F.linear-ReLU-F.linear from validated state; deploy MLPClassifier "
			"is a construction/export class and does not implement forward"
		),
		"selected_window_count": int(inputs.shape[0]),
		"selected_source_indices": source_indices.tolist(),
		"rtol": DEPLOY_FOLD_RTOL,
		"atol": DEPLOY_FOLD_ATOL,
		"prediction_mismatch_count": 0,
		"embeddings": comparison(source_embeddings, folded_embeddings),
		"logits": comparison(source_logits, folded_logits),
	}


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
	if not path.is_file():
		raise ArtifactError(
			f"checkpoint does not exist: {path}. The upstream PULSE repository "
			"does not bundle trained deployment weights; supply a real checkpoint."
		)
	try:
		loaded = torch.load(path, map_location="cpu", weights_only=True)
	except TypeError as exc:
		raise ArtifactError(
			"this exporter requires a PyTorch version that supports "
			"torch.load(..., weights_only=True)"
		) from exc
	except Exception as exc:
		raise ArtifactError(f"could not load checkpoint {path}: {exc}") from exc
	if not isinstance(loaded, Mapping):
		raise ArtifactError(
			f"checkpoint {path} must contain a state-dict mapping, got "
			f"{type(loaded).__name__}"
		)
	return loaded


def _unwrap_state_dict(mapping: Mapping[str, Any], source: str) -> tuple[Mapping[str, Any], str]:
	wrappers = [
		key for key in ("state_dict", "model_state_dict")
		if key in mapping and isinstance(mapping[key], Mapping)
	]
	if len(wrappers) > 1:
		raise ArtifactError(
			f"{source} has both state_dict and model_state_dict; keep one or use "
			"an unambiguous checkpoint"
		)
	if wrappers:
		key = wrappers[0]
		return mapping[key], f"{source}.{key}"
	return mapping, source


def _nested_role_mapping(
	mapping: Mapping[str, Any],
	keys: Sequence[str],
	source: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
	matches = [key for key in keys if key in mapping and isinstance(mapping[key], Mapping)]
	if len(matches) > 1:
		raise ArtifactError(
			f"{source} contains multiple candidate role mappings {matches}; "
			"retain exactly one"
		)
	if not matches:
		return None, None
	key = matches[0]
	return mapping[key], f"{source}.{key}"


def _tensor_keys(mapping: Mapping[str, Any]) -> set[str]:
	return {key for key, value in mapping.items() if isinstance(key, str) and torch.is_tensor(value)}


def _extract_role(
	mapping: Mapping[str, Any],
	spec: Sequence[tuple[str, tuple[int, ...], str]],
	role: str,
	source: str,
	prefixes: Sequence[str],
	strict_namespace: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], set[str]]:
	required_names = tuple(item[0] for item in spec)
	candidates = []
	for prefix in prefixes:
		keys = tuple(prefix + name for name in required_names)
		if all(key in mapping for key in keys):
			candidates.append((prefix, keys))
	if not candidates:
		preview = ", ".join(sorted(_tensor_keys(mapping))[:12]) or "<no tensors>"
		raise ArtifactError(
			f"{source} does not contain the exact {role} tensors. Expected "
			f"{list(required_names)} under one supported prefix; found {preview}"
		)
	if len(candidates) > 1:
		matched = [prefix or "<empty>" for prefix, _ in candidates]
		raise ArtifactError(
			f"{source} has ambiguous {role} prefixes {matched}; pass an explicit "
			f"--{role}-prefix"
		)

	prefix, source_keys = candidates[0]
	selected_keys = set(source_keys)
	if strict_namespace:
		tensors = _tensor_keys(mapping)
		if prefix:
			unexpected = sorted(key for key in tensors if key.startswith(prefix) and key not in selected_keys)
		else:
			unexpected = sorted(tensors - selected_keys)
		if unexpected:
			raise ArtifactError(
				f"{source} has unexpected {role} tensors under prefix "
				f"{prefix!r}: {unexpected}"
			)

	arrays: dict[str, np.ndarray] = {}
	source_dtypes: dict[str, str] = {}
	resolved_names: dict[str, str] = {}
	for (canonical_name, shape, artifact_key), source_key in zip(spec, source_keys):
		value = mapping[source_key]
		if not torch.is_tensor(value):
			raise ArtifactError(f"{source}.{source_key} is not a torch.Tensor")
		if not value.is_floating_point() or value.is_complex() or value.is_sparse:
			raise ArtifactError(
				f"{source}.{source_key} must be a dense real floating-point tensor"
			)
		if tuple(value.shape) != shape:
			raise ArtifactError(
				f"{source}.{source_key} has shape {tuple(value.shape)}, expected {shape}"
			)
		if not bool(torch.isfinite(value).all().item()):
			raise ArtifactError(f"{source}.{source_key} contains NaN or infinity")
		source_dtypes[canonical_name] = str(value.dtype)
		resolved_names[canonical_name] = source_key
		# clone() makes the exported snapshot independent of checkpoint storage.
		frozen = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
		frozen.requires_grad_(False)
		arrays[artifact_key] = frozen.numpy().copy()

	resolution = {
		"container": source,
		"prefix": prefix,
		"source_names": resolved_names,
		"source_dtypes": source_dtypes,
	}
	return arrays, resolution, selected_keys


def _explicit_or_auto_prefix(explicit: str | None, automatic: Sequence[str]) -> tuple[str, ...]:
	if explicit is not None:
		return (explicit,)
	return tuple(automatic)


def _extract_models(
	checkpoint: Mapping[str, Any],
	head_checkpoint: Mapping[str, Any] | None,
	encoder_prefix: str | None,
	head_prefix: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	primary, primary_name = _unwrap_state_dict(checkpoint, "checkpoint")
	encoder_nested, encoder_name = _nested_role_mapping(
		primary, ("encoder_state_dict", "encoder"), primary_name
	)
	encoder_container = encoder_nested if encoder_nested is not None else primary
	encoder_source = encoder_name if encoder_name is not None else primary_name

	if head_checkpoint is not None:
		head_root, head_root_name = _unwrap_state_dict(head_checkpoint, "head_checkpoint")
		head_nested, head_name = _nested_role_mapping(
			head_root,
			(
				"classifier_state_dict",
				"head_state_dict",
				"common_classifier_state_dict",
				"common_head_state_dict",
				"classifier",
				"head",
				"common_classifier",
				"common_head",
			),
			head_root_name,
		)
		head_container = head_nested if head_nested is not None else head_root
		head_source = head_name if head_name is not None else head_root_name
	else:
		head_nested, head_name = _nested_role_mapping(
			primary,
			(
				"classifier_state_dict",
				"head_state_dict",
				"common_classifier_state_dict",
				"common_head_state_dict",
				"classifier",
				"head",
				"common_classifier",
				"common_head",
			),
			primary_name,
		)
		head_container = head_nested if head_nested is not None else primary
		head_source = head_name if head_name is not None else primary_name

	encoder_is_nested = encoder_nested is not None
	head_is_nested = head_checkpoint is not None or head_nested is not None
	encoder_prefixes = _explicit_or_auto_prefix(
		encoder_prefix,
		("", "module.") if encoder_is_nested else
		("encoder.", "module.encoder.", "model.encoder.", ""),
	)
	head_prefixes = _explicit_or_auto_prefix(
		head_prefix,
		("", "module.") if head_is_nested else (
			"classifier.",
			"head.",
			"module.classifier.",
			"module.head.",
			"model.classifier.",
			"model.head.",
			"common_classifier.",
			"common_head.",
			"module.common_classifier.",
			"module.common_head.",
			"model.common_classifier.",
			"model.common_head.",
			"",
		),
	)

	try:
		encoder_arrays, encoder_resolution, encoder_keys = _extract_role(
			encoder_container,
			ENCODER_SPEC,
			"encoder",
			encoder_source,
			encoder_prefixes,
			strict_namespace=encoder_is_nested or head_checkpoint is not None,
		)
	except ArtifactError:
		raise

	try:
		head_arrays, head_resolution, head_keys = _extract_role(
			head_container,
			HEAD_SPEC,
			"head",
			head_source,
			head_prefixes,
			strict_namespace=head_is_nested,
		)
	except ArtifactError as exc:
		if head_checkpoint is None:
			raise ArtifactError(
				f"{exc}\nThe checkpoint may be an upstream encoder-only PULSE "
				"checkpoint. Such checkpoints do not contain the canonical common "
				"initial head; supply an exact saved head with --head-checkpoint, or "
				"explicitly provide --pulse-source-root, --pulse-config, and --simulation-seed."
			) from exc
		raise

	# A flat combined state dict is strict as a whole: every tensor must belong to
	# one of the two exact architectures. Nested containers are checked above.
	if encoder_container is primary and head_container is primary:
		unexpected = sorted(_tensor_keys(primary) - encoder_keys - head_keys)
		if unexpected:
			raise ArtifactError(
				f"{primary_name} contains tensors outside the exact encoder/common-head "
				f"schema: {unexpected}"
			)

	arrays = {**encoder_arrays, **head_arrays}
	arrays["head.flat"] = np.concatenate(
		[arrays[key].reshape(-1) for key in HEAD_FLAT_ORDER]
	).astype(np.float32, copy=False)
	if arrays["head.flat"].shape != (4550,):
		raise AssertionError("internal head flattening error")
	return arrays, {"encoder": encoder_resolution, "head": head_resolution}


def _finalize_replay_selection(
	selected_x: np.ndarray,
	selected_y: np.ndarray,
	selected_owner_ids: np.ndarray,
	indices: np.ndarray,
	selection: str,
	normalization: str,
	allow_owner_role_mismatch: bool,
	replay_split_id: str | None,
	replay_event_id: str | None,
	source_metadata: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	selected_x = np.ascontiguousarray(selected_x, dtype=np.float32).reshape(
		BATCH_COUNT, BATCH_SIZE, CHANNEL_COUNT, WINDOW_LENGTH
	)
	selected_y_i64 = np.ascontiguousarray(selected_y, dtype=np.int64).reshape(
		BATCH_COUNT, BATCH_SIZE
	)
	selected_owner_ids = np.ascontiguousarray(selected_owner_ids, dtype=np.str_).reshape(
		BATCH_COUNT, BATCH_SIZE
	)
	indices = np.ascontiguousarray(indices, dtype=np.int64).reshape(BATCH_COUNT, BATCH_SIZE)
	if not np.all(np.isfinite(selected_x)):
		raise ArtifactError("selected replay windows contain NaN or infinity")
	if np.any(selected_y_i64 < 0) or np.any(selected_y_i64 >= CLASS_COUNT):
		bad = np.unique(selected_y_i64[(selected_y_i64 < 0) | (selected_y_i64 >= CLASS_COUNT)])
		raise ArtifactError(f"labels must be in [0, 5], found {bad.tolist()}")
	if any(not owner for owner in selected_owner_ids.reshape(-1).tolist()):
		raise ArtifactError("selected replay contains an empty owner identifier")
	if np.any(indices < 0) or np.unique(indices).size != indices.size:
		raise ArtifactError("selected replay source indices must be distinct and non-negative")

	initiator_owners = np.unique(selected_owner_ids[:3]).tolist()
	responder_owners = np.unique(selected_owner_ids[3]).tolist()
	owner_role_binding_valid = (
		len(initiator_owners) == 1
		and len(responder_owners) == 1
		and initiator_owners[0] != responder_owners[0]
	)
	if not owner_role_binding_valid and not allow_owner_role_mismatch:
		raise ArtifactError(
			"replay ownership must bind scheduled_r_step_0, scheduled_r_step_1, and "
			"initiator_local to one initiator owner and responder_remote to one distinct "
			f"owner; got initiator owners {initiator_owners} and responder owners "
			f"{responder_owners}. Use --allow-owner-role-mismatch only for a diagnostic "
			"artifact that will not support the final privacy/role claim."
		)

	means_before = selected_x.mean(axis=-1, keepdims=True, dtype=np.float32)
	stds_before = selected_x.std(axis=-1, keepdims=True, dtype=np.float32)
	if normalization == "apply":
		denominators = np.maximum(stds_before, NORMALIZATION_EPSILON)
		replay_x = np.ascontiguousarray(
			(selected_x - means_before) / denominators,
			dtype=np.float32,
		)
		floored_count = int(np.count_nonzero(stds_before < NORMALIZATION_EPSILON))
	else:
		mean_error = np.abs(means_before)
		std_error = np.minimum(np.abs(stds_before), np.abs(stds_before - np.float32(1.0)))
		if np.any(mean_error > NORMALIZATION_VERIFY_TOLERANCE) or np.any(
			std_error > NORMALIZATION_VERIFY_TOLERANCE
		):
			raise ArtifactError(
				"--normalization verify failed: every channel/window must have mean "
				"near 0 and standard deviation near 1 (or 0 for a constant channel)"
			)
		replay_x = selected_x
		floored_count = int(np.count_nonzero(stds_before < NORMALIZATION_EPSILON))

	means_after = replay_x.mean(axis=-1, keepdims=True, dtype=np.float32)
	stds_after = replay_x.std(axis=-1, keepdims=True, dtype=np.float32)
	if normalization == "apply":
		expected_stds_after = stds_before / np.maximum(stds_before, NORMALIZATION_EPSILON)
	else:
		expected_stds_after = np.where(
			stds_before < NORMALIZATION_VERIFY_TOLERANCE,
			np.float32(0.0),
			np.float32(1.0),
		)
	max_mean_after = float(np.max(np.abs(means_after)))
	max_std_error_after = float(np.max(np.abs(stds_after - expected_stds_after)))
	if (
		max_mean_after > float(NORMALIZATION_VERIFY_TOLERANCE)
		or max_std_error_after > float(NORMALIZATION_VERIFY_TOLERANCE)
	):
		raise ArtifactError(
			"normalized replay validation failed: output channel/window statistics "
			"are outside the float32 tolerance"
		)

	arrays = {
		"replay.x": replay_x,
		"replay.y": selected_y_i64.astype(np.uint8),
		"replay.source_indices": indices,
		"replay.owner_ids": selected_owner_ids,
	}
	role_owners = {
		role: np.unique(selected_owner_ids[index]).tolist()
		for index, role in enumerate(REPLAY_ROLES)
	}
	metadata = {
		**source_metadata,
		"selection": selection,
		"source_indices": indices.tolist(),
		"ownership": {
			"selected_owner_ids": selected_owner_ids.tolist(),
			"role_owners": role_owners,
			"initiator_owners": initiator_owners,
			"responder_owners": responder_owners,
			"role_binding_valid": owner_role_binding_valid,
			"diagnostic_override_used": bool(
				allow_owner_role_mismatch and not owner_role_binding_valid
			),
			"requirement": (
				"the first three replay roles share one owner and the responder role "
				"has one distinct owner"
			),
		},
		"operator_context": {
			"split": {
				"value": replay_split_id,
				"source": "operator_supplied" if replay_split_id is not None else "not_supplied",
			},
			"event": {
				"value": replay_event_id,
				"source": "operator_supplied" if replay_event_id is not None else "not_supplied",
			},
		},
		"normalization": {
			"mode": normalization,
			"formula": "(x - mean(x, axis=time)) / max(std(x, axis=time), 1e-6)",
			"epsilon": float(NORMALIZATION_EPSILON),
			"verify_tolerance": float(NORMALIZATION_VERIFY_TOLERANCE),
			"floored_channel_windows": floored_count,
			"max_abs_output_mean": max_mean_after,
			"max_output_std_error": max_std_error_after,
		},
		"label_histogram": np.bincount(
			selected_y_i64.reshape(-1), minlength=CLASS_COUNT
		).tolist(),
	}
	return arrays, metadata


def _load_replay_data(
	path: Path,
	x_key: str,
	y_key: str,
	node_ids_key: str,
	indices_key: str | None,
	start_index: int,
	normalization: str,
	allow_owner_role_mismatch: bool,
	replay_split_id: str | None,
	replay_event_id: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	if not path.is_file():
		raise ArtifactError(f"input data archive does not exist: {path}")
	try:
		with np.load(path, allow_pickle=False) as archive:
			available = tuple(archive.files)
			if x_key not in archive or y_key not in archive or node_ids_key not in archive:
				raise ArtifactError(
					f"data archive must contain {x_key!r}, {y_key!r}, and "
					f"{node_ids_key!r}; "
					f"available keys are {available}"
				)
			x = np.asarray(archive[x_key])
			y = np.asarray(archive[y_key])
			node_ids = np.asarray(archive[node_ids_key])
			if indices_key is not None:
				if indices_key not in archive:
					raise ArtifactError(
						f"indices key {indices_key!r} is absent; available keys are {available}"
					)
				indices_value = np.asarray(archive[indices_key])
			else:
				indices_value = None
	except ArtifactError:
		raise
	except Exception as exc:
		raise ArtifactError(f"could not read data archive {path}: {exc}") from exc

	if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNEL_COUNT, WINDOW_LENGTH):
		raise ArtifactError(
			f"{x_key!r} must have shape [N, 3, 128], got {tuple(x.shape)}"
		)
	if x.dtype.kind not in "fiu":
		raise ArtifactError(f"{x_key!r} must be a numeric array, got {x.dtype}")
	if y.ndim != 1 or y.shape[0] != x.shape[0]:
		raise ArtifactError(
			f"{y_key!r} must have shape [N] matching x; got {tuple(y.shape)}"
		)
	if y.dtype.kind not in "iu" or y.dtype.kind == "b":
		raise ArtifactError(f"{y_key!r} must have an integer dtype, got {y.dtype}")
	if node_ids.ndim != 1 or node_ids.shape[0] != x.shape[0]:
		raise ArtifactError(
			f"{node_ids_key!r} must have shape [N] matching x; got {tuple(node_ids.shape)}"
		)
	if node_ids.dtype.kind not in "USiu" or node_ids.dtype.kind == "b":
		raise ArtifactError(
			f"{node_ids_key!r} must contain string or integer owner identifiers, "
			f"got {node_ids.dtype}"
		)
	try:
		owner_ids = np.asarray(node_ids, dtype=np.str_)
	except (TypeError, ValueError, UnicodeError) as exc:
		raise ArtifactError(f"could not convert {node_ids_key!r} to owner strings: {exc}") from exc
	if any(not owner for owner in owner_ids.tolist()):
		raise ArtifactError(f"{node_ids_key!r} contains an empty owner identifier")
	if start_index < 0:
		raise ArtifactError("--start-index must be non-negative")

	if indices_value is None:
		stop = start_index + BATCH_COUNT * BATCH_SIZE
		if stop > x.shape[0]:
			raise ArtifactError(
				f"need 64 samples from start index {start_index}, but {x_key!r} "
				f"contains only {x.shape[0]}"
			)
		indices = np.arange(start_index, stop, dtype=np.int64).reshape(BATCH_COUNT, BATCH_SIZE)
		selection = "contiguous"
	else:
		if start_index != 0:
			raise ArtifactError("--start-index cannot be combined with --indices-key")
		if indices_value.dtype.kind not in "iu" or indices_value.dtype.kind == "b":
			raise ArtifactError(f"{indices_key!r} must have an integer dtype")
		if indices_value.size != BATCH_COUNT * BATCH_SIZE:
			raise ArtifactError(
				f"{indices_key!r} must contain exactly 64 indices, got {indices_value.size}"
			)
		indices = indices_value.astype(np.int64, copy=False).reshape(BATCH_COUNT, BATCH_SIZE)
		if np.any(indices < 0) or np.any(indices >= x.shape[0]):
			raise ArtifactError(f"{indices_key!r} contains an out-of-range sample index")
		if np.unique(indices).size != indices.size:
			raise ArtifactError(f"{indices_key!r} must select 64 distinct samples")
		selection = f"archive_key:{indices_key}"

	flat_indices = indices.reshape(-1)
	source_metadata = {
		"source_x_key": x_key,
		"source_y_key": y_key,
		"source_node_ids_key": node_ids_key,
		"source_x_shape": list(x.shape),
		"source_x_dtype": str(x.dtype),
		"source_y_dtype": str(y.dtype),
		"source_node_ids_dtype": str(node_ids.dtype),
	}
	return _finalize_replay_selection(
		x[flat_indices],
		y[flat_indices],
		owner_ids[flat_indices],
		indices,
		selection,
		normalization,
		allow_owner_role_mismatch,
		replay_split_id,
		replay_event_id,
		source_metadata,
	)


def _read_deploy_manifest(
	path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
	try:
		with path.open("r", encoding="utf-8-sig", newline="") as source:
			reader = csv.DictReader(source)
			if tuple(reader.fieldnames or ()) != DEPLOY_MANIFEST_COLUMNS:
				raise ArtifactError(
					f"deploy manifest columns are {reader.fieldnames}, expected "
					f"{list(DEPLOY_MANIFEST_COLUMNS)}"
				)
			owners: list[str] = []
			labels: list[int] = []
			splits: list[str] = []
			for row_number, row in enumerate(reader):
				try:
					sample_index = int(row["sample_index"])
					label = int(row["label"])
					timestamp = float(row["timestamp"])
					metadata = json.loads(row["metadata_json"])
				except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
					raise ArtifactError(
						f"deploy manifest row {row_number + 2} contains an invalid value: {exc}"
					) from exc
				if sample_index != row_number:
					raise ArtifactError(
						"deploy manifest sample_index must be unique, contiguous, and equal "
						f"to HDF5 row order; row {row_number + 2} has {sample_index}"
					)
				owner = str(row["subject_id"]).strip()
				split = str(row["split"]).strip()
				if (
					not owner
					or not split
					or not str(row["session_id"]).strip()
					or not str(row["device_id"]).strip()
					or not math.isfinite(timestamp)
					or label < 0
					or label >= CLASS_COUNT
					or not isinstance(metadata, Mapping)
				):
					raise ArtifactError(
						f"deploy manifest row {row_number + 2} violates the HHAR schema"
					)
				owners.append(owner)
				labels.append(label)
				splits.append(split)
	except ArtifactError:
		raise
	except Exception as exc:
		raise ArtifactError(f"could not read deploy manifest {path}: {exc}") from exc
	if not owners:
		raise ArtifactError("deploy manifest contains no samples")
	owner_array = np.asarray(owners, dtype=np.str_)
	label_array = np.asarray(labels, dtype=np.int64)
	split_array = np.asarray(splits, dtype=np.str_)
	return owner_array, label_array, split_array, {
		"columns": list(DEPLOY_MANIFEST_COLUMNS),
		"row_count": len(owners),
		"owners": sorted(np.unique(owner_array).tolist()),
		"splits": sorted(np.unique(split_array).tolist()),
		"label_histogram": np.bincount(label_array, minlength=CLASS_COUNT).tolist(),
		"sample_index_contract": "zero-based contiguous HDF5 row index",
	}


def _pulse_train_splits_from_manifest(
	owner_ids: np.ndarray,
	split_ids: np.ndarray,
	source_split: str,
	simulation_seed: int,
	train_fraction: float,
	val_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	if (
		not math.isfinite(train_fraction)
		or not math.isfinite(val_fraction)
		or train_fraction <= 0.0
		or val_fraction <= 0.0
		or train_fraction + val_fraction >= 1.0
	):
		raise ArtifactError(
			"deploy train/validation fractions must be positive and leave a test split"
		)
	mask = split_ids == source_split
	if not np.any(mask):
		raise ArtifactError(
			f"deploy source split {source_split!r} is unavailable; found "
			f"{sorted(np.unique(split_ids).tolist())}"
		)
	owner_order = sorted(np.unique(owner_ids[mask]).tolist())
	rng = np.random.default_rng(simulation_seed)
	train: dict[str, np.ndarray] = {}
	counts: dict[str, dict[str, int]] = {}
	for owner in owner_order:
		indices = np.flatnonzero(mask & (owner_ids == owner))
		rng.shuffle(indices)
		n_train = int(len(indices) * train_fraction)
		n_val = int(len(indices) * val_fraction)
		n_test = len(indices) - n_train - n_val
		if n_train == 0 or n_val == 0 or n_test == 0:
			raise ArtifactError(f"deploy owner {owner!r} has too few samples for all splits")
		train[owner] = np.ascontiguousarray(indices[:n_train], dtype=np.int64)
		counts[owner] = {
			"total": int(len(indices)),
			"train": n_train,
			"validation": n_val,
			"test": n_test,
		}
	return train, {
		"algorithm": "PULSE.data.make_node_splits",
		"algorithm_detail": (
			"one numpy.random.default_rng(seed), owners sorted lexicographically, "
			"in-place shuffle per owner, floor fraction counts"
		),
		"simulation_seed": simulation_seed,
		"source_split": source_split,
		"owner_order": owner_order,
		"train_fraction": train_fraction,
		"validation_fraction": val_fraction,
		"test_fraction": 1.0 - train_fraction - val_fraction,
		"counts": counts,
	}


def _load_deploy_replay(
	paths: Mapping[str, Path],
	initiator_node: str,
	responder_node: str,
	source_split: str,
	simulation_seed: int,
	train_fraction: float,
	val_fraction: float,
	normalization: str,
	replay_split_id: str | None,
	replay_event_id: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	try:
		import h5py  # type: ignore[import-not-found]
	except ImportError as exc:
		raise ArtifactError(
			"--deploy-bundle requires h5py; install the exporter dependencies with "
			"'python -m pip install h5py'"
		) from exc
	owners, manifest_labels, split_ids, manifest_metadata = _read_deploy_manifest(
		paths["manifest.csv"]
	)
	train, split_metadata = _pulse_train_splits_from_manifest(
		owners,
		split_ids,
		source_split,
		simulation_seed,
		train_fraction,
		val_fraction,
	)
	if len(split_metadata["owner_order"]) != 9:
		raise ArtifactError(
			"Section-V deploy replay requires exactly nine HHAR owners to match "
			"PULSE data.num_nodes=9"
		)
	if initiator_node not in train:
		raise ArtifactError(
			f"--initiator-node {initiator_node!r} is absent from split {source_split!r}"
		)
	if responder_node not in train:
		raise ArtifactError(
			f"--responder-node {responder_node!r} is absent from split {source_split!r}"
		)
	if initiator_node == responder_node:
		raise ArtifactError("deploy initiator and responder nodes must be distinct")
	if len(train[initiator_node]) < 3 * BATCH_SIZE or len(train[responder_node]) < BATCH_SIZE:
		raise ArtifactError("selected deploy owners do not have enough training windows")
	indices = np.concatenate((
		train[initiator_node][:3 * BATCH_SIZE],
		train[responder_node][:BATCH_SIZE],
	)).reshape(BATCH_COUNT, BATCH_SIZE)
	flat_indices = indices.reshape(-1)

	try:
		with h5py.File(paths["data.h5"], "r") as archive:
			if "inputs/imu" not in archive or "labels" not in archive:
				raise ArtifactError("deploy data.h5 must contain /inputs/imu and /labels")
			x_dataset = archive["inputs/imu"]
			y_dataset = archive["labels"]
			if tuple(x_dataset.shape) != (len(owners), CHANNEL_COUNT, WINDOW_LENGTH):
				raise ArtifactError(
					"deploy /inputs/imu must have shape "
					f"[{len(owners)},3,128], got {tuple(x_dataset.shape)}"
				)
			if np.dtype(x_dataset.dtype) != np.dtype(np.float32):
				raise ArtifactError(
					f"deploy /inputs/imu must be float32, got {x_dataset.dtype}"
				)
			if tuple(y_dataset.shape) != (len(owners),) or np.dtype(y_dataset.dtype).kind not in "iu":
				raise ArtifactError(
					f"deploy /labels must be an integer [{len(owners)}] vector"
				)
			h5_labels = np.asarray(y_dataset, dtype=np.int64)
			if not np.array_equal(h5_labels, manifest_labels):
				mismatches = np.flatnonzero(h5_labels != manifest_labels)
				raise ArtifactError(
					"deploy HDF5 labels do not match manifest labels; first mismatch at "
					f"sample_index {int(mismatches[0])}"
				)
			selected_x = np.stack(
				[np.asarray(x_dataset[int(index)], dtype=np.float32) for index in flat_indices]
			)
	except ArtifactError:
		raise
	except Exception as exc:
		raise ArtifactError(f"could not read deploy HDF5 {paths['data.h5']}: {exc}") from exc

	source_metadata = {
		"source_format": "hhar_deploy_hdf5_plus_manifest",
		"source_x_key": "/inputs/imu",
		"source_y_key": "/labels",
		"source_node_ids_key": "manifest.csv:subject_id",
		"source_x_shape": [len(owners), CHANNEL_COUNT, WINDOW_LENGTH],
		"source_x_dtype": "float32",
		"source_y_dtype": str(h5_labels.dtype),
		"source_node_ids_dtype": str(owners.dtype),
		"manifest_validation": manifest_metadata,
		"split_derivation": split_metadata,
		"selection_policy": {
			"initiator_node": initiator_node,
			"responder_node": responder_node,
			"initiator_train_prefix_count": 3 * BATCH_SIZE,
			"responder_train_prefix_count": BATCH_SIZE,
			"observed_event": False,
			"interpretation": (
				"deterministic microbenchmark replay selected from the PULSE train split; "
				"not a logged contact or observed event"
			),
		},
	}
	return _finalize_replay_selection(
		selected_x,
		manifest_labels[flat_indices],
		owners[flat_indices],
		indices,
		"pulse_seeded_train_prefix",
		normalization,
		False,
		replay_split_id,
		replay_event_id,
		source_metadata,
	)


def _payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
	digest = hashlib.sha256()
	for key in sorted(PAYLOAD_KEYS):
		array = np.ascontiguousarray(arrays[key])
		digest.update(key.encode("utf-8"))
		digest.update(b"\0")
		digest.update(array.dtype.str.encode("ascii"))
		digest.update(b"\0")
		digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
		digest.update(b"\0")
		digest.update(array.tobytes(order="C"))
	return digest.hexdigest()


def _validate_artifact_arrays(arrays: Mapping[str, np.ndarray]) -> None:
	expected_shapes = {
		**{artifact_key: shape for _, shape, artifact_key in ENCODER_SPEC + HEAD_SPEC},
		"head.flat": (4550,),
		"replay.x": (BATCH_COUNT, BATCH_SIZE, CHANNEL_COUNT, WINDOW_LENGTH),
		"replay.y": (BATCH_COUNT, BATCH_SIZE),
		"replay.source_indices": (BATCH_COUNT, BATCH_SIZE),
		"replay.owner_ids": (BATCH_COUNT, BATCH_SIZE),
	}
	missing = sorted(set(expected_shapes) - set(arrays))
	if missing:
		raise ArtifactError(f"internal artifact is missing arrays: {missing}")
	for key, shape in expected_shapes.items():
		array = arrays[key]
		if array.shape != shape:
			raise ArtifactError(f"artifact {key} has shape {array.shape}, expected {shape}")
		expected_dtype = np.float32
		if key == "replay.y":
			expected_dtype = np.uint8
		elif key == "replay.source_indices":
			expected_dtype = np.int64
		elif key == "replay.owner_ids":
			if array.dtype.kind != "U":
				raise ArtifactError(
					f"artifact {key} has dtype {array.dtype}, expected a Unicode string dtype"
				)
			if any(not owner for owner in array.reshape(-1).tolist()):
				raise ArtifactError(f"artifact {key} contains an empty owner identifier")
			continue
		if array.dtype != expected_dtype:
			raise ArtifactError(
				f"artifact {key} has dtype {array.dtype}, expected {np.dtype(expected_dtype)}"
			)
		if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
			raise ArtifactError(f"artifact {key} contains NaN or infinity")
	if np.any(arrays["replay.y"] >= CLASS_COUNT):
		raise ArtifactError("artifact replay.y contains a label outside [0, 5]")
	expected_flat = np.concatenate([arrays[key].reshape(-1) for key in HEAD_FLAT_ORDER])
	if not np.array_equal(arrays["head.flat"], expected_flat):
		raise ArtifactError("artifact head.flat is not exact W1,b1,W2,b2 order")


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			mode="w+b", suffix=".npz", prefix=path.name + ".", dir=path.parent, delete=False
		) as output:
			temporary = Path(output.name)
			np.savez_compressed(output, **arrays)
			output.flush()
			os.fsync(output.fileno())
		os.replace(temporary, path)
	except Exception:
		if temporary is not None:
			try:
				temporary.unlink(missing_ok=True)
			except OSError:
				pass
		raise


def _atomic_save_json(path: Path, document: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			mode="w", encoding="utf-8", newline="\n", suffix=".json",
			prefix=path.name + ".", dir=path.parent, delete=False
		) as output:
			temporary = Path(output.name)
			json.dump(document, output, indent=2, sort_keys=True)
			output.write("\n")
			output.flush()
			os.fsync(output.fileno())
		os.replace(temporary, path)
	except Exception:
		if temporary is not None:
			try:
				temporary.unlink(missing_ok=True)
			except OSError:
				pass
		raise


def _source_record(path: Path) -> dict[str, Any]:
	return {
		"path": str(path.resolve()),
		"size_bytes": path.stat().st_size,
		"sha256": _sha256_file(path),
	}


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Export exact PULSE HAR encoder/head tensors and four normalized batch-16 "
			"replay batches from either legacy NPZ/checkpoint inputs or a PULSE deploy bundle."
		),
		epilog=(
			"Direct deploy-bundle example:\n"
			"  python tools/export_pulse_artifact.py --deploy-bundle /path/to/PULSE/deploy \\\n"
			"      --deploy-head-policy pulse-common --pulse-source-root /path/to/PULSE \\\n"
			"      --pulse-config /path/to/PULSE/configs/community_hhar_pulse.yaml \\\n"
			"      --simulation-seed 7 --initiator-node a --responder-node b \\\n"
			"      --deploy-source-split simulation --deploy-train-fraction 0.70 \\\n"
			"      --deploy-val-fraction 0.15 --output pulse_artifact.npz\n\n"
			"Combined/saved-head example:\n"
			"  python tools/export_pulse_artifact.py --checkpoint model.pt \\\n"
			"      --data processed_hhar.npz --output pulse_artifact.npz\n\n"
			"Canonical PULSE encoder-only example:\n"
			"  python tools/export_pulse_artifact.py --checkpoint uci_encoder.pt \\\n"
			"      --pulse-source-root /path/to/PULSE \\\n"
			"      --pulse-config /path/to/PULSE/configs/community_hhar_pulse.yaml \\\n"
			"      --simulation-seed 7 \\\n"
			"      --data processed_hhar.npz --output pulse_artifact.npz\n\n"
			"The seeded path is never implicit: all three provenance flags are required."
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	input_group = parser.add_mutually_exclusive_group(required=True)
	input_group.add_argument("--checkpoint", type=Path,
			help=(
				"combined model checkpoint, or encoder-only checkpoint with either "
				"--head-checkpoint or the explicit canonical seeded-head flags"
			))
	input_group.add_argument(
		"--deploy-bundle",
		type=Path,
		help=(
			"directory containing data.h5, manifest.csv, hhar_device_setup.py, and "
			"uci_har_encoder.pt"
		),
	)
	parser.add_argument("--head-checkpoint", type=Path,
			help=(
			"separate exact common-initial or declared head-snapshot state dict; also "
			"an explicit alternative to a deploy head policy"
		))
	parser.add_argument(
		"--pulse-source-root",
		type=Path,
		help=(
			"exact upstream PULSE source root containing models.py and utils/commons.py; "
			"requires --pulse-config and --simulation-seed and is mutually exclusive "
			"with --head-checkpoint"
		),
	)
	parser.add_argument(
		"--pulse-config",
		type=Path,
		help=(
			"exact controlled-evaluation YAML inside --pulse-source-root; required for "
			"canonical seeded-head export"
		),
	)
	parser.add_argument(
		"--simulation-seed",
		type=int,
		help=(
			"simulation seed used for deterministic deploy splits and, with the "
			"pulse-common policy, PULSE's init_id='common' head; legacy canonical-head "
			"mode requires --pulse-source-root and --pulse-config"
		),
	)
	parser.add_argument("--data", type=Path,
			help="processed PULSE NPZ containing x:[N,3,128], y:[N], and node_ids:[N]")
	parser.add_argument(
		"--deploy-head-policy",
		choices=("pulse-common", "node-seeded"),
		help=(
			"deploy head origin: canonical PULSE init_id='common', or the physical-device "
			"setup script's node-seeded head"
		),
	)
	parser.add_argument(
		"--deploy-head-seed",
		type=int,
		help="base/effective seed for --deploy-head-policy node-seeded at initiator position 0",
	)
	parser.add_argument("--initiator-node",
		help="deploy manifest subject_id owning the first three replay batches")
	parser.add_argument("--responder-node",
		help="distinct deploy manifest subject_id owning the responder replay batch")
	parser.add_argument("--deploy-source-split",
		help="deploy manifest split used to reconstruct PULSE node splits (for example simulation)")
	parser.add_argument("--deploy-train-fraction", type=float,
		help="PULSE train fraction used for deterministic deploy replay selection")
	parser.add_argument("--deploy-val-fraction", type=float,
		help="PULSE validation fraction used for deterministic deploy replay selection")
	parser.add_argument("--output", type=Path,
			help="output .npz path (required unless --validate-only)")
	parser.add_argument("--provenance", type=Path,
			help="provenance JSON path (default: OUTPUT with .provenance.json suffix)")
	parser.add_argument("--x-key", default="x", help="input window array key (default: x)")
	parser.add_argument("--y-key", default="y", help="input label array key (default: y)")
	parser.add_argument(
		"--node-ids-key",
		default="node_ids",
		help="input per-window owner array key (default: node_ids)",
	)
	parser.add_argument("--indices-key",
			help="optional input key containing 64 distinct indices, shaped [4,16] or [64]")
	parser.add_argument("--start-index", type=int, default=0,
			help="first of 64 contiguous samples when --indices-key is absent (default: 0)")
	parser.add_argument("--encoder-prefix",
			help="explicit state-dict prefix before canonical encoder names")
	parser.add_argument("--head-prefix",
			help="explicit state-dict prefix before canonical head names")
	parser.add_argument(
		"--normalization", choices=("apply", "verify"), default="verify",
		help=(
			"verify canonical normalization in an already-processed PULSE archive (default), "
			"or explicitly apply it to raw windows"
		),
	)
	parser.add_argument(
		"--allow-owner-role-mismatch",
		action="store_true",
		help=(
			"diagnostic only: permit replay owners that do not bind the first three roles "
			"to one initiator and the fourth role to a distinct responder"
		),
	)
	parser.add_argument(
		"--replay-split-id",
		help="operator-supplied identifier for the simulation/data split used by the replay",
	)
	parser.add_argument(
		"--replay-event-id",
		help="operator-supplied identifier for the encounter/event represented by the replay",
	)
	parser.add_argument("--validate-only", action="store_true",
			help="load and validate all inputs and print the schema summary without writing files")
	return parser


def _validate_paths(args: argparse.Namespace) -> None:
	deploy_mode = args.deploy_bundle is not None
	seeded_source = args.pulse_source_root is not None
	seeded_config = args.pulse_config is not None
	seeded_seed = args.simulation_seed is not None
	if deploy_mode:
		if args.data is not None or args.encoder_prefix is not None or args.indices_key is not None:
			raise ArtifactError(
				"--deploy-bundle is mutually exclusive with --data, --encoder-prefix, and --indices-key"
			)
		if args.start_index != 0 or args.allow_owner_role_mismatch:
			raise ArtifactError(
				"--start-index and --allow-owner-role-mismatch are legacy diagnostic options"
			)
		for flag, value in (
			("--simulation-seed", args.simulation_seed),
			("--initiator-node", args.initiator_node),
			("--responder-node", args.responder_node),
			("--deploy-source-split", args.deploy_source_split),
			("--deploy-train-fraction", args.deploy_train_fraction),
			("--deploy-val-fraction", args.deploy_val_fraction),
		):
			if value is None:
				raise ArtifactError(f"{flag} is required with --deploy-bundle")
		if args.simulation_seed < 0:
			raise ArtifactError("--simulation-seed must be non-negative")
		if not args.initiator_node.strip() or not args.responder_node.strip():
			raise ArtifactError("deploy node identifiers must not be empty")
		if args.initiator_node == args.responder_node:
			raise ArtifactError("deploy initiator and responder nodes must be distinct")
		if not args.deploy_source_split.strip():
			raise ArtifactError("--deploy-source-split must not be empty")
		if args.head_checkpoint is not None:
			if args.deploy_head_policy is not None or args.deploy_head_seed is not None:
				raise ArtifactError(
					"--head-checkpoint is mutually exclusive with deploy head policy/seed options"
				)
			if seeded_source or seeded_config:
				raise ArtifactError(
					"deploy --head-checkpoint is mutually exclusive with PULSE common-head sources"
				)
		elif args.deploy_head_policy == "pulse-common":
			if not seeded_source or not seeded_config:
				raise ArtifactError(
					"--deploy-head-policy pulse-common requires --pulse-source-root and --pulse-config"
				)
			if args.deploy_head_seed is not None or args.head_prefix is not None:
				raise ArtifactError(
					"PULSE common-head deploy export cannot use --deploy-head-seed or --head-prefix"
				)
		elif args.deploy_head_policy == "node-seeded":
			if args.deploy_head_seed is None:
				raise ArtifactError(
					"--deploy-head-policy node-seeded requires --deploy-head-seed"
				)
			if seeded_source or seeded_config or args.head_prefix is not None:
				raise ArtifactError(
					"node-seeded deploy head is mutually exclusive with PULSE sources and --head-prefix"
				)
		else:
			raise ArtifactError(
				"--deploy-bundle requires --deploy-head-policy or --head-checkpoint"
			)
	else:
		if args.checkpoint is None or args.data is None:
			raise ArtifactError("legacy export requires both --checkpoint and --data")
		deploy_values = (
			args.deploy_head_policy,
			args.deploy_head_seed,
			args.initiator_node,
			args.responder_node,
			args.deploy_source_split,
			args.deploy_train_fraction,
			args.deploy_val_fraction,
		)
		if any(value is not None for value in deploy_values):
			raise ArtifactError("deploy-specific options require --deploy-bundle")
		if len({seeded_source, seeded_config, seeded_seed}) != 1:
			raise ArtifactError(
				"--pulse-source-root, --pulse-config, and --simulation-seed must be supplied together"
			)
		if args.head_checkpoint is not None and seeded_source:
			raise ArtifactError(
				"--head-checkpoint is mutually exclusive with --pulse-source-root, "
				"--pulse-config, and --simulation-seed"
			)
		if seeded_source and args.head_prefix is not None:
			raise ArtifactError(
				"--head-prefix cannot be used with the canonical seeded common-head path"
			)
	for flag, value in (
		("--replay-split-id", args.replay_split_id),
		("--replay-event-id", args.replay_event_id),
	):
		if value is not None and not value.strip():
			raise ArtifactError(f"{flag} must not be empty or whitespace")
	if not args.validate_only and args.output is None:
		raise ArtifactError("--output is required unless --validate-only is used")
	if args.validate_only and args.provenance is not None:
		raise ArtifactError("--provenance cannot be used with --validate-only")
	if args.output is None:
		return
	output = args.output.resolve()
	if deploy_mode:
		paths = _resolve_deploy_bundle(args.deploy_bundle)
		inputs = {path.resolve() for name, path in paths.items() if name != "root"}
		try:
			output.relative_to(paths["root"])
		except ValueError:
			pass
		else:
			raise ArtifactError("deploy output must be outside the immutable --deploy-bundle")
	else:
		inputs = {args.checkpoint.resolve(), args.data.resolve()}
	if args.head_checkpoint is not None:
		inputs.add(args.head_checkpoint.resolve())
	if args.pulse_config is not None:
		inputs.add(args.pulse_config.resolve())
	if output in inputs:
		raise ArtifactError("--output must not overwrite a checkpoint or data input")
	provenance = args.provenance
	if provenance is not None and provenance.resolve() in inputs | {output}:
		raise ArtifactError("--provenance must be distinct from inputs and --output")
	if provenance is not None and deploy_mode:
		try:
			provenance.resolve().relative_to(paths["root"])
		except ValueError:
			pass
		else:
			raise ArtifactError("deploy provenance must be outside the immutable --deploy-bundle")


def main(argv: Sequence[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	try:
		_validate_paths(args)
		fold_equivalence: dict[str, Any] | None = None
		if args.deploy_bundle is not None:
			deploy_paths = _resolve_deploy_bundle(args.deploy_bundle)
			(
				model_arrays,
				model_resolution,
				head_initialization,
				source_encoder,
				_classifier_class,
				head_state,
			) = _load_deploy_models(
				deploy_paths,
				args.initiator_node,
				args.deploy_head_policy,
				args.deploy_head_seed,
				args.head_checkpoint,
				args.head_prefix,
				args.pulse_source_root,
				args.pulse_config,
				args.simulation_seed,
				args.deploy_train_fraction,
				args.deploy_val_fraction,
			)
			replay_arrays, replay_metadata = _load_deploy_replay(
				deploy_paths,
				args.initiator_node,
				args.responder_node,
				args.deploy_source_split,
				args.simulation_seed,
				args.deploy_train_fraction,
				args.deploy_val_fraction,
				args.normalization,
				args.replay_split_id,
				args.replay_event_id,
			)
			fold_equivalence = _selected_window_fold_equivalence(
				source_encoder,
				head_state,
				model_arrays,
				replay_arrays["replay.x"],
				replay_arrays["replay.source_indices"],
			)
			model_resolution["encoder"]["selected_window_equivalence"] = fold_equivalence
			readme_path = deploy_paths["root"] / "README.md"
			sources = {
				"checkpoint": _source_record(deploy_paths["uci_har_encoder.pt"]),
				"head_checkpoint": (
					_source_record(args.head_checkpoint)
					if args.head_checkpoint is not None else None
				),
				"data": _source_record(deploy_paths["data.h5"]),
				"manifest": _source_record(deploy_paths["manifest.csv"]),
				"deploy_setup": _source_record(deploy_paths["hhar_device_setup.py"]),
				"deploy_readme": _source_record(readme_path) if readme_path.is_file() else None,
				"deploy_bundle": {"path": str(deploy_paths["root"])},
				"pulse_config": (
					_source_record(args.pulse_config) if args.pulse_config is not None else None
				),
				"pulse_source": (
					head_initialization.get("source")
					if args.deploy_head_policy == "pulse-common" else None
				),
			}
		else:
			checkpoint = _load_torch_mapping(args.checkpoint)
			if args.pulse_source_root is not None:
				head_checkpoint, head_initialization = _canonical_common_head(
					args.pulse_source_root,
					args.simulation_seed,
					args.pulse_config,
					args.checkpoint,
				)
			else:
				head_checkpoint = (
					_load_torch_mapping(args.head_checkpoint)
					if args.head_checkpoint is not None else None
				)
				head_initialization = {
					"kind": (
						"supplied_head_checkpoint"
						if args.head_checkpoint is not None else "head_from_combined_checkpoint"
					),
				}
			model_arrays, model_resolution = _extract_models(
				checkpoint,
				head_checkpoint,
				args.encoder_prefix,
				args.head_prefix,
			)
			if args.pulse_source_root is not None:
				model_resolution["head"]["container"] = "canonical_seeded_common_head"
				model_resolution["head"]["origin"] = head_initialization["kind"]
			replay_arrays, replay_metadata = _load_replay_data(
				args.data,
				args.x_key,
				args.y_key,
				args.node_ids_key,
				args.indices_key,
				args.start_index,
				args.normalization,
				args.allow_owner_role_mismatch,
				args.replay_split_id,
				args.replay_event_id,
			)
			sources = {
				"checkpoint": _source_record(args.checkpoint),
				"head_checkpoint": (
					_source_record(args.head_checkpoint)
					if args.head_checkpoint is not None else None
				),
				"data": _source_record(args.data),
				"manifest": None,
				"deploy_setup": None,
				"deploy_readme": None,
				"deploy_bundle": None,
				"pulse_config": (
					_source_record(args.pulse_config)
					if args.pulse_config is not None else None
				),
				"pulse_source": (
					head_initialization["source"]
					if args.pulse_source_root is not None else None
				),
			}
		arrays = {**model_arrays, **replay_arrays}
		_validate_artifact_arrays(arrays)
		payload_sha256 = _payload_sha256(arrays)

		summary = {
			"schema": SCHEMA_VERSION,
			"payload_sha256": payload_sha256,
			"head_flat_order": list(HEAD_FLAT_ORDER),
			"head_initialization": head_initialization,
			"model_resolution": model_resolution,
			"selected_window_fold_equivalence": fold_equivalence,
			"replay": replay_metadata,
			"arrays": {
				key: {"shape": list(value.shape), "dtype": str(value.dtype)}
				for key, value in arrays.items()
			},
		}
		if args.validate_only:
			print(json.dumps(summary, indent=2, sort_keys=True))
			return 0

		artifact_arrays = {
			"artifact.schema": np.asarray(SCHEMA_VERSION),
			"artifact.payload_sha256": np.asarray(payload_sha256),
			"head.flat_order": np.asarray(HEAD_FLAT_ORDER),
			"replay.roles": np.asarray(REPLAY_ROLES),
			**arrays,
		}
		_atomic_save_npz(args.output, artifact_arrays)
		artifact_file_sha256 = _sha256_file(args.output)
		provenance_path = args.provenance or args.output.with_suffix(".provenance.json")
		provenance = {
			"schema": SCHEMA_VERSION,
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"artifact": {
				"path": str(args.output.resolve()),
				"file_sha256": artifact_file_sha256,
				"payload_sha256": payload_sha256,
				"arrays": summary["arrays"],
			},
			"sources": sources,
			"model": {
				"resolution": model_resolution,
				"head_initialization": head_initialization,
				"head_flat_order": ["W1", "b1", "W2", "b2"],
				"head_flat_state_names": list(HEAD_FLAT_ORDER),
				"head_parameter_count": 4550,
			},
			"replay": replay_metadata,
			"tool": {
				"script": str(Path(__file__).resolve()),
				"python": platform.python_version(),
				"numpy": np.__version__,
				"torch": torch.__version__,
			},
		}
		_atomic_save_json(provenance_path, provenance)
		print(json.dumps({
			"artifact": str(args.output),
			"provenance": str(provenance_path),
			"file_sha256": artifact_file_sha256,
			"payload_sha256": payload_sha256,
		}, indent=2, sort_keys=True))
		return 0
	except ArtifactError as exc:
		parser.exit(2, f"error: {exc}\n")
	return 2


if __name__ == "__main__":
	raise SystemExit(main())
