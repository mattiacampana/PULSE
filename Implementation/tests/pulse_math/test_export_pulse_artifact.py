"""Focused tests for canonical PULSE common-head artifact export."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


FIRMWARE_ROOT = Path(__file__).resolve().parents[2]
EXPORTER_PATH = FIRMWARE_ROOT / "tools" / "export_pulse_artifact.py"
GENERATOR_PATH = FIRMWARE_ROOT / "tools" / "generate_pulse_fixture.py"
SPEC = importlib.util.spec_from_file_location("senswear_export_pulse_artifact", EXPORTER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import diagnostic
	raise RuntimeError(f"could not load {EXPORTER_PATH}")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
GENERATOR_SPEC = importlib.util.spec_from_file_location(
	"senswear_generate_pulse_fixture", GENERATOR_PATH
)
if GENERATOR_SPEC is None or GENERATOR_SPEC.loader is None:  # pragma: no cover
	raise RuntimeError(f"could not load {GENERATOR_PATH}")
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(generator)


MODELS_SOURCE = '''\
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dims=None, batch_norm=False):
        super().__init__()
        hidden_dims = hidden_dims or []
        layers = []
        in_features = embedding_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_features, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_features = width
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, num_classes)

    def forward(self, embeddings):
        return self.output_layer(self.feature_layers(embeddings))
'''


COMMONS_SOURCE = '''\
def make_seeded_classifier(
    embedding_dim, num_classes, hidden_dims, simulation_seed, node_id, batch_norm=False
):
    seed_material = f"{simulation_seed}:{node_id}".encode("utf-8")
    head_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng():
        torch.manual_seed(head_seed)
        return MLPClassifier(
            embedding_dim, num_classes, hidden_dims, batch_norm=batch_norm
        )


def classifier_fingerprint(classifier):
    digest = hashlib.sha256()
    for name, parameter in classifier.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:12]
'''


BAD_MODELS_SOURCE = '''\
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dims=None, batch_norm=False):
        super().__init__()
        self.feature_layers = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU())
        self.output_layer = nn.Linear(32, num_classes)
'''


DEPLOY_SETUP_SOURCE = '''\
import torch
from torch import nn


class _ConvBranch(nn.Module):
    def __init__(self, input_channels, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(64, output_dim)

    def forward(self, x):
        return self.projection(self.network(x).flatten(1))


class IMUEncoderTiny(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=64):
        super().__init__()
        self.encoder = _ConvBranch(input_channels, embedding_dim)

    def forward(self, imu):
        return self.encoder(imu)


class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or []
        layers = []
        in_features = embedding_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(in_features, width), nn.ReLU()))
            in_features = width
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, num_classes)


def _make_head(embedding_dim, num_classes, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MLPClassifier(embedding_dim, num_classes, hidden_dims=[64])
'''


class CanonicalHeadExportTests(unittest.TestCase):
	def setUp(self) -> None:
		self.temporary = tempfile.TemporaryDirectory(prefix="pulse_export_test_")
		self.root = Path(self.temporary.name)
		self.pulse_root = self.root / "PULSE"
		(self.pulse_root / "utils").mkdir(parents=True)
		(self.pulse_root / "models.py").write_text(MODELS_SOURCE, encoding="utf-8")
		(self.pulse_root / "utils" / "commons.py").write_text(
			COMMONS_SOURCE, encoding="utf-8"
		)
		self.encoder_checkpoint = self.root / "encoder.pt"
		self.data_path = self.root / "hhar.npz"
		self._write_encoder_checkpoint(self.encoder_checkpoint)
		self.pulse_config = self._write_pulse_config(self.pulse_root, self.encoder_checkpoint)
		x = np.zeros((64, 3, 128), dtype=np.float32)
		y = (np.arange(64, dtype=np.int64) % 6).astype(np.int64)
		node_ids = np.asarray(["initiator"] * 48 + ["responder"] * 16)
		np.savez_compressed(self.data_path, x=x, y=y, node_ids=node_ids)

	def tearDown(self) -> None:
		self.temporary.cleanup()

	@staticmethod
	def _write_encoder_checkpoint(path: Path) -> None:
		state = {}
		for index, (name, shape, _) in enumerate(exporter.ENCODER_SPEC, start=1):
			count = int(np.prod(shape))
			state[name] = (
				torch.arange(count, dtype=torch.float32).reshape(shape) / float(count + index)
			)
		torch.save(state, path)

	@staticmethod
	def _write_pulse_config(source_root: Path, checkpoint: Path) -> Path:
		config_dir = source_root / "configs"
		config_dir.mkdir(parents=True, exist_ok=True)
		config = config_dir / "community_hhar_pulse.yaml"
		relative_checkpoint = Path("..") / ".." / checkpoint.name
		config.write_text(
			f'''\
seeds: [7, 129, 56, 83]
model:
  checkpoint: {relative_checkpoint.as_posix()}
  freeze_encoder: true
  input_channels: 3
  embedding_dim: 64
  num_classes: 6
  classifier_hidden_dims: [64]
training:
  local_steps: 2
  batch_size: 16
  learning_rate: 0.01
algorithm:
  name: pulse
pulse:
  alpha_max: 0.5
  utility_momentum: 0.2
  initial_utility: 0.0
  ucb_exploration: 0.25
  norm_epsilon: 1.0e-12
''',
			encoding="utf-8",
		)
		return config

	def _write_deploy_bundle(self) -> tuple[Path, dict[str, Path]]:
		bundle = self.root / "deploy"
		bundle.mkdir()
		setup_path = bundle / "hhar_device_setup.py"
		setup_path.write_text(DEPLOY_SETUP_SOURCE, encoding="utf-8")
		encoder_class, _, _ = exporter._load_deploy_setup_definitions(setup_path)
		with torch.random.fork_rng(devices=[]):
			torch.manual_seed(991)
			encoder = encoder_class(input_channels=3, embedding_dim=64)
		with torch.no_grad():
			encoder.encoder.network[1].running_mean.copy_(torch.linspace(-0.2, 0.2, 32))
			encoder.encoder.network[1].running_var.copy_(torch.linspace(0.2, 1.2, 32))
			encoder.encoder.network[5].running_mean.copy_(torch.linspace(-0.1, 0.1, 64))
			encoder.encoder.network[5].running_var.copy_(torch.linspace(0.3, 1.3, 64))
		torch.save(encoder.state_dict(), bundle / "uci_har_encoder.pt")
		(bundle / "data.h5").write_bytes(b"test-placeholder")
		(bundle / "manifest.csv").write_text(
			",".join(exporter.DEPLOY_MANIFEST_COLUMNS) + "\n", encoding="utf-8"
		)
		return bundle, exporter._resolve_deploy_bundle(bundle)

	def test_seeded_common_head_is_deterministic_and_exact(self) -> None:
		torch.manual_seed(123456)
		rng_before = torch.random.get_rng_state().clone()
		first, first_provenance = exporter._canonical_common_head(
			self.pulse_root, 7, self.pulse_config, self.encoder_checkpoint
		)
		second, second_provenance = exporter._canonical_common_head(
			self.pulse_root, 7, self.pulse_config, self.encoder_checkpoint
		)
		self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
		self.assertEqual(tuple(first), tuple(item[0] for item in exporter.HEAD_SPEC))
		for name in first:
			self.assertTrue(torch.equal(first[name], second[name]))

		model_namespace: dict[str, object] = {}
		exec(compile(MODELS_SOURCE, "models.py", "exec"), model_namespace)
		with torch.random.fork_rng():
			torch.manual_seed(exporter._derived_common_head_seed(7))
			expected = model_namespace["MLPClassifier"](64, 6, [64], batch_norm=False)
		for name, value in expected.state_dict().items():
			self.assertTrue(torch.equal(first[name], value))

		expected_full = exporter._classifier_fingerprint_full(expected)
		self.assertEqual(first_provenance["fingerprint_sha256_full"], expected_full)
		self.assertEqual(first_provenance["fingerprint_sha256_12"], expected_full[:12])
		self.assertEqual(
			first_provenance["upstream_classifier_fingerprint_12"], expected_full[:12]
		)
		self.assertEqual(first_provenance, second_provenance)

	def test_seeded_flags_are_paired_and_mutually_exclusive(self) -> None:
		parser = exporter._build_parser()
		base = [
			"--checkpoint", str(self.encoder_checkpoint),
			"--data", str(self.data_path),
			"--validate-only",
		]
		with self.assertRaisesRegex(exporter.ArtifactError, "supplied together"):
			exporter._validate_paths(parser.parse_args(base + ["--simulation-seed", "7"]))
		with self.assertRaisesRegex(exporter.ArtifactError, "mutually exclusive"):
			exporter._validate_paths(parser.parse_args(base + [
				"--pulse-source-root", str(self.pulse_root),
				"--pulse-config", str(self.pulse_config),
				"--simulation-seed", "7",
				"--head-checkpoint", str(self.encoder_checkpoint),
			]))
		with self.assertRaisesRegex(exporter.ArtifactError, "must not overwrite"):
			exporter._validate_paths(parser.parse_args([
				"--checkpoint", str(self.encoder_checkpoint),
				"--pulse-source-root", str(self.pulse_root),
				"--pulse-config", str(self.pulse_config),
				"--simulation-seed", "7",
				"--data", str(self.data_path),
				"--output", str(self.pulse_config),
			]))

	def test_export_writes_exact_head_and_complete_provenance(self) -> None:
		artifact = self.root / "pulse.npz"
		provenance_path = self.root / "pulse.provenance.json"
		stdout = io.StringIO()
		with contextlib.redirect_stdout(stdout):
			status = exporter.main([
				"--checkpoint", str(self.encoder_checkpoint),
				"--pulse-source-root", str(self.pulse_root),
				"--pulse-config", str(self.pulse_config),
				"--simulation-seed", "7",
				"--data", str(self.data_path),
				"--output", str(artifact),
				"--provenance", str(provenance_path),
				"--replay-split-id", "seed7/initiator/train",
				"--replay-event-id", "community-seed7/event-42",
			])
		self.assertEqual(status, 0)
		self.assertTrue(json.loads(stdout.getvalue())["file_sha256"])

		expected, expected_provenance = exporter._canonical_common_head(
			self.pulse_root, 7, self.pulse_config, self.encoder_checkpoint
		)
		with np.load(artifact, allow_pickle=False) as exported:
			for name, _, artifact_key in exporter.HEAD_SPEC:
				np.testing.assert_array_equal(exported[artifact_key], expected[name].numpy())
			expected_flat = np.concatenate([
				exported[key].reshape(-1) for key in exporter.HEAD_FLAT_ORDER
			]).astype(np.float32, copy=False)
			np.testing.assert_array_equal(exported["head.flat"], expected_flat)
			np.testing.assert_array_equal(
				exported["replay.owner_ids"],
				np.asarray([["initiator"] * 16] * 3 + [["responder"] * 16]),
			)

		provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
		head = provenance["model"]["head_initialization"]
		self.assertEqual(head["kind"], "pulse_seeded_common_initial_head")
		self.assertEqual(head["simulation_seed"], 7)
		self.assertEqual(head["init_id"], "common")
		self.assertEqual(head["derived_torch_seed"], exporter._derived_common_head_seed(7))
		self.assertEqual(
			head["fingerprint_sha256_full"], expected_provenance["fingerprint_sha256_full"]
		)
		self.assertEqual(len(head["fingerprint_sha256_full"]), 64)
		self.assertEqual(len(head["fingerprint_sha256_12"]), 12)
		self.assertEqual(head["torch_version"], torch.__version__)
		self.assertEqual(head["architecture"], {
			"embedding_dim": 64,
			"num_classes": 6,
			"classifier_hidden_dims": [64],
			"batch_norm": False,
			"parameter_count": 4550,
		})
		self.assertIsNone(provenance["sources"]["head_checkpoint"])
		self.assertEqual(
			provenance["sources"]["pulse_config"]["sha256"],
			exporter._sha256_file(self.pulse_config),
		)
		source = provenance["sources"]["pulse_source"]
		self.assertEqual(source["models_py"]["sha256"], exporter._sha256_file(
			self.pulse_root / "models.py"
		))
		self.assertEqual(source["commons_py"]["sha256"], exporter._sha256_file(
			self.pulse_root / "utils" / "commons.py"
		))
		self.assertEqual(source["config"]["record"]["sha256"], exporter._sha256_file(
			self.pulse_config
		))
		self.assertEqual(source["config"]["selected_simulation_seed"], 7)
		self.assertTrue(source["config"]["supplied_checkpoint_sha256_match"])
		self.assertEqual(
			source["config"]["declared_checkpoint"]["sha256"],
			exporter._sha256_file(self.encoder_checkpoint),
		)
		self.assertEqual(source["config"]["validated_fields"]["model"], {
			"freeze_encoder": True,
			"input_channels": 3,
			"embedding_dim": 64,
			"num_classes": 6,
			"classifier_hidden_dims": [64],
		})
		self.assertIn("revision", source["git"])
		self.assertIn("dirty", source["git"])
		self.assertEqual(
			provenance["model"]["resolution"]["head"]["origin"],
			"pulse_seeded_common_initial_head",
		)
		ownership = provenance["replay"]["ownership"]
		self.assertTrue(ownership["role_binding_valid"])
		self.assertFalse(ownership["diagnostic_override_used"])
		self.assertEqual(ownership["initiator_owners"], ["initiator"])
		self.assertEqual(ownership["responder_owners"], ["responder"])
		self.assertEqual(provenance["replay"]["normalization"]["mode"], "verify")
		self.assertEqual(provenance["replay"]["operator_context"]["split"], {
			"value": "seed7/initiator/train",
			"source": "operator_supplied",
		})
		self.assertEqual(provenance["replay"]["operator_context"]["event"], {
			"value": "community-seed7/event-42",
			"source": "operator_supplied",
		})
		fixture_data, fixture_provenance = generator.load_artifact_fixture(artifact)
		self.assertEqual(fixture_data["samples"].shape, (4, 16, 3, 128))
		self.assertEqual(fixture_provenance["initiator_owner"], "initiator")
		self.assertEqual(fixture_provenance["responder_owner"], "responder")

	def test_bad_source_layout_and_model_schema_are_rejected(self) -> None:
		missing = self.root / "missing"
		missing.mkdir()
		with self.assertRaisesRegex(exporter.ArtifactError, "models.py and utils/commons.py"):
			exporter._canonical_common_head(
				missing, 7, self.pulse_config, self.encoder_checkpoint
			)

		bad = self.root / "bad_pulse"
		(bad / "utils").mkdir(parents=True)
		(bad / "models.py").write_text(BAD_MODELS_SOURCE, encoding="utf-8")
		(bad / "utils" / "commons.py").write_text(COMMONS_SOURCE, encoding="utf-8")
		bad_config = self._write_pulse_config(bad, self.encoder_checkpoint)
		with self.assertRaisesRegex(exporter.ArtifactError, "required 64 -> 64"):
			exporter._canonical_common_head(
				bad, 7, bad_config, self.encoder_checkpoint
			)

	def test_bad_config_seed_model_and_checkpoint_binding_are_rejected(self) -> None:
		valid_text = self.pulse_config.read_text(encoding="utf-8")
		bad_seed = self.pulse_root / "configs" / "bad_seed.yaml"
		bad_seed.write_text(valid_text.replace(
			"seeds: [7, 129, 56, 83]", "seeds: [129, 56, 83]"
		), encoding="utf-8")
		with self.assertRaisesRegex(exporter.ArtifactError, "must occur"):
			exporter._canonical_common_head(
				self.pulse_root, 7, bad_seed, self.encoder_checkpoint
			)

		bad_model = self.pulse_root / "configs" / "bad_model.yaml"
		bad_model.write_text(valid_text.replace(
			"embedding_dim: 64", "embedding_dim: 32"
		), encoding="utf-8")
		with self.assertRaisesRegex(exporter.ArtifactError, "model.embedding_dim"):
			exporter._canonical_common_head(
				self.pulse_root, 7, bad_model, self.encoder_checkpoint
			)

		other_checkpoint = self.root / "other_encoder.pt"
		other_state = exporter._load_torch_mapping(self.encoder_checkpoint)
		other_state = {name: value.clone() for name, value in other_state.items()}
		other_state["features.0.bias"][0] += 1.0
		torch.save(other_state, other_checkpoint)
		with self.assertRaisesRegex(exporter.ArtifactError, "does not match"):
			exporter._canonical_common_head(
				self.pulse_root, 7, self.pulse_config, other_checkpoint
			)

	def test_replay_ownership_is_required_and_diagnostic_override_is_explicit(self) -> None:
		with np.load(self.data_path, allow_pickle=False) as valid:
			x = valid["x"].copy()
			y = valid["y"].copy()
		missing_owners = self.root / "missing_owners.npz"
		np.savez_compressed(missing_owners, x=x, y=y)
		with self.assertRaisesRegex(exporter.ArtifactError, "node_ids"):
			exporter._load_replay_data(
				missing_owners, "x", "y", "node_ids", None, 0, "verify", False, None, None
			)

		mismatched = self.root / "mismatched_owners.npz"
		np.savez_compressed(
			mismatched,
			x=x,
			y=y,
			node_ids=np.asarray(["same-owner"] * 64),
		)
		with self.assertRaisesRegex(exporter.ArtifactError, "one distinct owner"):
			exporter._load_replay_data(
				mismatched, "x", "y", "node_ids", None, 0, "verify", False, None, None
			)
		arrays, metadata = exporter._load_replay_data(
			mismatched,
			"x",
			"y",
			"node_ids",
			None,
			0,
			"verify",
			True,
			None,
			None,
		)
		self.assertEqual(arrays["replay.owner_ids"].shape, (4, 16))
		self.assertFalse(metadata["ownership"]["role_binding_valid"])
		self.assertTrue(metadata["ownership"]["diagnostic_override_used"])
		self.assertEqual(exporter._build_parser().get_default("normalization"), "verify")
		diagnostic_artifact = self.root / "diagnostic_mismatch.npz"
		with contextlib.redirect_stdout(io.StringIO()):
			status = exporter.main([
				"--checkpoint", str(self.encoder_checkpoint),
				"--pulse-source-root", str(self.pulse_root),
				"--pulse-config", str(self.pulse_config),
				"--simulation-seed", "7",
				"--data", str(mismatched),
				"--allow-owner-role-mismatch",
				"--output", str(diagnostic_artifact),
			])
		self.assertEqual(status, 0)
		diagnostic_provenance = json.loads(
			diagnostic_artifact.with_suffix(".provenance.json").read_text(encoding="utf-8")
		)
		self.assertTrue(
			diagnostic_provenance["replay"]["ownership"]["diagnostic_override_used"]
		)
		with self.assertRaisesRegex(ValueError, "one distinct responder"):
			generator.load_artifact_fixture(diagnostic_artifact)

	def test_deploy_train_split_matches_pulse_rng_contract(self) -> None:
		owners = np.asarray(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
		splits = np.asarray(["simulation"] * len(owners))
		train, metadata = exporter._pulse_train_splits_from_manifest(
			owners, splits, "simulation", 7, 0.70, 0.15
		)
		self.assertEqual(train["a"][:8].tolist(), [88, 42, 26, 50, 54, 70, 4, 53])
		self.assertEqual(train["b"][:8].tolist(), [100, 143, 107, 138, 179, 121, 105, 102])
		self.assertEqual(train["c"][:8].tolist(), [213, 275, 230, 240, 220, 236, 280, 266])
		self.assertEqual(metadata["owner_order"], ["a", "b", "c"])
		self.assertEqual(metadata["counts"]["a"], {
			"total": 100,
			"train": 70,
			"validation": 15,
			"test": 15,
		})

	def test_deploy_encoder_is_folded_and_node_head_is_exact(self) -> None:
		_, paths = self._write_deploy_bundle()
		arrays, resolution, head, source_encoder, _, head_state = exporter._load_deploy_models(
			paths,
			"a",
			"node-seeded",
			42,
			None,
			None,
			None,
			None,
			7,
			0.70,
			0.15,
		)
		self.assertEqual(head["kind"], "deploy_setup_node_initial_head")
		self.assertEqual(head["effective_torch_seed"], 42)
		self.assertEqual(resolution["encoder"]["batch_norm_epsilons"], [1.0e-5, 1.0e-5])
		self.assertEqual(
			resolution["encoder"]["folding"]["implementation"],
			"torch.nn.utils.fusion.fuse_conv_bn_weights",
		)
		self.assertRegex(
			resolution["encoder"]["folding"]["folded_named_tensor_sha256"],
			r"^[0-9a-f]{64}$",
		)
		rng = np.random.default_rng(45)
		x = rng.normal(size=(4, 16, 3, 128)).astype(np.float32)
		x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
		equivalence = exporter._selected_window_fold_equivalence(
			source_encoder,
			head_state,
			arrays,
			x,
			np.arange(64, dtype=np.int64).reshape(4, 16),
		)
		self.assertTrue(equivalence["passed"])
		self.assertEqual(equivalence["prediction_mismatch_count"], 0)
		self.assertLessEqual(
			equivalence["embeddings"]["max_abs_error"], exporter.DEPLOY_FOLD_ATOL
		)

	def test_deploy_common_head_records_explicit_encoder_and_data_overrides(self) -> None:
		_, paths = self._write_deploy_bundle()
		config_text = self.pulse_config.read_text(encoding="utf-8")
		config_text = (
			"data:\n"
			"  path: ../data/processed/hhar_by_user.npz\n"
			"  num_nodes: 9\n"
			"  node_selection: random\n"
			"  train_fraction: 0.70\n"
			"  val_fraction: 0.15\n"
			+ config_text
		)
		self.pulse_config.write_text(config_text, encoding="utf-8")
		_, _, head, _, _, _ = exporter._load_deploy_models(
			paths,
			"a",
			"pulse-common",
			None,
			None,
			None,
			self.pulse_root,
			self.pulse_config,
			7,
			0.70,
			0.15,
		)
		self.assertEqual(head["kind"], "pulse_seeded_common_initial_head")
		config = head["source"]["config"]
		self.assertEqual(config["checkpoint_binding"], "explicit_deploy_encoder_override")
		self.assertTrue(config["deploy_encoder_override"]["used"])
		self.assertEqual(
			config["deploy_data_override"]["binding"], "explicit_deploy_bundle_override"
		)
		self.assertEqual(config["validated_fields"]["data"]["train_fraction"], 0.70)

	def test_deploy_cli_rejects_ambiguous_legacy_and_head_inputs(self) -> None:
		bundle, _ = self._write_deploy_bundle()
		parser = exporter._build_parser()
		base = [
			"--deploy-bundle", str(bundle),
			"--simulation-seed", "7",
			"--initiator-node", "a",
			"--responder-node", "b",
			"--deploy-source-split", "simulation",
			"--deploy-train-fraction", "0.70",
			"--deploy-val-fraction", "0.15",
			"--validate-only",
		]
		with self.assertRaisesRegex(exporter.ArtifactError, "requires --deploy-head-seed"):
			exporter._validate_paths(parser.parse_args(
				base + ["--deploy-head-policy", "node-seeded"]
			))
		with self.assertRaisesRegex(exporter.ArtifactError, "mutually exclusive"):
			exporter._validate_paths(parser.parse_args(base + [
				"--deploy-head-policy", "node-seeded",
				"--deploy-head-seed", "42",
				"--data", str(self.data_path),
			]))
		valid = parser.parse_args(base + [
			"--deploy-head-policy", "pulse-common",
			"--pulse-source-root", str(self.pulse_root),
			"--pulse-config", str(self.pulse_config),
		])
		exporter._validate_paths(valid)

	def test_generator_uses_distinct_exact_agreement_and_scale_reductions(self) -> None:
		reference = generator.torch_reference(generator.build_fixture(), require_pytorch=True)
		local = torch.tensor(reference["local_gradient"], dtype=torch.float32)
		remote = torch.tensor(reference["remote_gradient"], dtype=torch.float32)
		denominator = local.norm() * remote.norm()
		expected_agreement = 0.0 if float(denominator.item()) <= float(
			generator.NORM_EPSILON
		) else float(torch.clamp(torch.dot(local, remote) / denominator, -1.0, 1.0).item())
		boundaries = (0, 4096, 4160, 4544, 4550)
		local_norm_sq = sum(
			float(local[start:end].pow(2).sum().item())
			for start, end in zip(boundaries, boundaries[1:])
		)
		remote_norm_sq = sum(
			float(remote[start:end].pow(2).sum().item())
			for start, end in zip(boundaries, boundaries[1:])
		)
		expected_scale = math.sqrt(
			local_norm_sq / max(remote_norm_sq, float(generator.NORM_EPSILON))
		)
		self.assertEqual(reference["agreement"], expected_agreement)
		self.assertEqual(reference["remote_scale"], expected_scale)
		self.assertEqual(
			generator.fp32le_crc32(reference["remote_gradient"]),
			generator.fp32le_crc32(np.asarray(reference["remote_gradient"], dtype="<f4")),
		)
		golden_path = self.root / "golden.c"
		generator.write_golden_source(golden_path, reference)
		expected_crc = generator.fp32le_crc32(reference["remote_gradient"])
		self.assertIn(
			f"pulse_fixture_golden_remote_gradient_fp32le_crc32 = 0x{expected_crc:08x}U",
			golden_path.read_text(encoding="utf-8"),
		)
		manifest_path = self.root / "manifest.json"
		generator.write_manifest(
			manifest_path,
			generator.build_fixture(),
			reference,
			{
				"kind": "exported_pulse_npz",
				"source": "artifact.npz",
				"artifact_sha256": "0" * 64,
			},
		)
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		self.assertNotIn("trained common", manifest["source"].lower())
		self.assertIn("provenance-bound", manifest["source"])
		self.assertEqual(
			manifest["reference"]["remote_gradient_fp32le_crc32"], expected_crc
		)

	def test_saved_head_checkpoint_path_is_retained(self) -> None:
		head, _ = exporter._canonical_common_head(
			self.pulse_root, 129, self.pulse_config, self.encoder_checkpoint
		)
		head_path = self.root / "head.pt"
		torch.save(head, head_path)
		stdout = io.StringIO()
		with contextlib.redirect_stdout(stdout):
			status = exporter.main([
				"--checkpoint", str(self.encoder_checkpoint),
				"--head-checkpoint", str(head_path),
				"--data", str(self.data_path),
				"--validate-only",
			])
		self.assertEqual(status, 0)
		self.assertEqual(
			json.loads(stdout.getvalue())["head_initialization"]["kind"],
			"supplied_head_checkpoint",
		)
		arrays, resolution = exporter._extract_models(
			exporter._load_torch_mapping(self.encoder_checkpoint),
			head,
			None,
			None,
		)
		self.assertEqual(arrays["head.flat"].shape, (4550,))
		self.assertEqual(resolution["head"]["container"], "head_checkpoint")


if __name__ == "__main__":
	unittest.main()
