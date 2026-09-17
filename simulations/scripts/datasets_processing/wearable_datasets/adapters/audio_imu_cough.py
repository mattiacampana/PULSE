"""Audio-IMU cough adapter producing log-mel-like spectra and aligned IMU."""

from __future__ import annotations

import re
import wave
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from wearable_datasets.config import DatasetConfig
from wearable_datasets.core import Sample, resample, robust_window_normalize
from wearable_datasets.download import extract, require_zip_members
from wearable_datasets.reporting import REPORTER
from .base import DatasetAdapter, register


DRYAD_FILE_ID = "3424353"
DRYAD_ARCHIVE_URL = f"https://datadryad.org/downloads/file_stream/{DRYAD_FILE_ID}"
DRYAD_ARCHIVE_NAME = "Multimodal_Cough_Dataset.zip"


def _payload_ready(raw_dir: Path) -> bool:
    return any(raw_dir.rglob("DataAnnotation.json")) and any(raw_dir.rglob("Accelerometer.csv")) and any(raw_dir.rglob("*.wav"))


def _axis(frame: pd.DataFrame, axis: str) -> str:
    normalized = {re.sub(r"[^a-z0-9]", "", str(name).lower()): str(name) for name in frame.columns}
    aliases = (axis, f"acc{axis}", f"acceleration{axis}", f"{axis}axis")
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for key, original in normalized.items():
        if (key.endswith(axis) or key.startswith(f"{axis}axis")) and ("acc" in key or "axis" in key):
            return original
    raise ValueError(f"Accelerometer column {axis!r} was not found")


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read mono PCM WAV without an optional audio dependency."""
    with wave.open(str(path)) as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        raw = source.readframes(source.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width {width}: {path}")
    values = np.frombuffer(raw, dtype=dtype).reshape(-1, channels).mean(axis=1).astype(np.float32)
    if width == 1:
        values = values - 128
    return rate, values / max(float(np.max(np.abs(values))), 1.0)


@register
class AudioImuCoughAdapter(DatasetAdapter):
    name = "audio_imu_cough"
    descriptor = {"task_type": "binary", "class_names": ["non_cough", "cough"], "encoder": "audio_imu_encoder_tiny", "branches": {"audio_log_spectrum": ["frequency_bin"], "imu": ["acc_x", "acc_y", "acc_z"]}, "sampling_rates_hz": {"audio": 16000, "imu": 100}, "normalization": "per_window_per_channel_zscore", "embedding_dim": 64}

    def acquire(self, raw_dir: Path, force: bool) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        if _payload_ready(raw_dir):
            REPORTER.info(self.name, "LOCAL", f"Using extracted dataset in {raw_dir}")
            return raw_dir

        archive = raw_dir / DRYAD_ARCHIVE_NAME
        if not archive.is_file():
            raise FileNotFoundError(
                f"Audio-IMU Cough cannot be downloaded automatically. Download "
                f"{DRYAD_ARCHIVE_NAME} manually from {DRYAD_ARCHIVE_URL} and place it at "
                f"{archive}. Then rerun without --force-download."
            )

        try:
            require_zip_members("**/*.wav", "*.wav", minimum_size=100_000_000)(archive)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise RuntimeError(
                f"The local file {archive} is not the complete Audio-IMU Cough ZIP: {error}. "
                f"Download it again manually from {DRYAD_ARCHIVE_URL} and save it with the "
                f"exact name {DRYAD_ARCHIVE_NAME}."
            ) from error

        REPORTER.info(self.name, "LOCAL", f"Found manually downloaded archive at {archive}")
        extract(archive, raw_dir, self.name)
        if not _payload_ready(raw_dir):
            raise RuntimeError(
                "The Dryad Audio-IMU archive was extracted, but DataAnnotation.json, "
                "Accelerometer.csv, and WAV recordings were not all found."
            )
        return raw_dir

    def preprocess(self, source: Path, config: DatasetConfig) -> list[Sample]:
        duration = config.window_seconds or 2.0
        result = []
        for audio_path in sorted(source.rglob("*.wav")):
            # The official release stores one Accelerometer.csv per trial; use
            # the inward-facing microphone when both microphone tracks exist.
            if audio_path.stem.lower().endswith("_out") and audio_path.with_name(audio_path.name[:-8] + "_In.wav").exists():
                continue
            imu_path = next((candidate for candidate in (
                audio_path.parent / "Accelerometer.csv",
                audio_path.with_suffix(".csv"),
                audio_path.with_name(audio_path.stem + "_imu.csv"),
            ) if candidate.exists()), None)
            if imu_path is None:
                continue
            audio_rate, audio = _read_wav(audio_path)
            audio = resample(audio[None, :], audio_rate, 16000)[0]
            audio_size = round(duration * 16000)
            audio = np.pad(audio[:audio_size], (0, max(0, audio_size - len(audio))))
            _, _, spectrum = signal.stft(audio, fs=16000, nperseg=400, noverlap=240, nfft=512, boundary=None)
            log_spectrum = np.log1p(np.abs(spectrum[:64])).astype(np.float32)
            try:
                imu_frame = pd.read_csv(imu_path)
                axes = [_axis(imu_frame, axis) for axis in "xyz"]
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            imu_values = imu_frame[axes].apply(pd.to_numeric, errors="coerce").interpolate().fillna(0).to_numpy(np.float32).T
            imu_values = resample(imu_values, float(config.options.get("imu_source_rate_hz", 100)), 100)
            imu_size = round(duration * 100)
            imu_values = np.pad(imu_values[:, :imu_size], ((0, 0), (0, max(0, imu_size - imu_values.shape[1]))))
            numeric_parent = next((part for part in reversed(audio_path.parts[:-1]) if part.isdigit()), None)
            subject_match = re.search(r"^(?:subject|participant|p)[_-]?(\d+)$", audio_path.parent.parent.name, re.I)
            subject = numeric_parent or (subject_match.group(1) if subject_match else audio_path.parent.name)
            # Trial_3_Nonverbal contains the elicited non-verbal events in the
            # official release. Keep DataAnnotation.json beside the recordings
            # so event-level consumers can refine the multi-event annotation.
            trial_name = audio_path.parent.name.lower()
            label = int("nonverbal" in trial_name or ("cough" in audio_path.stem.lower() and "noncough" not in audio_path.stem.lower()))
            result.append(Sample({"audio_log_spectrum": robust_window_normalize(log_spectrum), "imu": robust_window_normalize(imu_values)}, label, subject, audio_path.stem, "audio_imu", metadata={"source_audio": audio_path.name}))
        return result