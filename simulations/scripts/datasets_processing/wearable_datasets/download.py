"""Safe, resumable dataset download and archive extraction helpers."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Mapping

from .reporting import REPORTER, human_size


def download(
    url: str,
    destination: Path,
    force: bool = False,
    dataset: str = "download",
    validator: Callable[[Path], None] | None = None,
    headers: Mapping[str, str] | None = None,
) -> Path:
    """Download a public artifact using a temporary file and atomic rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        try:
            if validator is not None:
                validator(destination)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            REPORTER.info(dataset, "CACHE", f"Ignoring invalid cached file {destination}: {error}")
        else:
            REPORTER.info(dataset, "DOWNLOAD", f"Using cached {destination} ({human_size(destination.stat().st_size)})")
            REPORTER.detail(dataset, "SOURCE", url)
            return destination
    REPORTER.info(dataset, "DOWNLOAD", f"Fetching {destination.name}")
    REPORTER.info(dataset, "FROM", url)
    REPORTER.info(dataset, "TO", str(destination))
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request_headers = {
        # A number of public research repositories reject identifiable script
        # user agents even though their published files are browser-downloadable.
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    try:
        source_response = urllib.request.urlopen(request, timeout=60)
    except Exception as error:
        raise OSError(f"download failed for {url}: {error}") from error
    with source_response as source, temporary.open("wb") as target:
        expected = int(source.headers.get("Content-Length", 0))
        copied, next_report = 0, 10
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            copied += len(chunk)
            if expected:
                percent = int(copied * 100 / expected)
                if percent >= next_report:
                    REPORTER.detail(dataset, "PROGRESS", f"{min(percent, 100)}% ({human_size(copied)} / {human_size(expected)})")
                    next_report += 10
    try:
        if validator is not None:
            validator(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    REPORTER.info(dataset, "DOWNLOADED", f"{destination.name} ({human_size(destination.stat().st_size)})")
    return destination


def require_zip_members(*patterns: str, minimum_size: int = 1024) -> Callable[[Path], None]:
    """Build a validator that rejects HTML/error payloads and incomplete ZIPs."""
    def validate(path: Path) -> None:
        size = path.stat().st_size
        if size < minimum_size:
            raise ValueError(f"download is only {human_size(size)}; expected the complete dataset archive")
        if not zipfile.is_zipfile(path):
            preview = path.read_bytes()[:80].decode("utf-8", errors="replace").replace("\n", " ")
            raise ValueError(f"response is not a ZIP archive (starts with {preview!r})")
        with zipfile.ZipFile(path) as source:
            corrupt = source.testzip()
            if corrupt is not None:
                raise zipfile.BadZipFile(f"corrupt member: {corrupt}")
            names = source.namelist()
        if patterns and not any(any(Path(name).match(pattern) for pattern in patterns) for name in names):
            raise ValueError(f"archive does not contain expected files ({', '.join(patterns)})")
    return validate


def extract_nested_zips(
    archive: Path,
    destination: Path,
    dataset: str,
    ready: Callable[[], bool],
    max_archives: int = 20,
) -> None:
    """Extract an outer ZIP and any nested ZIPs until ``ready`` is true.

    Several UCI endpoints return a repository bundle containing the dataset's
    original ZIP rather than the original ZIP itself.  Discovery is
    case-insensitive and newly extracted archives are processed recursively.
    """
    extract(archive, destination, dataset)
    processed = {archive.resolve()}
    while not ready():
        candidates = sorted(
            path for path in destination.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".zip"
            and path.resolve() not in processed
        )
        if not candidates:
            return
        nested = candidates[0]
        processed.add(nested.resolve())
        if len(processed) > max_archives:
            raise RuntimeError(
                f"Stopped after extracting {max_archives} nested ZIP archives under {destination}; "
                "the downloaded package has an unexpected structure."
            )
        if not zipfile.is_zipfile(nested):
            REPORTER.info(dataset, "SKIP", f"Ignoring invalid nested ZIP candidate {nested}")
            continue
        REPORTER.info(dataset, "NESTED", f"Found embedded archive {nested.relative_to(destination)}")
        extract(nested, nested.parent, dataset)


def extract(archive: Path, destination: Path, dataset: str = "extract") -> None:
    """Extract ZIP, TAR, or multipart 7z archives without deleting sources."""
    destination.mkdir(parents=True, exist_ok=True)
    REPORTER.info(dataset, "EXTRACT", f"{archive.name} -> {destination}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
    elif archive.name.endswith(".7z.001"):
        executable = shutil.which("7z") or shutil.which("7zz")
        if not executable:
            raise RuntimeError("RingAuth extraction requires the '7z' or '7zz' executable.")
        subprocess.run([executable, "x", str(archive), f"-o{destination}", "-y"], check=True)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")
    REPORTER.info(dataset, "EXTRACTED", f"Archive contents are available in {destination}")