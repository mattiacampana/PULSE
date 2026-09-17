"""Best-effort Zephyr linker-map, GNU nm, and runtime RAM accounting.

Zephyr normally reserves thread stacks and the system heap in linker-accounted
RAM.  Their high-water marks describe utilization within that reservation, not
additional RAM, so this module deliberately never adds them to static RAM.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from common import (
    normalized_name,
    parse_float,
    read_csv_rows,
    sha256_file,
    write_csv_rows,
)


DEFAULT_CATEGORY_PATTERNS = {
    "encoder": [r"encoder", r"feature[_-]?extract", r"tflite", r"tensorflow", r"cmsis[_-]?nn", r"model_data"],
    "head": [r"pulse[._-]?head", r"classifier", r"\bmlp\b", r"dense[_-]?layer"],
    "training_tensors": [
        r"train",
        r"gradient",
        r"activation",
        r"tensor",
        r"optimizer",
        r"minibatch",
        r"backprop",
        r"replay",
        r"fixture",
    ],
    "protocol_buffers": [r"serializ", r"deserializ", r"protocol", r"message", r"gatt", r"bluetooth", r"\bbt[_-]"],
    "thread_stacks": [r"thread[_-]?stack", r"z_thread_stack", r"k_thread_stack"],
    "pulse_state": [r"pulse", r"peer[_-]?util", r"utility[_-]?table", r"peer[_-]?state"],
}

MEMORY_LINE_RE = re.compile(r"^\s*([^\s]+)\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)(?:\s+\S+)?\s*$")
TOP_SECTION_RE = re.compile(
    r"^(\S+)\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)"
    r"(?:\s+load address\s+(0x[0-9A-Fa-f]+))?\b"
)
PENDING_TOP_SECTION_RE = re.compile(r"^(\S+)\s*$")
PENDING_TOP_VALUE_RE = re.compile(
    r"^\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)"
    r"(?:\s+load address\s+(0x[0-9A-Fa-f]+))?\s*$"
)
INPUT_SECTION_RE = re.compile(
    r"^\s+(\.[^\s]+)\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)\s+(.+)$"
)
PENDING_SECTION_RE = re.compile(r"^\s+(\.[^\s]+)\s*$")
PENDING_VALUE_RE = re.compile(r"^\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)\s+(.+)$")
OBJECT_RE = re.compile(r"(?:^|[/\\(])[^/\\()\s]+\.(?:o|obj)(?:\)|\s|$)", re.IGNORECASE)
NM_RE = re.compile(
    r"^\s*(?:0x)?([0-9A-Fa-f]+)\s+(?:0x)?([0-9A-Fa-f]+)\s+([A-Za-z])\s+(.+?)\s*$"
)
USAGE_SYMBOL_RE = re.compile(
    r"^\s*(0x[0-9A-Fa-f]+)\s+(_flash_used|_image_ram_size)\s*="
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

NON_ALLOC_SECTION_PREFIXES = (
    ".debug",
    ".comment",
    ".stab",
    ".gnu.attributes",
    ".arm.attributes",
    ".note.gnu-stack",
    ".symtab",
    ".strtab",
    ".shstrtab",
)


def _parse_labelled_path(spec: str, default_label: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        return normalized_name(label), Path(raw_path)
    return default_label, Path(spec)


def _labelled_paths(specs: Iterable[str], default_prefix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, spec in enumerate(specs):
        default_label = default_prefix if index == 0 else f"image_{index + 1}"
        image, path = _parse_labelled_path(spec, default_label)
        if not image:
            raise ValueError(f"empty image label in {spec!r}")
        if image in result:
            raise ValueError(f"duplicate {default_prefix} input for image {image!r}")
        if not path.is_file():
            raise ValueError(f"{path}: resource provenance input is not a readable file")
        result[image] = path
    return result


def _unique_value(rows: list[dict[str, str]], field: str, path: Path) -> str:
    values = {
        str(row.get(field, "")).strip().lower()
        for row in rows
        if str(row.get(field, "")).strip()
        and str(row.get(field, "")).strip().lower()
        not in {"unknown", "unset", "operator_required", "na", "n_a", "none"}
    }
    if len(values) > 1:
        raise ValueError(f"{path}: conflicting {field} values: {sorted(values)}")
    return next(iter(values)) if values else ""


def _resource_provenance_rows(
    map_paths: Mapping[str, Path],
    nm_paths: Mapping[str, Path],
    elf_paths: Mapping[str, Path],
    metadata_paths: Mapping[str, Path],
    correctness_paths: Mapping[str, Path],
) -> list[dict[str, object]]:
    """Bind each resource label to exact hashed files and UART build identity.

    The Git revision (and, for model-bearing images, artifact digest) are
    printable constants in the ELF.  Checking their byte strings in the
    supplied ELF prevents a resource row from being paired with UART metadata
    from another build.  The matched baseline emits the same PULSE_META schema
    with artifact fields set to `none`, so its revision is bound in the same way.
    """

    images = sorted(
        set(map_paths)
        | set(nm_paths)
        | set(elf_paths)
        | set(metadata_paths)
        | set(correctness_paths)
    )
    result: list[dict[str, object]] = []
    for image in images:
        map_path = map_paths.get(image)
        nm_path = nm_paths.get(image)
        elf_path = elf_paths.get(image)
        metadata_path = metadata_paths.get(image)
        correctness_path = correctness_paths.get(image)

        metadata_rows = read_csv_rows(metadata_path) if metadata_path else []
        correctness_rows = read_csv_rows(correctness_path) if correctness_path else []
        if metadata_path and not metadata_rows:
            raise ValueError(f"{metadata_path}: image metadata contains no rows")
        if correctness_path and not correctness_rows:
            raise ValueError(f"{correctness_path}: correctness input contains no rows")

        firmware_revision = _unique_value(
            metadata_rows, "firmware_revision", metadata_path or Path("<metadata>")
        )
        correctness_revision = _unique_value(
            correctness_rows,
            "firmware_revision",
            correctness_path or Path("<correctness>"),
        )
        build_variant = _unique_value(
            metadata_rows, "build_variant", metadata_path or Path("<metadata>")
        )
        artifact_sha256 = _unique_value(
            metadata_rows, "artifact_sha256", metadata_path or Path("<metadata>")
        )
        correctness_artifact_sha256 = _unique_value(
            correctness_rows,
            "artifact_sha256",
            correctness_path or Path("<correctness>"),
        )

        for field, value, pattern in (
            ("firmware_revision", firmware_revision, GIT_REVISION_RE),
            ("correctness firmware_revision", correctness_revision, GIT_REVISION_RE),
            ("artifact_sha256", artifact_sha256, SHA256_RE),
            ("correctness artifact_sha256", correctness_artifact_sha256, SHA256_RE),
        ):
            if value and not pattern.fullmatch(value):
                raise ValueError(f"image {image}: {field} is not a concrete digest: {value!r}")
        if firmware_revision and correctness_revision and firmware_revision != correctness_revision:
            raise ValueError(
                f"image {image}: UART metadata revision {firmware_revision} does not match "
                f"correctness revision {correctness_revision}"
            )
        if (
            artifact_sha256
            and correctness_artifact_sha256
            and artifact_sha256 != correctness_artifact_sha256
        ):
            raise ValueError(
                f"image {image}: UART metadata artifact {artifact_sha256} does not match "
                f"correctness artifact {correctness_artifact_sha256}"
            )

        elf_bytes = elf_path.read_bytes() if elf_path else b""
        revision = firmware_revision or correctness_revision
        revision_embedded = bool(
            elf_bytes and revision and revision.encode("ascii") in elf_bytes
        )
        artifact = artifact_sha256 or correctness_artifact_sha256
        artifact_embedded = bool(
            elf_bytes and artifact and artifact.encode("ascii") in elf_bytes
        )
        if elf_path and revision and not revision_embedded:
            raise ValueError(
                f"image {image}: firmware revision {revision} from UART/correctness metadata "
                f"is not embedded in {elf_path}"
            )
        if elf_path and artifact and not artifact_embedded:
            raise ValueError(
                f"image {image}: artifact SHA-256 {artifact} from UART/correctness metadata "
                f"is not embedded in {elf_path}"
            )

        result.append(
            {
                "image": image,
                "firmware_revision": revision,
                "build_variant": build_variant,
                "artifact_sha256": artifact,
                "map_source_file": str(map_path or ""),
                "map_sha256": sha256_file(map_path) if map_path else "",
                "nm_source_file": str(nm_path or ""),
                "nm_sha256": sha256_file(nm_path) if nm_path else "",
                "elf_source_file": str(elf_path or ""),
                "elf_sha256": sha256_file(elf_path) if elf_path else "",
                "run_metadata_source_file": str(metadata_path or ""),
                "run_metadata_sha256": sha256_file(metadata_path) if metadata_path else "",
                "correctness_source_file": str(correctness_path or ""),
                "correctness_sha256": sha256_file(correctness_path)
                if correctness_path
                else "",
                "firmware_revision_embedded_in_elf": int(revision_embedded),
                "artifact_sha256_embedded_in_elf": int(artifact_embedded),
                "binding_method": (
                    "sha256_files_plus_uart_constants_embedded_in_elf"
                    if revision
                    else "sha256_files_no_uart_revision_schema"
                ),
            }
        )
    return result


def load_category_patterns(path: Path | None) -> dict[str, list[re.Pattern[str]]]:
    raw: Mapping[str, list[str]] = DEFAULT_CATEGORY_PATTERNS
    if path:
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict) or not all(isinstance(value, list) for value in parsed.values()):
            raise ValueError("category-rules JSON must map category names to regex lists")
        raw = {str(key): [str(item) for item in value] for key, value in parsed.items()}
    return {
        normalized_name(category): [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for category, patterns in raw.items()
    }


def classify_category(text: str, patterns: Mapping[str, list[re.Pattern[str]]]) -> str:
    for category, regexes in patterns.items():
        if any(regex.search(text) for regex in regexes):
            return category
    return "base_firmware"


def _memory_kind(name: str) -> str | None:
    normalized = normalized_name(name)
    if "flash" in normalized or normalized in {"rom", "code"}:
        return "flash"
    if "ram" in normalized or "sram" in normalized:
        return "static_ram"
    return None


def _region_for_address(regions: list[dict[str, int | str]], address: int) -> dict[str, int | str] | None:
    matches = [
        region
        for region in regions
        if int(region["origin"]) <= address < int(region["origin"]) + int(region["length"])
    ]
    return min(matches, key=lambda region: int(region["length"])) if matches else None


def _merged_interval_bytes(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _is_non_alloc_section(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(NON_ALLOC_SECTION_PREFIXES)


def _within_interval(address: int, size: int, intervals: list[tuple[int, int]]) -> bool:
    end = address + size
    return any(start <= address and end <= interval_end for start, interval_end in intervals)


def parse_map(
    path: Path, image: str, patterns: Mapping[str, list[re.Pattern[str]]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source_sha256 = sha256_file(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    regions: list[dict[str, int | str]] = []
    in_memory_configuration = False
    for line in lines:
        if line.strip() == "Memory Configuration":
            in_memory_configuration = True
            continue
        if in_memory_configuration and line.strip().startswith("Linker script and memory map"):
            break
        if not in_memory_configuration or not line.strip() or line.lstrip().startswith("Name"):
            continue
        match = MEMORY_LINE_RE.match(line)
        if not match:
            continue
        name, raw_origin, raw_length = match.groups()
        kind = _memory_kind(name)
        if kind:
            regions.append(
                {
                    "name": name,
                    "kind": kind,
                    "origin": int(raw_origin, 16),
                    "length": int(raw_length, 16),
                }
            )

    top_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    loadable_ram_intervals: list[tuple[int, int]] = []
    usage_symbols: dict[str, int] = {}
    contributions: list[tuple[str, int, int, str, bool]] = []
    in_linker_map = False
    pending_section: str | None = None
    pending_top_section: str | None = None

    def record_top_section(
        name: str, raw_address: str, raw_size: str, raw_load_address: str | None
    ) -> None:
        if _is_non_alloc_section(name):
            return
        address, size = int(raw_address, 16), int(raw_size, 16)
        region = _region_for_address(regions, address)
        if region is None or size <= 0:
            return
        top_intervals[str(region["kind"])].append((address, address + size))
        if raw_load_address is None:
            return
        load_address = int(raw_load_address, 16)
        load_region = _region_for_address(regions, load_address)
        if load_region is not None and load_region["kind"] == "flash":
            top_intervals["flash"].append((load_address, load_address + size))
            if region["kind"] == "static_ram":
                loadable_ram_intervals.append((address, address + size))

    for line in lines:
        if not in_linker_map:
            if line.strip().startswith("Linker script and memory map"):
                in_linker_map = True
            continue

        usage = USAGE_SYMBOL_RE.match(line)
        if usage:
            raw_value, symbol = usage.groups()
            memory = "flash" if symbol == "_flash_used" else "static_ram"
            usage_symbols[memory] = int(raw_value, 16)

        if pending_top_section is not None:
            values = PENDING_TOP_VALUE_RE.match(line)
            if values:
                record_top_section(pending_top_section, *values.groups())
                pending_top_section = None
                pending_section = None
                continue
            pending_top_section = None

        top = TOP_SECTION_RE.match(line)
        if top:
            record_top_section(*top.groups())
            pending_section = None
            continue
        input_match = INPUT_SECTION_RE.match(line)
        if input_match:
            section, raw_address, raw_size, source = input_match.groups()
            address, size = int(raw_address, 16), int(raw_size, 16)
            if not _is_non_alloc_section(section) and OBJECT_RE.search(source):
                load_in_flash = _within_interval(address, size, loadable_ram_intervals)
                contributions.append((section, address, size, source.strip(), load_in_flash))
            pending_section = None
            continue
        pending = PENDING_SECTION_RE.match(line)
        if pending:
            section = pending.group(1)
            pending_section = None if _is_non_alloc_section(section) else section
            continue
        if pending_section:
            values = PENDING_VALUE_RE.match(line)
            if values and OBJECT_RE.search(values.group(3)):
                raw_address, raw_size, source = values.groups()
                address, size = int(raw_address, 16), int(raw_size, 16)
                load_in_flash = _within_interval(address, size, loadable_ram_intervals)
                contributions.append(
                    (pending_section, address, size, source.strip(), load_in_flash)
                )
            pending_section = None

        pending_top = PENDING_TOP_SECTION_RE.match(line)
        if pending_top:
            pending_top_section = pending_top.group(1)

    aggregated: dict[tuple[str, str], int] = defaultdict(int)
    details: list[dict[str, object]] = []
    for section, address, size, source, load_in_flash in contributions:
        if size <= 0:
            continue
        region = _region_for_address(regions, address)
        if not region:
            continue
        memory = str(region["kind"])
        category = classify_category(f"{section} {source}", patterns)
        aggregated[(memory, category)] += size
        details.append(
            {
                "image": image,
                "memory": memory,
                "category": category,
                "section": section,
                "address": f"0x{address:x}",
                "bytes": size,
                "object": source,
                "source_file": str(path),
                "source_sha256": source_sha256,
            }
        )
        # Initialized RAM occupies RAM at run time and a load image in flash.
        if memory == "static_ram" and (
            load_in_flash or section.startswith((".data", ".sdata"))
        ):
            aggregated[("flash", category)] += size

    summaries: list[dict[str, object]] = []
    for kind in ("flash", "static_ram"):
        candidate_regions = [region for region in regions if region["kind"] == kind]
        capacity = sum(int(region["length"]) for region in candidate_regions)
        used = usage_symbols.get(kind, 0)
        categorized = sum(value for (memory, _), value in aggregated.items() if memory == kind)
        if used == 0:
            used = _merged_interval_bytes(top_intervals.get(kind, []))
            used_method = "union_of_output_and_load_sections"
            if used == 0:
                used = categorized
                used_method = "sum_of_object_contributions"
            elif categorized > used:
                used = categorized
                used_method = "max_output_union_and_object_contributions"
        else:
            used_method = "zephyr_linker_usage_symbol"
            if categorized > used:
                used = categorized
                used_method = "max_zephyr_usage_symbol_and_object_contributions"
        if used > categorized:
            aggregated[(kind, "linker_padding_or_unattributed")] += used - categorized
        summaries.append(
            {
                "image": image,
                "memory": kind,
                "used_bytes": used,
                "capacity_bytes": capacity if capacity else None,
                "free_bytes": capacity - used if capacity else None,
                "utilization": used / capacity if capacity else None,
                "measurement": used_method,
                "source_file": str(path),
                "source_sha256": source_sha256,
            }
        )

    breakdown = [
        {
            "image": image,
            "memory": memory,
            "category": category,
            "bytes": size,
            "source": "linker_map",
            "source_file": str(path),
            "source_sha256": source_sha256,
        }
        for (memory, category), size in sorted(aggregated.items())
        if size
    ]
    return summaries, breakdown, details


def parse_nm(
    path: Path, image: str, patterns: Mapping[str, list[re.Pattern[str]]]
) -> list[dict[str, object]]:
    source_sha256 = sha256_file(path)
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = NM_RE.match(line)
        if not match:
            continue
        raw_address, raw_size, symbol_type, symbol = match.groups()
        size = int(raw_size, 16)
        upper_type = symbol_type.upper()
        if upper_type in {"T", "R"}:
            memory = "flash"
        elif upper_type in {"B", "D", "S", "G", "C"}:
            memory = "static_ram"
        else:
            memory = "other"
        rows.append(
            {
                "image": image,
                "memory": memory,
                "category": classify_category(symbol, patterns),
                "address": f"0x{int(raw_address, 16):x}",
                "bytes": size,
                "symbol_type": symbol_type,
                "symbol": symbol,
                "source_file": str(path),
                "source_sha256": source_sha256,
                "source_line": line_number,
            }
        )
    return rows


def _runtime_rows(paths: Iterable[Path], pulse_static_ram: float | None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in paths:
        source_sha256 = sha256_file(path)
        for raw in read_csv_rows(path):
            row = {normalized_name(key): value for key, value in raw.items()}
            direct = parse_float(row.get("peak_ram_bytes"))
            capacity = parse_float(row.get("ram_capacity_bytes"))
            minimum_free = parse_float(row.get("minimum_free_ram_bytes"))
            stack_value = parse_float(row.get("peak_stack_bytes"))
            if stack_value is None:
                stack_value = parse_float(row.get("thread_stack_peak_bytes"))
            heap_value = parse_float(row.get("peak_heap_bytes"))
            if heap_value is None:
                heap_value = parse_float(row.get("system_heap_peak_bytes"))
            stack = stack_value or 0.0
            heap = heap_value or 0.0
            stack_heap_upper_bound = (
                stack + heap if stack_value is not None or heap_value is not None else None
            )
            other = parse_float(row.get("other_dynamic_bytes")) or 0.0
            additional = parse_float(row.get("additional_dynamic_bytes"))
            static = parse_float(row.get("static_ram_bytes"))
            if static is None:
                static = pulse_static_ram
            if direct is not None:
                peak = direct
                method = "reported_peak_ram_bytes"
            elif capacity is not None and minimum_free is not None:
                peak = capacity - minimum_free
                method = "capacity_minus_minimum_free_ram"
            elif static is not None and additional is not None:
                # This path is intentionally opt-in. Statically allocated Zephyr
                # stacks/heaps already appear in .bss/.noinit and must not be
                # blindly added to linker RAM a second time.
                peak = static + additional
                method = "static_plus_explicit_nonoverlapping_dynamic"
            elif static is not None:
                # This is the sound total footprint available from the emitted
                # event fields.  The linker total is reserved for the image at
                # run time and already includes the stack/heap pools.  Exact
                # live-byte usage cannot be reconstructed from independent
                # stack and heap maxima: their peaks need not be simultaneous,
                # and other Zephyr thread stacks are not measured by the event.
                peak = static
                method = "linker_reserved_ram_includes_stack_heap"
            else:
                peak = None
                method = "insufficient_nonoverlapping_data"
            output: dict[str, object] = dict(row)
            output.update(
                {
                    "peak_ram_bytes": peak,
                    "static_ram_bytes": static,
                    "peak_stack_bytes": stack,
                    "peak_heap_bytes": heap,
                    "stack_heap_watermark_upper_bound_bytes": stack_heap_upper_bound,
                    "other_dynamic_bytes": other,
                    "additional_dynamic_bytes": additional,
                    "ram_capacity_bytes": capacity,
                    "minimum_free_ram_bytes": minimum_free,
                    "calculation_method": method,
                    "source_file": str(path),
                    "source_sha256": source_sha256,
                }
            )
            result.append(output)
    return result


def analyze_resources(
    map_specs: Iterable[str],
    nm_specs: Iterable[str],
    runtime_paths: Iterable[Path],
    output_dir: Path,
    category_rules: Path | None,
    elf_specs: Iterable[str] = (),
    metadata_specs: Iterable[str] = (),
    correctness_specs: Iterable[str] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    patterns = load_category_patterns(category_rules)
    map_paths = _labelled_paths(map_specs, "pulse")
    nm_paths = _labelled_paths(nm_specs, "pulse")
    elf_paths = _labelled_paths(elf_specs, "pulse")
    metadata_paths = _labelled_paths(metadata_specs, "pulse")
    correctness_paths = _labelled_paths(correctness_specs, "correctness")
    runtime_paths = list(runtime_paths)
    summaries: list[dict[str, object]] = []
    breakdown: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for image, path in map_paths.items():
        image_summary, image_breakdown, image_details = parse_map(path, image, patterns)
        summaries.extend(image_summary)
        breakdown.extend(image_breakdown)
        details.extend(image_details)

    symbols: list[dict[str, object]] = []
    for image, path in nm_paths.items():
        symbols.extend(parse_nm(path, image, patterns))

    mapped_images = {str(row["image"]) for row in summaries}
    nm_only: dict[tuple[str, str, str], int] = defaultdict(int)
    for symbol in symbols:
        image = str(symbol["image"])
        if image in mapped_images or symbol["memory"] == "other":
            continue
        memory = str(symbol["memory"])
        category = str(symbol["category"])
        size = int(symbol["bytes"])
        nm_only[(image, memory, category)] += size
        if memory == "static_ram" and str(symbol["symbol_type"]).upper() == "D":
            nm_only[(image, "flash", category)] += size
    for (image, memory, category), size in sorted(nm_only.items()):
        breakdown.append(
            {
                "image": image,
                "memory": memory,
                "category": category,
                "bytes": size,
                "source": "nm_lower_bound",
                "source_file": str(nm_paths[image]),
                "source_sha256": sha256_file(nm_paths[image]),
            }
        )
    for image, memory in sorted({(key[0], key[1]) for key in nm_only}):
        used = sum(
            size
            for (row_image, row_memory, _), size in nm_only.items()
            if row_image == image and row_memory == memory
        )
        summaries.append(
            {
                "image": image,
                "memory": memory,
                "used_bytes": used,
                "capacity_bytes": None,
                "free_bytes": None,
                "utilization": None,
                "measurement": "sum_nm_symbols_lower_bound",
                "source_file": str(nm_paths[image]),
                "source_sha256": sha256_file(nm_paths[image]),
            }
        )

    summary_by_key = {(row["image"], row["memory"]): row for row in summaries}
    if ("pulse", "flash") in summary_by_key and ("baseline", "flash") in summary_by_key:
        for memory in ("flash", "static_ram"):
            pulse_row = summary_by_key.get(("pulse", memory))
            baseline_row = summary_by_key.get(("baseline", memory))
            if pulse_row and baseline_row:
                pulse_row["incremental_bytes_vs_baseline"] = float(pulse_row["used_bytes"]) - float(
                    baseline_row["used_bytes"]
                )

    pulse_static = None
    pulse_static_row = summary_by_key.get(("pulse", "static_ram"))
    if pulse_static_row:
        pulse_static = float(pulse_static_row["used_bytes"])
    runtime = _runtime_rows(runtime_paths, pulse_static)
    provenance = _resource_provenance_rows(
        map_paths,
        nm_paths,
        elf_paths,
        metadata_paths,
        correctness_paths,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "resource_summary.csv",
        summaries,
        preferred=[
            "image",
            "memory",
            "used_bytes",
            "capacity_bytes",
            "free_bytes",
            "utilization",
            "incremental_bytes_vs_baseline",
            "measurement",
            "source_file",
            "source_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "resource_breakdown.csv",
        breakdown,
        preferred=[
            "image",
            "memory",
            "category",
            "bytes",
            "source",
            "source_file",
            "source_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "map_contributions.csv",
        details,
        preferred=[
            "image",
            "memory",
            "category",
            "section",
            "address",
            "bytes",
            "object",
            "source_file",
            "source_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "symbol_sizes.csv",
        symbols,
        preferred=[
            "image",
            "memory",
            "category",
            "address",
            "bytes",
            "symbol_type",
            "symbol",
            "source_file",
            "source_sha256",
            "source_line",
        ],
    )
    write_csv_rows(
        output_dir / "runtime_ram.csv",
        runtime,
        preferred=[
            "event_path",
            "role",
            "peak_ram_bytes",
            "static_ram_bytes",
            "peak_stack_bytes",
            "peak_heap_bytes",
            "stack_heap_watermark_upper_bound_bytes",
            "other_dynamic_bytes",
            "additional_dynamic_bytes",
            "ram_capacity_bytes",
            "minimum_free_ram_bytes",
            "calculation_method",
            "source_file",
            "source_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "resource_provenance.csv",
        provenance,
        preferred=[
            "image",
            "firmware_revision",
            "build_variant",
            "artifact_sha256",
            "elf_source_file",
            "elf_sha256",
            "map_source_file",
            "map_sha256",
            "nm_source_file",
            "nm_sha256",
            "run_metadata_source_file",
            "run_metadata_sha256",
            "correctness_source_file",
            "correctness_sha256",
            "firmware_revision_embedded_in_elf",
            "artifact_sha256_embedded_in_elf",
            "binding_method",
        ],
    )
    return summaries, breakdown, runtime
