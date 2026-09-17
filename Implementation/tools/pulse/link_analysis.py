"""Validate canonical BLE sniffer exports and account LL-PDU traffic per PULSE event.

This module intentionally does not parse packet-capture formats.  A capture must first be
exported to the canonical CSV schema documented in ``tools/pulse/README.md``.  The original
capture is retained as provenance and hashed here, but is never interpreted by this code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from common import (
    PULSE_BUILD_CONTEXT_FIELDS,
    PULSE_EVENT_CONFIGURATION_FIELDS,
    normalized_name,
    parse_bool,
    parse_float,
    read_csv_rows,
    write_csv_rows,
)
from summarize import ACCEPTED_PATHS, accepted_session_integrity


DIRECTIONS = {"initiator_to_responder", "responder_to_initiator"}
PDU_KINDS = {"data", "advertising"}
PAIR_JOIN_STATUSES = {
    "paired_remote_session_exact_exchange_id",
    "initiator_only_pre_session_failure_exact_exchange_id",
}
IDENTITY_FIELDS = ("run_id", "pair_id", "exchange_id")
RESULT_CONTEXT_FIELDS = (
    *PULSE_BUILD_CONTEXT_FIELDS,
    *PULSE_EVENT_CONFIGURATION_FIELDS,
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ADDRESS_RE = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")
ADDRESS_TYPES = {
    "public",
    "random",
    "random_static",
    "random_resolvable",
    "random_non_resolvable",
}


@dataclass(frozen=True)
class SyncModel:
    method: str
    slope: float
    intercept_s: float
    uncertainty_s: float
    anchor_count: int
    anchor_start_s: float | None = None
    anchor_end_s: float | None = None
    maximum_anchor_residual_s: float = 0.0
    valid: bool = True
    issues: tuple[str, ...] = ()

    def map_time(self, timestamp_s: float) -> float:
        return self.slope * timestamp_s + self.intercept_s


@dataclass
class Capture:
    capture_id: str
    run_id: str
    pair_id: str
    source_file: Path
    canonical_sha256: str
    raw_pcap_path: Path
    raw_pcap_sha256: str
    capture_start_s: float
    capture_end_s: float
    rows: list[dict[str, str]]
    sync: SyncModel | None = None
    issues: list[str] | None = None

    def __post_init__(self) -> None:
        if self.issues is None:
            self.issues = []


@dataclass(frozen=True)
class EventWindow:
    run_id: str
    pair_id: str
    exchange_id: str
    event_path: str
    role: str
    connection_state: str
    trial_id: str
    event_index: str
    marker_event_index: str
    warmup: str
    post_warmup: str
    success: str
    remote_session_started: str
    join_status: str
    start_s: float
    end_s: float
    source_file: str
    result_context: tuple[tuple[str, str], ...]

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.run_id, self.pair_id, self.exchange_id)

    @property
    def event_uid(self) -> str:
        return "/".join(self.identity)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(row: Mapping[str, object], field: str, context: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"{context}: required {field} is empty")
    return value


def _required_float(row: Mapping[str, object], field: str, context: str) -> float:
    value = parse_float(row.get(field))
    if value is None:
        raise ValueError(f"{context}: {field} must be a finite number")
    return value


def _required_int(
    row: Mapping[str, object], field: str, context: str, minimum: int, maximum: int
) -> int:
    value = _required_float(row, field, context)
    if not value.is_integer() or not minimum <= value <= maximum:
        raise ValueError(
            f"{context}: {field} must be an integer in [{minimum}, {maximum}]"
        )
    return int(value)


def _required_bool(row: Mapping[str, object], field: str, context: str) -> bool:
    value = parse_bool(row.get(field))
    if value is None:
        raise ValueError(f"{context}: {field} must be an explicit boolean")
    return value


def _canonical_rows(path: Path) -> list[dict[str, str]]:
    rows = [
        {normalized_name(key): value for key, value in row.items()}
        for row in read_csv_rows(path)
    ]
    if not rows:
        raise ValueError(f"{path}: canonical packet CSV contains no packet rows")
    required = {
        "capture_id",
        "run_id",
        "pair_id",
        "packet_index",
        "timestamp_s",
        "capture_start_s",
        "capture_end_s",
        "direction",
        "transmitter_address",
        "transmitter_address_type",
        "receiver_address",
        "receiver_address_type",
        "pdu_kind",
        "llid",
        "length_bytes",
        "crc_ok",
        "gap_before",
        "is_retransmission",
        "connection_epoch",
        "sn",
        "nesn",
        "payload_sha256",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{path}: canonical packet CSV missing column(s): {','.join(missing)}")
    return rows


def _parse_raw_specs(specs: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"invalid raw capture provenance {spec!r}; expected CAPTURE_ID=PCAP_PATH"
            )
        capture_id, raw_path = spec.split("=", 1)
        capture_id = capture_id.strip()
        path = Path(raw_path.strip()).resolve()
        if not capture_id or not raw_path.strip():
            raise ValueError(
                f"invalid raw capture provenance {spec!r}; expected CAPTURE_ID=PCAP_PATH"
            )
        if capture_id in result:
            raise ValueError(f"duplicate raw PCAP mapping for capture_id {capture_id!r}")
        if not path.is_file():
            raise ValueError(f"raw PCAP for capture_id {capture_id!r} does not exist: {path}")
        result[capture_id] = path
    return result


def _load_captures(packet_csvs: Iterable[Path], raw_specs: Iterable[str]) -> list[Capture]:
    raw_paths = _parse_raw_specs(raw_specs)
    captures: list[Capture] = []
    seen_ids: set[str] = set()
    for path in packet_csvs:
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"canonical packet CSV does not exist: {path}")
        rows = _canonical_rows(path)
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[_required_text(row, "capture_id", str(path))].append(row)
        for capture_id, group in sorted(groups.items()):
            if capture_id in seen_ids:
                raise ValueError(
                    f"capture_id {capture_id!r} occurs in more than one canonical CSV"
                )
            seen_ids.add(capture_id)
            if capture_id not in raw_paths:
                raise ValueError(
                    f"capture_id {capture_id!r} has no raw PCAP provenance; pass "
                    f"--raw-pcap {capture_id}=PATH"
                )
            first = group[0]
            run_id = _required_text(first, "run_id", f"{path}:{capture_id}")
            pair_id = _required_text(first, "pair_id", f"{path}:{capture_id}")
            start_s = _required_float(first, "capture_start_s", f"{path}:{capture_id}")
            end_s = _required_float(first, "capture_end_s", f"{path}:{capture_id}")
            if end_s <= start_s:
                raise ValueError(f"{path}:{capture_id}: capture_end_s must exceed capture_start_s")
            invariant_fields = {
                "run_id": run_id,
                "pair_id": pair_id,
                "capture_start_s": str(first.get("capture_start_s", "")).strip(),
                "capture_end_s": str(first.get("capture_end_s", "")).strip(),
            }
            for position, row in enumerate(group, 1):
                for field, expected in invariant_fields.items():
                    if str(row.get(field, "")).strip() != expected:
                        raise ValueError(
                            f"{path}:{capture_id}: row {position} changes invariant {field}"
                        )
            captures.append(
                Capture(
                    capture_id=capture_id,
                    run_id=run_id,
                    pair_id=pair_id,
                    source_file=path,
                    canonical_sha256=_sha256(path),
                    raw_pcap_path=raw_paths[capture_id],
                    raw_pcap_sha256=_sha256(raw_paths[capture_id]),
                    capture_start_s=start_s,
                    capture_end_s=end_s,
                    rows=group,
                )
            )
    unknown_raw = sorted(set(raw_paths).difference(seen_ids))
    if unknown_raw:
        raise ValueError(
            "raw PCAP mapping(s) have no canonical capture_id: " + ", ".join(unknown_raw)
        )
    if not captures:
        raise ValueError("link analysis requires at least one canonical packet CSV")
    return captures


def _fit_sync_models(
    captures: Sequence[Capture],
    shared_timebase: bool,
    shared_uncertainty_s: float | None,
    anchors_csv: Path | None,
) -> None:
    if shared_timebase == (anchors_csv is not None):
        raise ValueError("select exactly one synchronization mode: shared timebase or sync anchors")
    if shared_timebase:
        if shared_uncertainty_s is None or not math.isfinite(shared_uncertainty_s):
            raise ValueError("shared timebase requires explicit --time-uncertainty-s")
        if shared_uncertainty_s < 0.0:
            raise ValueError("shared timebase uncertainty must be nonnegative")
        for capture in captures:
            capture.sync = SyncModel(
                method="shared_timebase",
                slope=1.0,
                intercept_s=0.0,
                uncertainty_s=shared_uncertainty_s,
                anchor_count=0,
            )
        return

    if shared_uncertainty_s is not None:
        raise ValueError(
            "--time-uncertainty-s applies only to --shared-timebase; affine anchors carry "
            "their own explicit uncertainty"
        )

    assert anchors_csv is not None
    anchor_rows = [
        {normalized_name(key): value for key, value in row.items()}
        for row in read_csv_rows(anchors_csv)
    ]
    required = {"capture_id", "sniffer_time_s", "event_time_s", "uncertainty_s"}
    if not anchor_rows:
        raise ValueError(f"{anchors_csv}: synchronization-anchor CSV contains no rows")
    missing = sorted(required.difference(anchor_rows[0]))
    if missing:
        raise ValueError(f"{anchors_csv}: missing anchor column(s): {','.join(missing)}")
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for position, row in enumerate(anchor_rows, 2):
        context = f"{anchors_csv}:row {position}"
        capture_id = _required_text(row, "capture_id", context)
        sniffer = _required_float(row, "sniffer_time_s", context)
        event = _required_float(row, "event_time_s", context)
        uncertainty = _required_float(row, "uncertainty_s", context)
        if uncertainty < 0.0:
            raise ValueError(f"{context}: uncertainty_s must be nonnegative")
        grouped[capture_id].append((sniffer, event, uncertainty))

    known = {capture.capture_id for capture in captures}
    unknown = sorted(set(grouped).difference(known))
    if unknown:
        raise ValueError("sync anchors reference unknown capture_id(s): " + ", ".join(unknown))
    for capture in captures:
        anchors = sorted(grouped.get(capture.capture_id, []))
        issues: list[str] = []
        if len(anchors) < 2:
            capture.sync = SyncModel(
                method="affine_anchors",
                slope=math.nan,
                intercept_s=math.nan,
                uncertainty_s=math.nan,
                anchor_count=len(anchors),
                valid=False,
                issues=("fewer than two synchronization anchors",),
            )
            continue
        xs = [item[0] for item in anchors]
        ys = [item[1] for item in anchors]
        if len(set(xs)) < 2:
            capture.sync = SyncModel(
                method="affine_anchors",
                slope=math.nan,
                intercept_s=math.nan,
                uncertainty_s=math.nan,
                anchor_count=len(anchors),
                valid=False,
                issues=("synchronization anchors do not span two distinct sniffer times",),
            )
            continue
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        intercept = mean_y - slope * mean_x
        residuals = [abs((slope * x + intercept) - y) for x, y in zip(xs, ys)]
        if not math.isfinite(slope) or slope <= 0.0:
            issues.append("affine clock slope is not finite and positive")
        for index, (residual, anchor) in enumerate(zip(residuals, anchors), 1):
            if residual > anchor[2] + 1e-15:
                issues.append(
                    f"anchor {index} residual {residual:.12g}s exceeds stated uncertainty "
                    f"{anchor[2]:.12g}s"
                )
        anchor_start = min(xs)
        anchor_end = max(xs)
        if capture.capture_start_s < anchor_start or capture.capture_end_s > anchor_end:
            issues.append("anchors do not bracket the complete capture; extrapolation is forbidden")
        capture.sync = SyncModel(
            method="affine_anchors",
            slope=slope,
            intercept_s=intercept,
            uncertainty_s=max(
                residual + anchor[2] for residual, anchor in zip(residuals, anchors)
            ),
            anchor_count=len(anchors),
            anchor_start_s=anchor_start,
            anchor_end_s=anchor_end,
            maximum_anchor_residual_s=max(residuals),
            valid=not issues,
            issues=tuple(issues),
        )


def _event_text(row: Mapping[str, object], field: str, context: str) -> str:
    return _required_text(row, field, context)


def _event_window(row: Mapping[str, object], context: str) -> EventWindow:
    start_s = _required_float(row, "start_time_s", context)
    end_s = _required_float(row, "end_time_s", context)
    if end_s <= start_s:
        raise ValueError(f"{context}: end_time_s must exceed start_time_s")
    if parse_bool(row.get("truncated_start")) is True or parse_bool(row.get("truncated_end")) is True:
        raise ValueError(f"{context}: truncated event windows cannot support link accounting")
    return EventWindow(
        run_id=_event_text(row, "run_id", context),
        pair_id=_event_text(row, "pair_id", context),
        exchange_id=_event_text(row, "exchange_id", context),
        event_path=normalized_name(_event_text(row, "event_path", context)),
        role=normalized_name(_event_text(row, "role", context)),
        connection_state=normalized_name(str(row.get("connection_state", ""))),
        trial_id=str(row.get("trial_id", "")).strip(),
        event_index=str(row.get("event_index", "")).strip(),
        marker_event_index=str(row.get("marker_event_index", "")).strip(),
        warmup=str(row.get("warmup", "")).strip(),
        post_warmup=str(row.get("post_warmup", "")).strip(),
        success=str(row.get("success", "")).strip(),
        remote_session_started=str(row.get("remote_session_started", "")).strip(),
        join_status=normalized_name(str(row.get("join_status", ""))),
        start_s=start_s,
        end_s=end_s,
        source_file=str(row.get("source_file", "")).strip(),
        result_context=tuple(
            (field, str(row.get(field, "")).strip())
            for field in RESULT_CONTEXT_FIELDS
        ),
    )


def _load_event_windows(path: Path) -> tuple[list[EventWindow], list[str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"{path}: event_power_metrics.csv contains no rows")
    component_rows = [
        row
        for row in rows
        if normalized_name(str(row.get("event_path", ""))) in ACCEPTED_PATHS
        and normalized_name(str(row.get("role", ""))) in {"initiator", "responder"}
    ]
    initiators, _, pairing_issues = accepted_session_integrity(component_rows)
    pair_rows: dict[tuple[str, str, str], Mapping[str, str]] = {}
    issues = list(pairing_issues)
    for position, row in enumerate(rows, 2):
        event_path = normalized_name(str(row.get("event_path", "")))
        role = normalized_name(str(row.get("role", "")))
        if event_path in ACCEPTED_PATHS and role == "pair":
            key = tuple(str(row.get(field, "")).strip() for field in IDENTITY_FIELDS)
            if not all(key):
                issues.append(f"event-power row {position} accepted pair missing exact identity")
            elif key in pair_rows:
                issues.append(f"duplicate accepted pair power windows for {key}")
            else:
                pair_rows[key] = row

    windows: list[EventWindow] = []
    for key, initiator in sorted(initiators.items()):
        if key not in pair_rows:
            issues.append(f"accepted session {key} has no canonical role=pair power window")
            continue
        pair_row = pair_rows[key]
        remote_started = parse_bool(initiator.get("remote_session_started"))
        expected_join = (
            "paired_remote_session_exact_exchange_id"
            if remote_started is True
            else "initiator_only_pre_session_failure_exact_exchange_id"
        )
        actual_join = normalized_name(str(pair_row.get("join_status", "")))
        if actual_join != expected_join:
            issues.append(
                f"accepted pair window {key} has join_status {actual_join or '<missing>'}; "
                f"expected {expected_join}"
            )
        if parse_bool(pair_row.get("remote_session_started")) is not remote_started:
            issues.append(f"accepted pair window {key} remote_session_started mismatch")
        if normalized_name(str(pair_row.get("event_path", ""))) != normalized_name(
            str(initiator.get("event_path", ""))
        ):
            issues.append(f"accepted pair window {key} event_path mismatch")
        if str(pair_row.get("trial_id", "")).strip() != str(
            initiator.get("trial_id", "")
        ).strip():
            issues.append(f"accepted pair window {key} trial_id mismatch")
        if normalized_name(str(pair_row.get("connection_state", ""))) != normalized_name(
            str(initiator.get("connection_state", ""))
        ):
            issues.append(f"accepted pair window {key} connection_state mismatch")
        for field in RESULT_CONTEXT_FIELDS:
            if str(pair_row.get(field, "")).strip() != str(
                initiator.get(field, "")
            ).strip():
                issues.append(
                    f"accepted pair window {key} {field} build/config context mismatch"
                )
        pair_warmup = parse_bool(pair_row.get("warmup"))
        initiator_warmup = parse_bool(initiator.get("warmup"))
        pair_post_warmup = parse_bool(pair_row.get("post_warmup"))
        initiator_post_warmup = parse_bool(initiator.get("post_warmup"))
        if pair_warmup != initiator_warmup or pair_post_warmup != initiator_post_warmup:
            issues.append(f"accepted pair window {key} warmup label mismatch")
        if parse_bool(pair_row.get("success")) != parse_bool(initiator.get("success")):
            issues.append(f"accepted pair window {key} success mismatch")
        try:
            windows.append(_event_window(pair_row, f"{path}:accepted pair {key}"))
        except ValueError as error:
            issues.append(str(error))

    unexpected_pairs = sorted(set(pair_rows).difference(initiators))
    for key in unexpected_pairs:
        issues.append(f"orphan accepted role=pair power window for {key}")

    for position, row in enumerate(rows, 2):
        event_path = normalized_name(str(row.get("event_path", "")))
        role = normalized_name(str(row.get("role", "")))
        if event_path in ACCEPTED_PATHS or role == "pair":
            continue
        if event_path not in {"local", "no_contact", "scan_only", "failed"}:
            continue
        join_status = normalized_name(str(row.get("join_status", "")))
        if join_status != "matched":
            issues.append(
                f"{path}:row {position} non-accepted window has join_status "
                f"{join_status or '<missing>'}; expected matched"
            )
        if event_path in {"local", "no_contact"} and role != "local":
            issues.append(
                f"{path}:row {position} {event_path} window has role {role or '<missing>'}; "
                "expected local"
            )
        try:
            windows.append(_event_window(row, f"{path}:row {position}"))
        except ValueError as error:
            issues.append(str(error))

    identity_groups: dict[tuple[str, str, str], list[EventWindow]] = defaultdict(list)
    for window in windows:
        identity_groups[window.identity].append(window)
    for key, members in sorted(identity_groups.items()):
        if len(members) != 1:
            issues.append(f"duplicate canonical event identity {key}: {len(members)} windows")

    chronological: dict[tuple[str, str], list[EventWindow]] = defaultdict(list)
    for window in windows:
        chronological[(window.run_id, window.pair_id)].append(window)
    for key, members in chronological.items():
        ordered = sorted(members, key=lambda item: (item.start_s, item.end_s))
        for left, right in zip(ordered, ordered[1:]):
            if right.start_s < left.end_s:
                issues.append(
                    f"overlapping canonical event windows for run/pair {key}: "
                    f"{left.exchange_id} and {right.exchange_id}"
                )
    if not windows:
        issues.append("no canonical PULSE event windows were found")
    return windows, issues


def _address(value: object) -> str:
    return str(value if value is not None else "").strip().lower().replace("-", ":")


def _load_role_identities(
    metadata_csv: Path,
) -> tuple[dict[tuple[str, str], dict[str, tuple[str, str]]], list[str]]:
    """Load complementary logical-role BLE identities from both boards' PULSE_META rows."""

    rows = read_csv_rows(metadata_csv)
    identities: dict[tuple[str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    peers: dict[tuple[str, str, str], tuple[str, str]] = {}
    issues: list[str] = []
    for position, raw_row in enumerate(rows, 2):
        row = {normalized_name(key): value for key, value in raw_row.items()}
        run_id = str(row.get("run_id", "")).strip()
        pair_id = str(row.get("pair_id", "")).strip()
        role = normalized_name(str(row.get("role", "")))
        if role not in {"initiator", "responder"}:
            continue
        context = f"{metadata_csv}:row {position}"
        if not run_id or not pair_id:
            issues.append(f"{context}: role metadata lacks run_id or pair_id")
            continue
        local_address = _address(row.get("local_identity_address"))
        local_type = normalized_name(str(row.get("local_identity_address_type", "")))
        peer_address = _address(row.get("peer_identity_address"))
        peer_type = normalized_name(str(row.get("peer_identity_address_type", "")))
        if not ADDRESS_RE.fullmatch(local_address):
            issues.append(f"{context}: invalid/missing local_identity_address")
        if local_type not in ADDRESS_TYPES:
            issues.append(f"{context}: invalid/missing local_identity_address_type")
        if not ADDRESS_RE.fullmatch(peer_address):
            issues.append(f"{context}: invalid/missing peer_identity_address")
        if peer_type not in ADDRESS_TYPES:
            issues.append(f"{context}: invalid/missing peer_identity_address_type")
        if any(
            issue.startswith(f"{context}:") for issue in issues
        ):
            continue
        block = (run_id, pair_id)
        if role in identities[block]:
            issues.append(f"{context}: duplicate {role} BLE identity metadata for {block}")
            continue
        identities[block][role] = (local_address, local_type)
        peers[(run_id, pair_id, role)] = (peer_address, peer_type)

    for block, roles in sorted(identities.items()):
        if set(roles) != {"initiator", "responder"}:
            issues.append(f"run/pair {block} lacks complementary initiator/responder identities")
            continue
        if roles["initiator"] == roles["responder"]:
            issues.append(f"run/pair {block} maps both roles to the same BLE identity")
        if peers.get((*block, "initiator")) != roles["responder"]:
            issues.append(f"run/pair {block} initiator peer identity does not match responder")
        if peers.get((*block, "responder")) != roles["initiator"]:
            issues.append(f"run/pair {block} responder peer identity does not match initiator")
    if not identities:
        issues.append("run metadata contains no usable initiator/responder BLE identities")
    return dict(identities), issues


def _audit_row(
    audit_type: str,
    status: str,
    issues: Sequence[str],
    **values: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "audit_type": audit_type,
        "status": status,
        "issue_count": len(issues),
        "issues": " | ".join(issues),
    }
    row.update(values)
    return row


def _validate_packets(
    capture: Capture,
    role_identities: Mapping[tuple[str, str], Mapping[str, tuple[str, str]]],
) -> tuple[list[dict[str, object]], dict[int, list[str]]]:
    assert capture.sync is not None
    packets: list[dict[str, object]] = []
    row_issues: dict[int, list[str]] = defaultdict(list)
    indices: dict[int, int] = defaultdict(int)
    previous_index: int | None = None
    previous_time: float | None = None
    last_unique: dict[tuple[str, str], dict[str, object]] = {}
    for source_position, row in enumerate(capture.rows, 2):
        context = f"{capture.source_file}:row {source_position}"
        index = _required_int(row, "packet_index", context, 0, 2**63 - 1)
        raw_time = _required_float(row, "timestamp_s", context)
        if not capture.capture_start_s <= raw_time <= capture.capture_end_s:
            row_issues[index].append("packet timestamp lies outside declared capture bounds")
        direction = normalized_name(_required_text(row, "direction", context))
        if direction not in DIRECTIONS:
            row_issues[index].append(
                "direction must be initiator_to_responder or responder_to_initiator"
            )
        pdu_kind = normalized_name(_required_text(row, "pdu_kind", context))
        if pdu_kind not in PDU_KINDS:
            row_issues[index].append("pdu_kind must be data or advertising")
        length_bytes = _required_int(row, "length_bytes", context, 0, 255)
        crc_ok = _required_bool(row, "crc_ok", context)
        gap_before = _required_bool(row, "gap_before", context)
        retransmission = _required_bool(row, "is_retransmission", context)
        fingerprint = _required_text(row, "payload_sha256", context).lower()
        if not SHA256_RE.fullmatch(fingerprint):
            row_issues[index].append("payload_sha256 must be a 64-digit hexadecimal SHA-256")

        transmitter_address = _address(row.get("transmitter_address"))
        transmitter_type = normalized_name(str(row.get("transmitter_address_type", "")))
        receiver_address = _address(row.get("receiver_address"))
        receiver_type = normalized_name(str(row.get("receiver_address_type", "")))
        if not ADDRESS_RE.fullmatch(transmitter_address) or transmitter_type not in ADDRESS_TYPES:
            row_issues[index].append("invalid transmitter BLE identity address/type")
        identities = role_identities.get((capture.run_id, capture.pair_id), {})
        transmitter_role = "initiator" if direction == "initiator_to_responder" else "responder"
        receiver_role = "responder" if transmitter_role == "initiator" else "initiator"
        if identities.get(transmitter_role) != (transmitter_address, transmitter_type):
            row_issues[index].append(
                f"transmitter identity does not match {transmitter_role} PULSE_META identity"
            )

        connection_epoch = str(row.get("connection_epoch", "")).strip()
        llid: int | None = None
        sn: int | None = None
        nesn: int | None = None
        if pdu_kind == "data":
            if not connection_epoch:
                row_issues[index].append("data PDU requires non-empty connection_epoch")
            llid = _required_int(row, "llid", context, 1, 3)
            sn = _required_int(row, "sn", context, 0, 1)
            nesn = _required_int(row, "nesn", context, 0, 1)
            if identities.get(receiver_role) != (receiver_address, receiver_type):
                row_issues[index].append(
                    f"receiver identity does not match {receiver_role} PULSE_META identity"
                )
        elif pdu_kind == "advertising":
            if connection_epoch or str(row.get("llid", "")).strip() or str(row.get("sn", "")).strip() or str(row.get("nesn", "")).strip():
                row_issues[index].append(
                    "advertising PDU must leave connection_epoch, llid, sn, and nesn empty"
                )
            if retransmission:
                row_issues[index].append(
                    "advertising repeats are separate transmissions, not ARQ retransmissions"
                )
            if receiver_address or receiver_type:
                if identities.get(receiver_role) != (receiver_address, receiver_type):
                    row_issues[index].append(
                        "directed advertising receiver identity does not match logical peer"
                    )

        indices[index] += 1
        if indices[index] > 1:
            row_issues[index].append("duplicate packet_index within capture")
        if previous_index is not None and index <= previous_index:
            row_issues[index].append("packet_index is not strictly increasing in CSV order")
        if previous_time is not None and raw_time < previous_time:
            row_issues[index].append("timestamp_s moves backwards in CSV order")
        previous_index = index
        previous_time = raw_time

        event_time = capture.sync.map_time(raw_time) if capture.sync.valid else math.nan
        packet: dict[str, object] = {
            "capture_id": capture.capture_id,
            "run_id": capture.run_id,
            "pair_id": capture.pair_id,
            "packet_index": index,
            "source_row": source_position,
            "sniffer_time_s": raw_time,
            "event_time_s": event_time,
            "time_uncertainty_s": capture.sync.uncertainty_s,
            "direction": direction,
            "transmitter_address": transmitter_address,
            "transmitter_address_type": transmitter_type,
            "receiver_address": receiver_address,
            "receiver_address_type": receiver_type,
            "pdu_kind": pdu_kind,
            "connection_epoch": connection_epoch,
            "llid": "" if llid is None else llid,
            "sn": "" if sn is None else sn,
            "nesn": "" if nesn is None else nesn,
            "length_bytes": length_bytes,
            "ll_pdu_bytes": 2 + length_bytes,
            "crc_ok": int(crc_ok),
            "gap_before": int(gap_before),
            "is_retransmission": int(retransmission),
            "payload_sha256": fingerprint,
            "canonical_source_file": str(capture.source_file),
            "raw_pcap_path": str(capture.raw_pcap_path),
            "raw_pcap_sha256": capture.raw_pcap_sha256,
        }
        if pdu_kind == "data" and crc_ok and direction in DIRECTIONS and connection_epoch:
            stream = (connection_epoch, direction)
            previous = last_unique.get(stream)
            if retransmission:
                if previous is None:
                    row_issues[index].append("retransmission has no preceding unique PDU in stream")
                elif any(
                    packet[field] != previous[field]
                    for field in ("sn", "llid", "length_bytes", "payload_sha256")
                ):
                    row_issues[index].append(
                        "retransmission does not match preceding unique PDU SN/LLID/length/hash"
                    )
            else:
                if previous is not None and packet["sn"] == previous["sn"]:
                    row_issues[index].append(
                        "new data PDU did not toggle SN; label it retransmission or fix capture gap"
                    )
                last_unique[stream] = packet
        packets.append(packet)
    for packet in packets:
        issues = row_issues.get(int(packet["packet_index"]), [])
        packet["packet_integrity_valid"] = int(not issues)
        packet["packet_integrity_issues"] = " | ".join(issues)
    return packets, row_issues


def _overlaps(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return left_start < right_end and right_start < left_end


def analyze_link(
    packet_csvs: Iterable[Path],
    event_windows_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    raw_pcap_specs: Iterable[str],
    *,
    shared_timebase: bool = False,
    time_uncertainty_s: float | None = None,
    sync_anchors_csv: Path | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Create traceable link metrics without interpreting the raw PCAP.

    Returned lists are packet metrics, event metrics, capture metadata, and audit rows.  Semantic
    audit failures are written to disk and represented by ``quality_valid=0``; malformed schemas
    raise ``ValueError`` because no defensible row-level result can be produced.
    """

    captures = _load_captures(packet_csvs, raw_pcap_specs)
    event_windows_sha256 = _sha256(event_windows_csv)
    metadata_sha256 = _sha256(metadata_csv)
    sync_anchors_sha256 = _sha256(sync_anchors_csv) if sync_anchors_csv is not None else ""
    _fit_sync_models(
        captures,
        shared_timebase=shared_timebase,
        shared_uncertainty_s=time_uncertainty_s,
        anchors_csv=sync_anchors_csv,
    )
    windows, event_integrity_issues = _load_event_windows(event_windows_csv)
    role_identities, identity_issues = _load_role_identities(metadata_csv)
    global_integrity_issues = [*event_integrity_issues, *identity_issues]
    audits: list[dict[str, object]] = [
        _audit_row(
            "event_window_integrity",
            "pass" if not event_integrity_issues else "fail",
            event_integrity_issues,
            scope="all_events",
        ),
        _audit_row(
            "role_identity_integrity",
            "pass" if not identity_issues else "fail",
            identity_issues,
            scope="run_metadata",
        ),
    ]

    packets_by_capture: dict[str, list[dict[str, object]]] = {}
    packet_issues_by_capture: dict[str, dict[int, list[str]]] = {}
    for capture in captures:
        assert capture.sync is not None
        packets, row_issues = _validate_packets(capture, role_identities)
        packets_by_capture[capture.capture_id] = packets
        packet_issues_by_capture[capture.capture_id] = row_issues
        sync_issues = list(capture.sync.issues)
        audits.append(
            _audit_row(
                "capture_sync",
                "pass" if capture.sync.valid else "fail",
                sync_issues,
                scope="capture",
                capture_id=capture.capture_id,
                run_id=capture.run_id,
                pair_id=capture.pair_id,
            )
        )
        packet_issue_strings = [
            f"packet {index}: {issue}"
            for index in sorted(row_issues)
            for issue in row_issues[index]
        ]
        audits.append(
            _audit_row(
                "capture_packet_integrity",
                "pass" if not packet_issue_strings else "fail",
                packet_issue_strings,
                scope="capture",
                capture_id=capture.capture_id,
                run_id=capture.run_id,
                pair_id=capture.pair_id,
            )
        )

    windows_by_block: dict[tuple[str, str], list[EventWindow]] = defaultdict(list)
    for window in windows:
        windows_by_block[(window.run_id, window.pair_id)].append(window)
    captures_by_block: dict[tuple[str, str], list[Capture]] = defaultdict(list)
    for capture in captures:
        captures_by_block[(capture.run_id, capture.pair_id)].append(capture)

    assignments: dict[tuple[str, int], EventWindow] = {}
    possible_boundaries: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for capture in captures:
        candidates = windows_by_block.get((capture.run_id, capture.pair_id), [])
        sync = capture.sync
        assert sync is not None
        for packet in packets_by_capture[capture.capture_id]:
            key = (capture.capture_id, int(packet["packet_index"]))
            if not sync.valid:
                packet["assignment_status"] = "invalid_sync"
                continue
            time_s = float(packet["event_time_s"])
            uncertainty = sync.uncertainty_s
            lower = time_s - uncertainty
            upper = time_s + uncertainty
            definite = [
                event for event in candidates if lower >= event.start_s and upper < event.end_s
            ]
            possible = [
                event
                for event in candidates
                if upper >= event.start_s and lower < event.end_s
            ]
            if len(definite) == 1 and len(possible) == 1:
                event = definite[0]
                assignments[key] = event
                packet["assignment_status"] = "matched_exact_event_window"
                packet["exchange_id"] = event.exchange_id
                packet["event_path"] = event.event_path
                packet["event_role"] = event.role
                packet["trial_id"] = event.trial_id
                packet["event_index"] = event.event_index
                packet["warmup"] = event.warmup
                packet["post_warmup"] = event.post_warmup
            elif possible:
                packet["assignment_status"] = "boundary_ambiguous"
                packet["possible_exchange_ids"] = ";".join(
                    sorted(event.exchange_id for event in possible)
                )
                for event in possible:
                    possible_boundaries[event.identity].append(key)
            else:
                packet["assignment_status"] = "outside_event_windows"

    event_metrics: list[dict[str, object]] = []
    for event in sorted(windows, key=lambda item: (item.run_id, item.pair_id, item.start_s)):
        coverage = []
        for capture in captures_by_block.get((event.run_id, event.pair_id), []):
            sync = capture.sync
            assert sync is not None
            if not sync.valid:
                continue
            start = sync.map_time(capture.capture_start_s)
            end = sync.map_time(capture.capture_end_s)
            if start + sync.uncertainty_s <= event.start_s and end - sync.uncertainty_s >= event.end_s:
                coverage.append(capture)
        event_issues: list[str] = []
        if len(coverage) != 1:
            event_issues.append(
                f"event requires exactly one complete capture; found {len(coverage)}"
            )
        capture = coverage[0] if len(coverage) == 1 else None
        assigned: list[dict[str, object]] = []
        capture_packet_issues: list[str] = []
        gap_count = 0
        if capture is not None:
            capture_sync = capture.sync
            assert capture_sync is not None
            assigned = [
                packet
                for packet in packets_by_capture[capture.capture_id]
                if assignments.get((capture.capture_id, int(packet["packet_index"]))) is event
            ]
            row_issues = packet_issues_by_capture[capture.capture_id]
            for packet in assigned:
                index = int(packet["packet_index"])
                capture_packet_issues.extend(row_issues.get(index, []))

            capture_packets = packets_by_capture[capture.capture_id]
            previous_time = capture_sync.map_time(capture.capture_start_s)
            for packet in capture_packets:
                packet_time = float(packet["event_time_s"])
                if int(packet["gap_before"]):
                    gap_start = previous_time - capture_sync.uncertainty_s
                    gap_end = packet_time + capture_sync.uncertainty_s
                    packet_is_in_event = (
                        assignments.get((capture.capture_id, int(packet["packet_index"]))) is event
                    )
                    if packet_is_in_event or _overlaps(
                        gap_start, gap_end, event.start_s, event.end_s
                    ):
                        gap_count += 1
                previous_time = packet_time
        boundary_count = len(possible_boundaries.get(event.identity, []))
        crc_failure_count = sum(not bool(packet["crc_ok"]) for packet in assigned)
        if boundary_count:
            event_issues.append(f"{boundary_count} packet(s) intersect an uncertain event boundary")
        if gap_count:
            event_issues.append(f"{gap_count} declared capture gap(s) overlap the event")
        if crc_failure_count:
            event_issues.append(f"{crc_failure_count} CRC-failed PDU(s) occur in the event")
        if capture_packet_issues:
            event_issues.append(
                f"{len(capture_packet_issues)} packet-integrity issue(s) occur in the event"
            )

        valid_packets = [packet for packet in assigned if bool(packet["crc_ok"])]
        initiator_packets = [
            packet for packet in valid_packets if packet["direction"] == "initiator_to_responder"
        ]
        responder_packets = [
            packet for packet in valid_packets if packet["direction"] == "responder_to_initiator"
        ]
        remote_started = parse_bool(event.remote_session_started)
        success = parse_bool(event.success)
        requires_both_roles = (
            event.event_path in ACCEPTED_PATHS and remote_started is True and success is True
        )
        role_coverage_ok = not requires_both_roles or bool(initiator_packets and responder_packets)
        if not role_coverage_ok:
            event_issues.append(
                "successful remote session has no CRC-valid LL PDU in one or both directions"
            )
        unique = [packet for packet in valid_packets if not bool(packet["is_retransmission"])]
        retries = [packet for packet in valid_packets if bool(packet["is_retransmission"])]
        total_bytes = sum(int(packet["ll_pdu_bytes"]) for packet in valid_packets)
        unique_bytes = sum(int(packet["ll_pdu_bytes"]) for packet in unique)
        retry_bytes = sum(int(packet["ll_pdu_bytes"]) for packet in retries)
        row: dict[str, object] = {
            "run_id": event.run_id,
            "pair_id": event.pair_id,
            "exchange_id": event.exchange_id,
            "event_path": event.event_path,
            "role": event.role,
            "connection_state": event.connection_state,
            "trial_id": event.trial_id,
            "event_index": event.event_index,
            "marker_event_index": event.marker_event_index,
            "warmup": event.warmup,
            "post_warmup": event.post_warmup,
            "success": event.success,
            "remote_session_started": event.remote_session_started,
            "join_status": event.join_status,
            "start_time_s": event.start_s,
            "end_time_s": event.end_s,
            "duration_s": event.end_s - event.start_s,
            "capture_id": capture.capture_id if capture else "",
            "coverage_count": len(coverage),
            "packet_count": len(valid_packets),
            "observed_packet_count": len(assigned),
            "crc_failure_count": crc_failure_count,
            "capture_gap_count": gap_count,
            "boundary_ambiguous_packet_count": boundary_count,
            "unique_packet_count": len(unique),
            "retransmission_count": len(retries),
            "link_layer_bytes": total_bytes,
            "unique_link_layer_bytes": unique_bytes,
            "retransmitted_link_layer_bytes": retry_bytes,
            "initiator_tx_link_layer_bytes": sum(
                int(packet["ll_pdu_bytes"]) for packet in initiator_packets
            ),
            "responder_tx_link_layer_bytes": sum(
                int(packet["ll_pdu_bytes"]) for packet in responder_packets
            ),
            "retransmission_packet_rate": len(retries) / len(valid_packets) if valid_packets else 0.0,
            "retransmission_byte_rate": retry_bytes / total_bytes if total_bytes else 0.0,
            "pdu_byte_definition": "ll_data_header_2_bytes_plus_length_octet_count",
            "role_coverage_ok": int(role_coverage_ok),
            "quality_valid": int(not event_issues and not global_integrity_issues),
            "quality_issues": " | ".join(event_issues),
            "canonical_event_source": str(event_windows_csv),
            "canonical_event_sha256": event_windows_sha256,
            "canonical_capture_sha256": capture.canonical_sha256 if capture else "",
            "raw_pcap_path": str(capture.raw_pcap_path) if capture else "",
            "raw_pcap_sha256": capture.raw_pcap_sha256 if capture else "",
            "run_metadata_sha256": metadata_sha256,
            "sync_anchor_sha256": sync_anchors_sha256,
        }
        row.update(dict(event.result_context))
        event_metrics.append(row)
        audits.append(
            _audit_row(
                "event_link_quality",
                "pass" if not event_issues and not global_integrity_issues else "fail",
                [*global_integrity_issues, *event_issues],
                scope="event",
                capture_id=capture.capture_id if capture else "",
                run_id=event.run_id,
                pair_id=event.pair_id,
                exchange_id=event.exchange_id,
                event_path=event.event_path,
                role=event.role,
            )
        )

    capture_metadata: list[dict[str, object]] = []
    for capture in captures:
        sync = capture.sync
        assert sync is not None
        packets = packets_by_capture[capture.capture_id]
        capture_metadata.append(
            {
                "capture_id": capture.capture_id,
                "run_id": capture.run_id,
                "pair_id": capture.pair_id,
                "canonical_source_file": str(capture.source_file),
                "canonical_csv_sha256": capture.canonical_sha256,
                "raw_pcap_path": str(capture.raw_pcap_path),
                "raw_pcap_sha256": capture.raw_pcap_sha256,
                "event_windows_source_file": str(event_windows_csv),
                "event_windows_sha256": event_windows_sha256,
                "run_metadata_source_file": str(metadata_csv),
                "run_metadata_sha256": metadata_sha256,
                "sync_anchor_source_file": str(sync_anchors_csv) if sync_anchors_csv else "",
                "sync_anchor_sha256": sync_anchors_sha256,
                "packet_count": len(packets),
                "capture_start_sniffer_s": capture.capture_start_s,
                "capture_end_sniffer_s": capture.capture_end_s,
                "capture_start_event_s": sync.map_time(capture.capture_start_s) if sync.valid else None,
                "capture_end_event_s": sync.map_time(capture.capture_end_s) if sync.valid else None,
                "sync_method": sync.method,
                "sync_slope": sync.slope,
                "sync_intercept_s": sync.intercept_s,
                "sync_uncertainty_s": sync.uncertainty_s,
                "sync_anchor_count": sync.anchor_count,
                "sync_anchor_start_sniffer_s": sync.anchor_start_s,
                "sync_anchor_end_sniffer_s": sync.anchor_end_s,
                "maximum_anchor_residual_s": sync.maximum_anchor_residual_s,
                "assigned_packet_count": sum(
                    packet.get("assignment_status") == "matched_exact_event_window"
                    for packet in packets
                ),
                "outside_event_packet_count": sum(
                    packet.get("assignment_status") == "outside_event_windows"
                    for packet in packets
                ),
                "boundary_ambiguous_packet_count": sum(
                    packet.get("assignment_status") == "boundary_ambiguous"
                    for packet in packets
                ),
                "crc_failure_count": sum(not bool(packet["crc_ok"]) for packet in packets),
                "declared_gap_count": sum(bool(packet["gap_before"]) for packet in packets),
                "packet_integrity_issue_count": sum(
                    len(values) for values in packet_issues_by_capture[capture.capture_id].values()
                ),
                "sync_valid": int(sync.valid),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "link_packet_metrics.csv",
        [packet for capture in captures for packet in packets_by_capture[capture.capture_id]],
        preferred=[
            "capture_id", "run_id", "pair_id", "packet_index", "source_row",
            "sniffer_time_s", "event_time_s", "time_uncertainty_s", "assignment_status",
            "possible_exchange_ids", "exchange_id", "event_path", "event_role", "trial_id",
            "event_index", "warmup", "post_warmup", "direction", "transmitter_address",
            "transmitter_address_type", "receiver_address", "receiver_address_type", "pdu_kind",
            "connection_epoch", "llid", "sn", "nesn", "length_bytes", "ll_pdu_bytes",
            "crc_ok", "gap_before", "is_retransmission", "payload_sha256",
            "packet_integrity_valid", "packet_integrity_issues",
            "canonical_source_file", "raw_pcap_path", "raw_pcap_sha256",
        ],
    )
    write_csv_rows(
        output_dir / "event_link_metrics.csv",
        event_metrics,
        preferred=[
            "run_id", "pair_id", "exchange_id", "event_path", "role", "connection_state",
            "trial_id", "event_index", "marker_event_index", "warmup", "post_warmup",
            "success", "remote_session_started", "join_status", "start_time_s", "end_time_s",
            "duration_s", "capture_id", "coverage_count", "packet_count",
            "observed_packet_count", "crc_failure_count", "capture_gap_count",
            "boundary_ambiguous_packet_count", "unique_packet_count", "retransmission_count",
            "link_layer_bytes", "unique_link_layer_bytes", "retransmitted_link_layer_bytes",
            "initiator_tx_link_layer_bytes", "responder_tx_link_layer_bytes",
            "retransmission_packet_rate", "retransmission_byte_rate", "pdu_byte_definition",
            "role_coverage_ok", "quality_valid", "quality_issues", "canonical_event_source",
            "canonical_event_sha256",
            "canonical_capture_sha256", "raw_pcap_path", "raw_pcap_sha256",
            "run_metadata_sha256", "sync_anchor_sha256", *RESULT_CONTEXT_FIELDS,
        ],
    )
    write_csv_rows(
        output_dir / "link_capture_metadata.csv",
        capture_metadata,
        preferred=[
            "capture_id", "run_id", "pair_id", "canonical_source_file",
            "canonical_csv_sha256", "raw_pcap_path", "raw_pcap_sha256", "packet_count",
            "event_windows_source_file", "event_windows_sha256", "run_metadata_source_file",
            "run_metadata_sha256", "sync_anchor_source_file", "sync_anchor_sha256",
            "capture_start_sniffer_s", "capture_end_sniffer_s", "capture_start_event_s",
            "capture_end_event_s", "sync_method", "sync_slope", "sync_intercept_s",
            "sync_uncertainty_s", "sync_anchor_count", "sync_anchor_start_sniffer_s",
            "sync_anchor_end_sniffer_s", "maximum_anchor_residual_s", "assigned_packet_count",
            "outside_event_packet_count", "boundary_ambiguous_packet_count",
            "crc_failure_count", "declared_gap_count", "packet_integrity_issue_count",
            "sync_valid",
        ],
    )
    write_csv_rows(
        output_dir / "link_audit.csv",
        audits,
        preferred=[
            "audit_type", "scope", "status", "capture_id", "run_id", "pair_id",
            "exchange_id", "event_path", "role", "issue_count", "issues",
        ],
    )
    return (
        [packet for capture in captures for packet in packets_by_capture[capture.capture_id]],
        event_metrics,
        capture_metadata,
        audits,
    )


def require_valid_link_audit(audits: Sequence[Mapping[str, object]]) -> None:
    """Reject any semantic defect after preserving the diagnostic CSV outputs."""

    failures = [row for row in audits if str(row.get("status", "")).strip() != "pass"]
    if failures:
        details = " | ".join(
            f"{row.get('audit_type')}[{row.get('capture_id') or row.get('exchange_id') or 'all'}]: "
            f"{row.get('issues')}"
            for row in failures
        )
        raise ValueError("link-layer audit failed: " + details)
