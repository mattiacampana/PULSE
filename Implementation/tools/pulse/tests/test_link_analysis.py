import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

from link_analysis import analyze_link, require_valid_link_audit  # noqa: E402
from figures import prepare_figure_csvs  # noqa: E402
from pulse_tools import main as pulse_tools_main  # noqa: E402
from serial_parser import parse_serial_files  # noqa: E402
from summarize import summarize_directory  # noqa: E402


INIT_ADDRESS = "aa:bb:cc:dd:ee:01"
RESP_ADDRESS = "aa:bb:cc:dd:ee:02"
PACKET_FIELDS = [
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
]


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metadata_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": "run1",
            "pair_id": "pair1",
            "board_id": "board_a",
            "role": "initiator",
            "local_identity_address": INIT_ADDRESS,
            "local_identity_address_type": "public",
            "peer_identity_address": RESP_ADDRESS,
            "peer_identity_address_type": "public",
        },
        {
            "run_id": "run1",
            "pair_id": "pair1",
            "board_id": "board_b",
            "role": "responder",
            "local_identity_address": RESP_ADDRESS,
            "local_identity_address_type": "public",
            "peer_identity_address": INIT_ADDRESS,
            "peer_identity_address_type": "public",
        },
    ]


def accepted_event_rows(remote_started: int = 1, success: int = 1) -> list[dict[str, object]]:
    join = (
        "paired_remote_session_exact_exchange_id"
        if remote_started
        else "initiator_only_pre_session_failure_exact_exchange_id"
    )
    common = {
        "run_id": "run1",
        "pair_id": "pair1",
        "trial_id": 7,
        "exchange_id": 107,
        "event_path": "accepted_connected" if remote_started else "accepted_discovery",
        "connection_state": "connected" if remote_started else "disconnected",
        "warmup": 0,
        "success": success,
        "remote_session_started": remote_started,
        "truncated_start": 0,
        "truncated_end": 0,
    }
    rows = [
        {
            **common,
            "board_id": "board_a",
            "role": "initiator",
            "event_index": 11,
            "start_time_s": 1.05,
            "end_time_s": 1.95,
            "join_status": "matched",
        }
    ]
    if remote_started:
        rows.append(
            {
                **common,
                "board_id": "board_b",
                "role": "responder",
                "event_index": 19,
                "start_time_s": 1.1,
                "end_time_s": 1.9,
                "join_status": "matched",
            }
        )
    rows.append(
        {
            **common,
            "board_id": "",
            "role": "pair",
            "event_index": 11,
            "start_time_s": 1.0,
            "end_time_s": 2.0,
            "join_status": join,
        }
    )
    return rows


def data_packet(
    index: int,
    timestamp_s: float,
    direction: str,
    sn: int,
    length: int,
    digest_digit: str,
    *,
    retransmission: int = 0,
    capture_start_s: float = 0.0,
    capture_end_s: float = 3.0,
) -> dict[str, object]:
    initiator_tx = direction == "initiator_to_responder"
    return {
        "capture_id": "cap1",
        "run_id": "run1",
        "pair_id": "pair1",
        "packet_index": index,
        "timestamp_s": timestamp_s,
        "capture_start_s": capture_start_s,
        "capture_end_s": capture_end_s,
        "direction": direction,
        "transmitter_address": INIT_ADDRESS if initiator_tx else RESP_ADDRESS,
        "transmitter_address_type": "public",
        "receiver_address": RESP_ADDRESS if initiator_tx else INIT_ADDRESS,
        "receiver_address_type": "public",
        "pdu_kind": "data",
        "llid": 2,
        "length_bytes": length,
        "crc_ok": 1,
        "gap_before": 0,
        "is_retransmission": retransmission,
        "connection_epoch": "conn1",
        "sn": sn,
        "nesn": 0,
        "payload_sha256": digest_digit * 64,
    }


def advertising_packet(timestamp_s: float = 1.5) -> dict[str, object]:
    return {
        "capture_id": "cap1",
        "run_id": "run1",
        "pair_id": "pair1",
        "packet_index": 1,
        "timestamp_s": timestamp_s,
        "capture_start_s": 0.0,
        "capture_end_s": 3.0,
        "direction": "responder_to_initiator",
        "transmitter_address": RESP_ADDRESS,
        "transmitter_address_type": "public",
        "receiver_address": "",
        "receiver_address_type": "",
        "pdu_kind": "advertising",
        "llid": "",
        "length_bytes": 8,
        "crc_ok": 1,
        "gap_before": 0,
        "is_retransmission": 0,
        "connection_epoch": "",
        "sn": "",
        "nesn": "",
        "payload_sha256": "d" * 64,
    }


class LinkAnalysisTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, Path, Path]:
        packets = root / "packets.csv"
        events = root / "event_power_metrics.csv"
        metadata = root / "run_metadata.csv"
        pcap = root / "capture.pcapng"
        pcap.write_bytes(b"raw-pcap-provenance\x00\x01")
        write_rows(metadata, metadata_rows())
        return packets, events, metadata, pcap

    def test_shared_timebase_accounts_ll_header_retries_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            rows = [
                data_packet(1, 1.2, "initiator_to_responder", 0, 10, "a"),
                data_packet(2, 1.3, "responder_to_initiator", 0, 20, "b"),
                data_packet(
                    3,
                    1.4,
                    "initiator_to_responder",
                    0,
                    10,
                    "a",
                    retransmission=1,
                ),
                data_packet(4, 1.5, "initiator_to_responder", 1, 5, "c"),
            ]
            write_rows(packets, rows, PACKET_FIELDS)
            write_rows(events, accepted_event_rows())

            packet_metrics, event_metrics, capture_metadata, audits = analyze_link(
                [packets],
                events,
                metadata,
                root / "out",
                [f"cap1={pcap}"],
                shared_timebase=True,
                time_uncertainty_s=0.001,
            )
            require_valid_link_audit(audits)

            self.assertEqual(4, len(packet_metrics))
            self.assertEqual(
                "ll_data_header_2_bytes_plus_length_octet_count",
                event_metrics[0]["pdu_byte_definition"],
            )
            self.assertEqual(53, event_metrics[0]["link_layer_bytes"])
            self.assertEqual(41, event_metrics[0]["unique_link_layer_bytes"])
            self.assertEqual(12, event_metrics[0]["retransmitted_link_layer_bytes"])
            self.assertEqual(31, event_metrics[0]["initiator_tx_link_layer_bytes"])
            self.assertEqual(22, event_metrics[0]["responder_tx_link_layer_bytes"])
            self.assertEqual(1, event_metrics[0]["retransmission_count"])
            self.assertEqual(1, event_metrics[0]["quality_valid"])
            self.assertEqual(hashlib.sha256(pcap.read_bytes()).hexdigest(), capture_metadata[0]["raw_pcap_sha256"])
            self.assertEqual("shared_timebase", capture_metadata[0]["sync_method"])
            self.assertEqual(4, len(read_rows(root / "out" / "link_packet_metrics.csv")))
            self.assertEqual(1, len(read_rows(root / "out" / "event_link_metrics.csv")))

            summary = root / "summary"
            summarize_directory([root / "out"], summary, min_repetitions=1)
            link_summary = read_rows(summary / "link_summary.csv")
            self.assertTrue(
                any(
                    row["metric"] == "link_layer_bytes" and row["median"] == "53"
                    for row in link_summary
                )
            )
            figures = root / "figures"
            prepare_figure_csvs([summary], figures)
            self.assertTrue(
                any(row["metric"] == "link_layer_bytes" for row in read_rows(figures / "figure_link.csv"))
            )

    def test_serial_parser_preserves_ble_identity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "board.log"
            log.write_text(
                "PULSE_META,run_id=run1,pair_id=pair1,board_id=board_a,role=initiator,"
                f"local_identity_address={INIT_ADDRESS},local_identity_address_type=public,"
                f"peer_identity_address={RESP_ADDRESS},peer_identity_address_type=public,"
                "build_guard=final_capture_requirements_enforced\n",
                encoding="utf-8",
            )
            parse_serial_files([log], root / "parsed")
            row = read_rows(root / "parsed" / "run_metadata.csv")[0]
            self.assertEqual(INIT_ADDRESS, row["local_identity_address"])
            self.assertEqual(RESP_ADDRESS, row["peer_identity_address"])
            self.assertEqual("final_capture_requirements_enforced", row["build_guard"])

    def test_affine_sync_uses_bracketing_anchors_and_explicit_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            rows = [
                data_packet(1, 12.0, "initiator_to_responder", 0, 10, "a", capture_end_s=30),
                data_packet(2, 13.0, "responder_to_initiator", 0, 10, "b", capture_end_s=30),
            ]
            write_rows(packets, rows, PACKET_FIELDS)
            write_rows(events, accepted_event_rows())
            anchors = root / "anchors.csv"
            write_rows(
                anchors,
                [
                    {"capture_id": "cap1", "sniffer_time_s": 0, "event_time_s": 0, "uncertainty_s": 0.0005},
                    {"capture_id": "cap1", "sniffer_time_s": 30, "event_time_s": 3, "uncertainty_s": 0.0005},
                ],
            )

            packet_metrics, event_metrics, capture_metadata, audits = analyze_link(
                [packets],
                events,
                metadata,
                root / "out",
                [f"cap1={pcap}"],
                sync_anchors_csv=anchors,
            )
            require_valid_link_audit(audits)
            self.assertAlmostEqual(1.2, float(packet_metrics[0]["event_time_s"]))
            self.assertAlmostEqual(0.1, float(capture_metadata[0]["sync_slope"]))
            self.assertEqual(2, capture_metadata[0]["sync_anchor_count"])
            self.assertEqual(1, event_metrics[0]["quality_valid"])

    def test_pre_session_failure_is_exact_initiator_only_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            write_rows(packets, [advertising_packet()], PACKET_FIELDS)
            write_rows(events, accepted_event_rows(remote_started=0, success=0))

            _, event_metrics, _, audits = analyze_link(
                [packets],
                events,
                metadata,
                root / "out",
                [f"cap1={pcap}"],
                shared_timebase=True,
                time_uncertainty_s=0.0001,
            )
            require_valid_link_audit(audits)
            self.assertEqual(1, len(event_metrics))
            self.assertEqual("0", str(event_metrics[0]["remote_session_started"]))
            self.assertEqual(
                "initiator_only_pre_session_failure_exact_exchange_id",
                event_metrics[0]["join_status"],
            )
            self.assertEqual(1, event_metrics[0]["quality_valid"])

    def test_boundary_uncertainty_is_a_hard_audit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            packet = advertising_packet(timestamp_s=1.0005)
            write_rows(packets, [packet], PACKET_FIELDS)
            local_event = {
                "run_id": "run1", "pair_id": "pair1", "exchange_id": 5,
                "event_path": "local", "role": "local", "connection_state": "connected",
                "trial_id": 0, "event_index": 0, "warmup": 0, "success": 1,
                "remote_session_started": 0, "join_status": "matched",
                "start_time_s": 1.0, "end_time_s": 2.0,
                "truncated_start": 0, "truncated_end": 0,
            }
            write_rows(events, [local_event])

            packet_metrics, event_metrics, _, audits = analyze_link(
                [packets],
                events,
                metadata,
                root / "out",
                [f"cap1={pcap}"],
                shared_timebase=True,
                time_uncertainty_s=0.001,
            )
            self.assertEqual("boundary_ambiguous", packet_metrics[0]["assignment_status"])
            self.assertEqual(0, event_metrics[0]["quality_valid"])
            with self.assertRaisesRegex(ValueError, "link-layer audit failed"):
                require_valid_link_audit(audits)
            summary = root / "summary"
            summarize_directory([root / "out"], summary, min_repetitions=1)
            summary_rows = read_rows(summary / "link_summary.csv")
            self.assertEqual("1", summary_rows[0]["quality_excluded_count"])
            self.assertEqual("0", summary_rows[0]["n"])

    def test_crc_gap_duplicate_and_role_errors_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            first = data_packet(1, 1.2, "initiator_to_responder", 0, 10, "a")
            first["crc_ok"] = 0
            first["gap_before"] = 1
            first["transmitter_address"] = RESP_ADDRESS
            second = data_packet(1, 1.3, "initiator_to_responder", 0, 11, "b")
            write_rows(packets, [first, second], PACKET_FIELDS)
            write_rows(events, accepted_event_rows())

            _, event_metrics, capture_metadata, audits = analyze_link(
                [packets],
                events,
                metadata,
                root / "out",
                [f"cap1={pcap}"],
                shared_timebase=True,
                time_uncertainty_s=0.0001,
            )
            self.assertEqual(0, event_metrics[0]["quality_valid"])
            self.assertGreater(int(capture_metadata[0]["packet_integrity_issue_count"]), 0)
            self.assertEqual(1, capture_metadata[0]["crc_failure_count"])
            self.assertEqual(1, capture_metadata[0]["declared_gap_count"])
            with self.assertRaises(ValueError):
                require_valid_link_audit(audits)

    def test_uncovered_event_and_unbracketed_affine_sync_fail_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            packet = advertising_packet(timestamp_s=1.25)
            packet["capture_start_s"] = 0.5
            packet["capture_end_s"] = 1.5
            write_rows(packets, [packet], PACKET_FIELDS)
            write_rows(events, accepted_event_rows(remote_started=0, success=0))
            anchors = root / "anchors.csv"
            write_rows(
                anchors,
                [
                    {"capture_id": "cap1", "sniffer_time_s": 0.75, "event_time_s": 0.75, "uncertainty_s": 0.001},
                    {"capture_id": "cap1", "sniffer_time_s": 1.25, "event_time_s": 1.25, "uncertainty_s": 0.001},
                ],
            )

            _, event_metrics, metadata_rows_out, audits = analyze_link(
                [packets],
                events,
                metadata,
                root / "out",
                [f"cap1={pcap}"],
                sync_anchors_csv=anchors,
            )
            self.assertEqual(0, metadata_rows_out[0]["sync_valid"])
            self.assertEqual(0, event_metrics[0]["coverage_count"])
            self.assertEqual(0, event_metrics[0]["quality_valid"])
            with self.assertRaises(ValueError):
                require_valid_link_audit(audits)

    def test_missing_raw_capture_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, _ = self._paths(root)
            write_rows(packets, [advertising_packet()], PACKET_FIELDS)
            write_rows(events, accepted_event_rows(remote_started=0, success=0))
            with self.assertRaisesRegex(ValueError, "no raw PCAP provenance"):
                analyze_link(
                    [packets],
                    events,
                    metadata,
                    root / "out",
                    [],
                    shared_timebase=True,
                    time_uncertainty_s=0.001,
                )

    def test_link_cli_writes_outputs_and_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets, events, metadata, pcap = self._paths(root)
            write_rows(packets, [advertising_packet()], PACKET_FIELDS)
            write_rows(events, accepted_event_rows(remote_started=0, success=0))
            output = root / "out"
            result = pulse_tools_main(
                [
                    "link",
                    "--input",
                    str(packets),
                    "--event-windows-csv",
                    str(events),
                    "--metadata-csv",
                    str(metadata),
                    "--raw-pcap",
                    f"cap1={pcap}",
                    "--shared-timebase",
                    "--time-uncertainty-s",
                    "0.0001",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(0, result)
            self.assertTrue((output / "link_audit.csv").is_file())


if __name__ == "__main__":
    unittest.main()
