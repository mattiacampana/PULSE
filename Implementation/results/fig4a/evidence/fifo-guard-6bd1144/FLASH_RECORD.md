# Figure 4a FIFO-guard firmware flash record

- Date: 2026-09-16 (Europe/Helsinki)
- Board: `SensWear/nrf54l15/cpuapp`
- Clean source snapshot: `6bd1144c107318e896efb1e39a6c833ebcf558f3`
- Source bundle SHA-256: `c2a1327d80c63f0458fa1773ff6f375caf50f7a532a208e3d8b10bdaf21f8268`
- Build guard: `final_capture_requirements_enforced`
- Hardware revision: `V1R1`
- Exported model artifact SHA-256: `6716da0b2cf47106d541f526f3ae258d46f820608290c7ceb563e121d41ee320`
- Firmware HEX SHA-256: `582e93a57462e4daf52d0b04588a82809a6c021d07d7fa43db61f7310191f391`
- Firmware ELF SHA-256: `2aa57a3dadde93bc9f7ddd72ffd35833aafbbae494a98ab130a9d32e384ec3aa`
- Capture mode: Keysight 34465A internal positive current-level trigger, no GPIO markers
- Campaign: 5 warm-ups + 100 measured trials per path; initiator local then accepted-connected
- Timing: 60 s start delay; 5 s minimum event start-to-start period

The only firmware change from the previous Figure 4a source snapshot is an isolated
guard in Bosch's `bhy2.c` FIFO parser: an event with size zero now returns
`BHY2_E_INVALID_EVENT_SIZE` instead of looping forever. A host regression test
for zero-length and valid events is included in this source snapshot. The active
working tree was not committed, reset, or stashed.

Both live CMSIS-DAP probes were matched to the target boards using on-chip FICR
device IDs immediately before programming. OpenOCD's nRF54L15 RRAM loader wrote
and read-back verified the same 574,060-byte HEX image on both probes, then
reset each board to run:

- Probe `576E8834454A8BEB` / COM8 / board `0aaf76ede7a33ea9` / responder.
- Probe `7D64D54AB317FE0D` / COM10 / board `f5ea5fa8fcfa5209` / initiator.

This is **not yet a UART-audited paper preflight**. A read-only debugger pilot
showed the initiator advance through all 105 local trials and all 105
accepted-connected trials. The responder processed connected exchanges. The
initiator entered its terminal sleep after accepted-connected iteration 104;
its campaign measured-failure counter was zero, and the final event record had
`success=1` with zero work, infrastructure, and result-release errors. The
BHI360 quaternion, acceleration, and gyro timestamps advanced during the run,
and the IMU event queue remained empty at progress checks. This verifies that
the previous scheduler stall is gone, but it does not replace the full UART
row-count and failure audit. The old stalled capture is invalid. Preserve both
complete UART logs and verify every row before using DMM data for the paper.
