# Figure 4a markerless firmware flash record

- Timestamp: `2026-09-15T18:09:33.9190742+03:00`
- Board: `SensWear/nrf54l15/cpuapp`
- Source snapshot: `1164e3356e3d6c0e54f8ac46522ba65a1cf8fb15`
- Source snapshot bundle SHA-256: `4fb3d1f5191cfc9280eca2944d3011b085c27535fa786e372bc03442b05a21dc`
- Build guard: `final_capture_requirements_enforced`
- Hardware revision: `V1R1`
- Exported artifact SHA-256: `6716da0b2cf47106d541f526f3ae258d46f820608290c7ceb563e121d41ee320`
- Firmware HEX SHA-256: `68df553482b6ae0cb09d00968c05493a634bc19596cc096ffe95dbabf73f937d`
- Firmware ELF SHA-256: `a1a685fea1fea8ba947e3720d9ce57c78f6ba213cae466d5dd4055d27aba225c`
- Capture mode: Keysight 34465A internal positive current-level trigger; no GPIO markers
- Campaign: 5 warm-ups + 100 measured trials per path; initiator local then accepted-connected
- Timing: 60 s start delay; 5 s minimum event start-to-start period

Programming completed successfully through OpenOCD and the repository's nRF54L15 RRAM loader:

- CMSIS-DAP probe `576E8834454A8BEB` / COM8 / board `0aaf76ede7a33ea9` /
  responder: 574,064 bytes written; reset-and-run succeeded.
- CMSIS-DAP probe `7D64D54AB317FE0D` / COM10 / board `f5ea5fa8fcfa5209` /
  initiator: 574,064 bytes written; reset-and-run succeeded.

Both probes received the same `zephyr.hex`. The source snapshot was created in an isolated Git clone;
the active branch, index, and working files were not committed, reset, or stashed.
