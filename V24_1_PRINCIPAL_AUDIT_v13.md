# V24.1 Principal Audit — V13

## Scope
V13 is a bounded stabilization patch on top of V12. The architecture is preserved: capture -> perception -> tracking -> active -> ball -> world -> safety/control. `brain.py` remains the entrypoint; no `main.py` is introduced. PPO remains disabled/not trained.

## Evidence from V12 live run
The supplied 35 s live run demonstrated:
- observed_hz: 9.51
- model errors: 0
- ball_valid_rate: 0.8198
- ball_predicted_rate: 0.0781
- active_valid_rate: 0.6156
- players_valid_rate: 0.7237
- world_valid_rate: 0.1532
- control_valid_rate: 0.1321
- model inference latency mean/max: ~0.807 / 1.203 s
- model_recent_rate: 0.3634

The important diagnostic pattern was that the published player population still dropped from roughly 12–20 to 0–3 during detector-age gaps. Inspection showed that the tracker itself retained bounded tracks (`TrackingConfig.max_missed=20`), but the world-facing player lists and active association still used a much shorter `missed <= 3` horizon.

## V13 changes
1. Extend RT-DETR player reuse window from 1.35 s to 1.80 s. This is still a bounded stale-observation window and stale detections remain marked `RTDETR_STALE`.
2. Extend active-marker missing tolerance from 4 to 8 frames.
3. Allow active association against maintained tracks up to the same bounded 8-frame gap.
4. Extend world-facing own/opp/unknown player retention from 3 to 8 missed frames.
5. Extend world consistency checks to the same 8-frame carried-track window.
6. Keep tracker hard deletion bounded by `TrackingConfig.max_missed=20`; no unlimited ghost tracks are introduced.
7. Add live-check diagnostics showing `players=carried/total_tracks` so future runs distinguish world-facing carried players from raw maintained tracker identities.
8. Add regression tests for active association across a bounded lost-track gap and the extended reuse settings.

## Status
- IMPLEMENTED: bounded player carry window, extended active hysteresis, wider stale RT-DETR identity reuse, score validator from V11/V12, CPU-only scheduler.
- HEURISTIC: team classification, CV player recovery, active marker geometry, possession, tactical state.
- ML INFERENCE: RT-DETR on CPU as asynchronous global re-detector.
- NOT TRAINED: PPO.
- PARTIAL: real-time world/control readiness; V13 still requires a new Windows live run.

## Acceptance targets
The next live run should be judged primarily on continuity, not only averages:
- observed_hz >= 9
- player carried count should not repeatedly collapse to 0–3 during normal RT-DETR age gaps
- active_valid_rate >= 0.80 initially, then >= 0.90 target
- world_valid_rate >= 0.50 initially, then >= 0.75 target
- control_valid_rate >= 0.50 initially, then >= 0.70 target
- model_errors = 0
- score_invalid_transition_count = 0

A new live run is required before declaring control-ready.


## V14 focus
Active-player recovery now includes bounded local cyan evidence around tracked player heads, plus slower bounded confidence decay during short marker gaps. This is intended to address live active dropouts while keeping the fail-closed control gate.
