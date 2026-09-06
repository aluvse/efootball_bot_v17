# V24.1 Principal Audit — V14

## Scope
V14 is a focused active-player continuity patch on top of V13. The existing architecture is preserved: capture -> perception -> tracking -> active -> ball -> world -> safety/control. `brain.py` remains the entrypoint; no `main.py` is introduced. PPO remains disabled/not trained.

## Evidence from the V13 live run
The supplied 35 s Windows run showed:
- observed_hz: 8.657
- model errors: 0
- model_recent_rate: 0.855
- players_valid_rate: 0.954
- ball_valid_rate: 0.772
- ball_predicted_rate: 0.059
- active_valid_rate: 0.617
- world_valid_rate: 0.314
- control_valid_rate: 0.304
- score_values: 0:0 then 1:0
- score_invalid_transition_samples: 0
- model inference latency mean/max: ~0.798 / 1.063 s

The key pattern is now clear: player continuity is substantially improved, while active-player validity still collapses for multi-frame periods. Those active drops propagate directly into world/control invalidation. This is not being treated as permission to loosen the safety gate.

## V14 changes
1. Add bounded local cyan-marker recovery around the top/head region of maintained player tracks. This is a secondary visual measurement and does not infer the active player from distance alone.
2. Extend active-marker missing tolerance from 8 to 15 frames.
3. Slow active confidence decay during a bounded marker gap from approximately 0.90/0.93 per frame to 0.97 per frame.
4. Preserve the existing two-observation active-player switch hysteresis.
5. Preserve bounded tracker deletion (`TrackingConfig.max_missed=20`) and the V13 world-facing carried-player window.
6. Add regression tests for local marker recovery and slow bounded active hold.

## Interpretation
V13 largely solved the earlier player-list collapse: `players_valid_rate` reached 95.4%. The remaining main bottleneck is active selection continuity, not roster continuity.

Ball continuity is acceptable for diagnostic/world-state work but is not yet strong enough for a control-ready declaration; unknown-ball frames still occur and predicted-ball frames remain intentionally excluded from trusted world validity.

The V13 score counter reported 7 internal invalid-transition detections while `score_invalid_transition_samples` remained 0. The latter is the live-frame acceptance metric: no rejected score transition was exposed as an invalid live sample. This counter should remain visible for diagnostics rather than being used as a readiness pass by itself.

## Status
- IMPLEMENTED: bounded active-marker local recovery, active hold hysteresis, player carry window, async CPU RT-DETR, ball state machine, score temporal validation, fail-closed safety gate.
- HEURISTIC: active marker recovery, team classification, CV ball recovery, possession, tactical state.
- ML INFERENCE: RT-DETR on CPU as asynchronous global re-detector.
- NOT TRAINED: PPO.
- PARTIAL: real-time world/control readiness. A fresh Windows live run is still required.

## Acceptance targets for the next live run
Primary continuity targets:
- observed_hz >= 9
- players_valid_rate >= 0.90
- active_valid_rate >= 0.80 initially, then >= 0.90
- world_valid_rate >= 0.50 initially, then >= 0.75
- control_valid_rate >= 0.50 initially, then >= 0.70
- model_errors = 0
- score_invalid_transition_samples = 0

Do not enable PPO rollout or unattended control until the observation-only run meets the safety/readiness criteria.
