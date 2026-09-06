# V24.1 Principal Audit — V10 Realtime Tracking Stabilization

## 1. Executive summary
V10 is a focused stabilization pass on the existing V24.1 architecture. PPO is not modified or trained. RT-DETR remains an asynchronous CPU global re-detector. The fast path now emphasizes persistent player/active/ball tracking and throttled global CV work.

The V9 live test exposed three runtime facts: RT-DETR inference completed without errors but was slow on the Ryzen CPU; model results were often 1+ seconds old; and fast-state validity was dominated by ball/active continuity loss. V10 addresses those failure modes without pretending the CPU detector is real-time.

## 2. Current architecture
Capture -> PerceptionEngine fast loop -> persistent PlayerTracker / ActiveMarker / BallManager -> WorldModel / ControlGate.
RT-DETR runs separately with latest-frame semantics and source timestamps. Its role is global player/ball re-detection and correction, not every-frame control input.

## 3. Confirmed defects from V9
- RT-DETR CPU inference latency was ~0.77 s on the reported run; completed 14/21 submissions, errors 0.
- Model age averaged ~1.36 s and reached ~1.89 s.
- Ball valid rate was ~0.20; detector-ball rate was 0; prediction/CV/unknown dominated.
- Active valid rate was ~0.85, but active association could still drop to zero for periods.
- Track count oscillated materially and could exceed the intended roster size because unmatched detections could create duplicate identities.
- Live OCR could observe multiple score states; continuous score is kept separate from one-frame observations.

## 4. Root causes
- RT-DETR is computationally heavy on CPU relative to the requested 10 Hz observation loop.
- PlayerTracker spawn logic was too permissive when detector boxes moved and CV recovery coexisted with recent detector observations.
- BallManager performed expensive global CV scans too often and required repeated global evidence more frequently than a persistent local track should need.
- Prediction was being used too often because local reacquisition was not sufficiently privileged.

## 5. Changed files
- `efootball_agent/config.py`: V10 tracking/reacquisition and bounded track settings.
- `efootball_agent/perception/pipeline.py`: throttled CV recovery while preserving fast tracking.
- `efootball_agent/perception/tracker.py`: duplicate-spawn suppression, explicit track cap, deterministic existing-track proximity gate.
- `efootball_agent/perception/ball.py`: fast local ball reacquisition around the previous trusted measurement, throttled global CV search, longer but still bounded prediction bridge.
- `efootball_agent/tests/test_core.py`: regression for local ball reacquisition.

## 6. P0 fixes
### IMPLEMENTED
- RT-DETR remains non-blocking and timestamped.
- Latest-frame queue semantics retained.
- Player tracking persists identities between global detections.
- Duplicate track spawning is suppressed.
- Track count is bounded.
- Active-marker path remains independent of detector freshness.
- Ball local reacquisition is now preferred after a trusted ball measurement.
- Global CV ball search is throttled.
- Prediction is bounded and remains explicitly `PREDICTED`, never detector-confirmed.
- Control remains fail-closed.

### HEURISTIC
- Player optical-flow continuity.
- Ball local white-blob reacquisition.
- Foot proximity and possession scoring.
- Team appearance smoothing.

### ML INFERENCE
- RT-DETR R50vd, existing model assets.

### NOT TRAINED
- PPO / RL policy.

### PARTIAL
- True realtime world/control stability still requires a fresh Windows live run on V10.

## 7. Tests
Local project tests after V10 changes:

`52 passed`

## 8. Real-frame / runtime evidence used
V9 Windows live test showed:
- `observed_hz=6.6` despite a requested 10 Hz.
- `model_fresh_rate=0`.
- `model_recent_rate≈0.409`.
- `model_age_mean_s≈1.357`.
- `model_age_max_s≈1.891`.
- `model_inference_latency_mean_s≈0.771`.
- `model_completed=14`, `model_errors=0`.
- `active_valid_rate≈0.848`.
- `ball_valid_rate≈0.197`, `ball_detector_rate=0`.
- `world_valid_rate≈0.152`, `control_valid_rate≈0.136`.

These values are the reason V10 does not treat RT-DETR freshness as a per-frame requirement.

## 9. BEFORE vs AFTER design intent
Before V10:
- global CV and recovery were more frequent;
- unmatched detections could create duplicate track identities;
- ball reacquisition was too dependent on global scans;
- prediction became a frequent fallback.

After V10:
- persistent tracks are canonical;
- RT-DETR is a slower global correction layer;
- active marker follows existing tracks;
- ball uses local reacquisition first and global search less often;
- control only receives valid world state.

## 10. Remaining limitations
- CPU RT-DETR remains much slower than the fast perception loop.
- Ball CV can still be ambiguous in severe occlusion or when the ball visually merges with feet/boots.
- Team classification can remain UNKNOWN for some players.
- Possession is still heuristic and should be validated over longer live sequences.
- A real V10 live stability run is still required before PPO rollout.

## 11. Exact launch commands
```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check_v10
```

Do not launch `brain.py` for this validation. It is observation-only acceptance first; PPO training remains disabled until the V10 live metrics are acceptable.
