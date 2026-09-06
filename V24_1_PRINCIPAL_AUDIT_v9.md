# V24.1 Principal Audit — V9 Realtime Tracking Stabilization

## 1. Executive summary
V9 keeps the existing V24 modular architecture and addresses realtime continuity rather than changing the RL stack.

The core design is now a hybrid tracker:
- RT-DETR = asynchronous global re-detection / correction.
- Player tracker = deterministic Hungarian assignment + constant-velocity prediction + local Lucas-Kanade optical flow for short detection gaps.
- Ball = detector/CV evidence + short-lived prediction + lightweight alpha-beta filtering.
- Active player = existing cyan marker association with temporal hysteresis, now backed by more persistent tracks.
- Possession = footpoint-aware distance instead of track-center-only distance.

This is intentionally optimized for the supplied AMD/Windows machine profile: Ryzen 7 5700X3D + RX 6600 + 32 GB RAM. The project remains CPU-only; no CUDA/NVIDIA dependency was added.

## 2. Confirmed V8 runtime baseline
The supplied V8 live test demonstrated:
- observed_hz = 10.0
- RT-DETR completed = 15
- RT-DETR errors = 0
- mean RT-DETR latency ≈ 0.67 s
- mean model age ≈ 1.23 s
- max model age ≈ 1.66 s
- model_recent_rate = 0.60
- ball_valid_rate = 0.24
- active_valid_rate = 0.79
- world_valid_rate = 0.19
- control_valid_rate = 0.16

The key conclusion was that RT-DETR must not be the per-frame dependency for realtime control.

## 3. V9 changes
### IMPLEMENTED
- Persistent player tracks are updated between detector refreshes with local optical flow.
- Track bounding boxes are translated with successful flow so active-player association and possession use current geometry.
- CV_RECOVERY no longer inflates the roster once the detector has recently supplied a global roster; it is treated as gap-filling evidence.
- Player association keeps deterministic Hungarian assignment and stable track IDs.
- Ball measurements use a lightweight alpha-beta constant-velocity filter.
- Existing-ball local continuity is preferred over global CV candidates when a prior measurement exists.
- Possession is evaluated at estimated footpoint rather than body-center position.
- V9 exposes flow telemetry: `flow_tracks` and `flow_usable`.

### HEURISTIC
- Lucas-Kanade optical flow can drift on grass texture/lines and is therefore bounded by missed-frame lifecycle and confidence thresholds.
- Ball CV remains heuristic and must not be treated as ML truth.
- Possession remains an inference heuristic; no game-state API is available.

### ML INFERENCE
- RT-DETR remains the only ML detector in the perception path.

### NOT TRAINED
- PPO was not changed or trained.

### PARTIAL
- A real Windows V9 10-second run has not been performed inside this build environment. The acceptance result must therefore be measured on the user's 5700X3D machine.

## 4. Why this algorithm
The objective is low-latency control, not maximum detector FPS. A heavy detector at ~1.5 FPS should serve as a global re-anchor while lightweight local trackers run at the control cadence. This avoids waiting for RT-DETR on every decision tick.

A full Kalman/SORT/DeepSORT stack was intentionally not introduced because the existing project already had deterministic assignment and appearance/team features. V9 extends that tracker with cheap optical-flow motion between detector updates instead of replacing the architecture.

## 5. Tests
- `pytest -q` => 51 passed.
- Added regression coverage for flow-based track persistence, footpoint-aware possession, and alpha-beta ball velocity updates.

## 6. Acceptance test for Windows runtime
Run:

```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check_v9
```

Required telemetry to inspect:
- `observed_hz`
- `model_completed`
- `model_errors`
- `model_age_mean_s`
- `model_age_max_s`
- `active_valid_rate`
- `ball_valid_rate`
- `world_valid_rate`
- `control_valid_rate`

The intended contract is that the fast tracker and local CV maintain continuity between global RT-DETR updates. RT-DETR `fresh=true` on every frame is not required.

## 7. PPO readiness
Do not train PPO until the V9 live test confirms stable ball + active + player continuity. PPO should consume the canonical `WorldState`; it should not compensate for broken perception.

For later RL work, a fast policy inference path is preferred: small MLP policy at the decision rate, with RT-DETR completely decoupled from action latency.
