# V24.1 Principal Audit — V12 Async-Track Stabilization

## 1. Scope
V12 is a focused continuation of the existing V24.1/V11 architecture. No new entrypoint is introduced; `brain.py` remains the project entrypoint. The implementation remains screen-based and modular. PPO is not trained, tuned, or authorized for live rollout.

## 2. V11 Windows live evidence
The user-run V11 observation-only check completed successfully on the real `eFootball™` client capture at 1280x720. RT-DETR stayed on CPU with zero worker errors.

Reported live results:
- observed_hz: 10.0
- model_completed: 13
- model_errors: 0
- model_recent_rate: 0.41
- ball_valid_rate: 0.92
- ball_predicted_rate: 0.06
- active_valid_rate: 0.69
- players_valid_rate: 0.52
- world_valid_rate: 0.09
- control_valid_rate: 0.09
- RT-DETR mean inference latency: ~0.771 s
- RT-DETR mean age: ~1.374 s
- score invalid transitions: 0
- score stable values observed: 0:0 and 0:1

Interpretation: the ball and score state machines improved materially, but the live bottleneck moved to player-track continuity, active-player retention, and the world/control gate. The log shows player counts collapsing from roughly 7–18 down to 0–2 whenever the async detector result aged out. That pattern is consistent with tracks being retired too quickly for a ~0.77 s detector cadence.

## 3. V12 changes

### 3.1 Persistent player lifecycle across RT-DETR gaps — IMPLEMENTED
`TrackingConfig.max_missed` is increased to a bounded 20-frame grace window at the 10 Hz decision loop. LOST tracks remain available for a finite period rather than being deleted after less than one second. Confidence decays during gaps, so this is bounded persistence, not permanent ghost retention.

### 3.2 Stale RT-DETR is identity evidence, not current geometry — IMPLEMENTED
When RT-DETR is within the bounded reuse window but older than the fresh threshold, pipeline detections are marked `RTDETR_STALE`. The tracker blends these old detector positions with current-frame optical-flow prediction instead of snapping tracks back to stale coordinates.

### 3.3 Active-player association over short LOST intervals — IMPLEMENTED
The active marker can associate to tracks with a small bounded `missed` value rather than rejecting every track immediately after a detector gap. Existing-player confidence decays more slowly during brief marker loss. Switch hysteresis remains in force.

### 3.4 World/control validity uses maintained tracks — IMPLEMENTED
The world model now allows short LOST intervals for player continuity. Own/opp/unknown player sets use a bounded `missed <= 3` window. `world_valid` and `control_valid` use a stable maintained player count instead of requiring `missed == 0` on every frame. The fail-closed gate remains: trusted ball, active player, possession owner, and sufficient maintained players are still required.

### 3.5 Ball and score behavior — PRESERVED
V11 ball state machine and score temporal validator are retained. No broadening of ball acceptance or removal of score sanity checks was made.

## 4. Known limitations / honest status
- CPU RT-DETR remains slow relative to the 10 Hz perception loop. This build does not pretend detector freshness is real-time.
- Team classification remains heuristic and can limit possession/world validity even when player geometry is available.
- A fresh V12 Windows live run is required before claiming target metrics are met.
- Score OCR was temporally consistent in the supplied V11 run, but the semantic correctness of `0:1` was not independently verified from the live log alone.
- PPO remains `NOT TRAINED` and must stay out of live control until perception/world/control acceptance is demonstrated.

## 5. Regression verification
`pytest -q` on the V12 source tree: **58 passed**.

A focused tracker regression also confirmed that five tracks survive a 1.5 s no-detection interval as bounded LOST tracks; they are still subject to confidence decay and the finite `max_missed` retirement rule.

## 6. Next acceptance run
On the Windows machine:

```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check_v12
```

Acceptance targets remain: observed_hz >= 9, ball_valid_rate >= 0.85, active_valid_rate >= 0.90, players_valid_rate >= 0.90, world_valid_rate >= 0.75, control_valid_rate >= 0.70, score invalid transitions = 0, and model_errors = 0.
