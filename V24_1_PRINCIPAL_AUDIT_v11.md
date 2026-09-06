# V24.1 Principal Audit — V11 Ball/Player/Score State Stabilization

## 1. Scope
V11 is a focused continuation of the existing V24.1/V10 architecture. It does not introduce a new entrypoint, does not rewrite the system, and does not train or tune PPO. `brain.py` remains the project entrypoint.

RT-DETR remains an asynchronous CPU global re-detector. The fast perception path remains the canonical source for frame-to-frame identity and control gating.

## 2. V10 live evidence that drives V11
The reported V10 Windows live run reached 9.5 observed Hz with zero RT-DETR worker errors. The important remaining gaps were ball validity (~0.75), active validity (~0.81), world validity (~0.43), and control validity (~0.33). RT-DETR inference remained slow on CPU (about 0.71 s mean), so forcing detector freshness every decision tick would be architecturally wrong.

The V10 run also showed an implausible score observation sequence `0:0 -> 2:0`, confirming that score OCR needed a temporal transition validator rather than simple two-frame confirmation of an arbitrary candidate.

## 3. V11 changes

### 3.1 Ball state machine
`BallManager` now tracks whether a real measurement has established a confirmed temporal streak. Prediction is allowed only after at least two consecutive real measurements, and only for a short bridge of at most 0.20 s. A single isolated CV/detector observation cannot immediately authorize `PREDICTED` state.

Prediction remains separate from measurement: `PREDICTED` is never treated as a trusted control ball.

### 3.2 Persistent player identity / recovery suppression
The persistent `PlayerTracker` remains the identity authority. V11 tightens CV recovery before adding detections: recovered components are compared against both current RT-DETR proposals and existing track predictions, and at most two CV-recovery detections are added per frame. This reduces roster inflation from pitch highlights and duplicate recovery blobs while preserving `UNKNOWN` team identity separately from player existence.

### 3.3 Active-player hysteresis
Active association no longer increments the marker-missed counter twice for the same missing marker. Active switching now requires a clearly better candidate for two consecutive marker observations. Short marker loss therefore retains the prior active track instead of immediately changing or dropping identity.

### 3.4 Score temporal validator
After score initialization, only these score transitions are accepted as possible goal updates:
- no change;
- exactly +1 for the left side; or
- exactly +1 for the right side.

Decreases, multi-goal jumps, and simultaneous changes are rejected as OCR outliers and are removed from the pending confirmation state. Score semantics remain canonical through `controlled_side`, currently configured as `left`.

### 3.5 World/control gate
The V10 fail-closed design is retained. V11 does not loosen validity thresholds to make metrics look better. `world_valid` additionally requires a stable/recent player set and a valid possession owner track. Control remains neutral whenever the world is not trustworthy.

## 4. Status classification

### IMPLEMENTED
- Existing modular V24.1 architecture preserved.
- CPU-only RT-DETR contract preserved; no CUDA dependency added.
- Ball prediction requires prior measurement confirmation and is time-bounded.
- Player track identity persists independently from detector freshness.
- CV recovery is bounded and gated against existing tracks.
- Active marker loss/switching uses explicit hysteresis.
- Score OCR has a post-initialization one-goal temporal sanity filter.
- Control gate remains fail-closed.
- Y / goalkeeper rush remains outside the PPO/control action space.

### HEURISTIC
- Ball CV/footpoint evidence.
- Optical-flow track propagation.
- Active marker geometry and hysteresis.
- Possession estimation.
- Team appearance smoothing.
- Score OCR preprocessing and transition validation.

### ML INFERENCE
- Existing RT-DETR R50vd detector when local model assets are present.

### NOT TRAINED
- PPO / RL policy.
- No policy rollout is authorized by this V11 package.

### PARTIAL
- True Windows live V11 metrics were not generated inside this build environment. A fresh live acceptance run remains required.
- The bundled offline frame set can exercise deterministic CV, active, score, tracking, and safety regressions, but it cannot substitute for the Windows live RT-DETR run when the model assets are not present in the archive.

## 5. Regression tests

`pytest -q` result for the V11 source tree:

`57 passed`

The test suite includes regressions for:
- ball prediction confirmation gating;
- ball reset state;
- score multi-goal OCR jump rejection;
- score pending-state clearing;
- active-switch hysteresis;
- tracker flow continuity;
- player-foot possession;
- CPU-only RT-DETR contract;
- score and ball behavior on the supplied real-frame fixtures.

## 6. Acceptance target for the next Windows live run

The next observation-only run should target:

- observed_hz >= 9;
- ball_valid_rate >= 0.85;
- active_valid_rate >= 0.90;
- players_valid_rate >= 0.90;
- world_valid_rate >= 0.75;
- control_valid_rate >= 0.70;
- RT-DETR model_errors = 0;
- score false-jump count = 0.

Do not train or tune PPO until the live run clears the perception/world/control gates consistently.

## 7. Launch

```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check_v11
```

This is observation-only acceptance. The regular runtime entrypoint remains `python brain.py` after the perception acceptance stage.
