# V24.1 Principal Audit — perception/runtime repair

## 1. Executive summary
The existing V24 modular architecture was preserved. The current repair concentrates on perception, world-state integrity and safe control gating. PPO was not trained or tuned.

## 2. Current architecture
`capture -> RT-DETR -> player validation -> team -> tracking -> active -> ball fusion -> possession/world -> control gate -> planner/PPO`. Existing modules remain in place; no `main.py` was added.

## 3. Confirmed defects
- RT-DETR environment previously failed because torchvision was unavailable. The Windows install contract is now torch 2.5.1 + torchvision 0.20.1 + transformers 4.57.6 and the local model loader works.
- Tesseract is now found on the tested Windows host.
- Ball CV previously accepted UI/field-like color candidates. The ball path now requires field/UI/shape/size checks and has a player-foot correlation path. Detector evidence remains authoritative when plausible; weak isolated detector candidates can be rejected in favor of strong foot-correlated CV evidence.
- RT-DETR ball proposals now preserve their bbox so geometric validation can actually be applied.
- Active marker association uses the cyan bar and track association rather than only center proximity.
- UNKNOWN players are retained.
- Tracker lifecycle is explicit and deterministic.
- Offline score probing now performs a second confirmation sample so a one-frame probe does not always report the previous stable score.

## 4. Root causes
The most important ball failure was semantic: a small white field marking could satisfy color/compactness tests while the real ball near the active player's feet was not considered as a dedicated local hypothesis. The detector adapter also discarded ball bbox geometry.

The score probe failure was partly a lifecycle issue: `confirm_frames=2` correctly prevents single-frame score commitment, but a one-frame offline probe consequently exposed the initial `(0,0)` stable state even when an OCR candidate had been observed.

Player recall was limited by an unnecessarily high RT-DETR threshold and by recovery being disabled whenever enough detector players were already present. The threshold was lowered and recovery/merge remain bounded by player geometry.

## 5. Changed files
- `efootball_agent/config.py`
- `efootball_agent/perception/models.py`
- `efootball_agent/perception/player.py`
- `efootball_agent/perception/ball.py`
- `efootball_agent/perception/score.py`
- `efootball_agent/perception/pipeline.py`
- `efootball_agent/tools/perception_probe.py`
- `efootball_agent/tests/test_core.py`

## 6. P0 fixes
### Ball
Implemented detector/CV/Predicted/Unknown source separation, bbox validation, field/UI checks, temporal consistency, and player-foot correlation. The supplied frame shows the gameplay ball immediately beside the active player's foot; the regression logic explicitly prevents the isolated field marking at about x=592,y=244 from winning over the foot-correlated ball around x=657,y=262.

### Score
Implemented combined two-digit OCR over the calibrated score cells, with per-digit fallback. Static probes perform a second synthetic confirmation call. The live runtime still retains temporal confirmation and never fabricates a score.

### Players
RT-DETR threshold is now 0.18. Player validation adds a human-bbox aspect sanity check and CV recovery is available as a bounded supplemental source. Recovery candidates cannot erase detector players and UNKNOWN identities remain valid tracks.

## 7. Tests
`pytest -q` -> **38 passed**.

## 8. Real-frame probe results
On the supplied latest frame (`frame_0012`):
- RT-DETR loaded and produced 22 raw person proposals in the successful fresh run.
- 17 passed pitch validation in that run.
- Active marker: valid, confidence 0.7813.
- Ball in the successful fresh run: detector selected `(0.4629,0.3390)` with 0.442 detector confidence and CV corroboration 0.8324; this is the known field-marker failure mode now explicitly guarded by the new player-foot path.
- World/control were valid in that earlier run only because the old detector result was still allowed after corroboration; the new repair does not treat isolated field-marker detector proposals as sufficient when a stronger foot-correlated hypothesis exists.
- Tesseract is installed and `verify_install.py` passes.

Important visual note: the attached screenshot itself visibly shows `0 0` in the HUD. The reported external truth of `0:3` therefore cannot be independently verified from that still image alone. The OCR path now reports raw observed digits as separate fields and supports multi-digit values such as `0:3`; a live frame whose HUD visibly contains `0:3` is the correct acceptance fixture.

## 9. BEFORE vs AFTER
| Area | BEFORE | AFTER |
|---|---|---|
| RT-DETR | import/async/stale issues | local loader works, stamped freshness |
| Players | raw_person sometimes 0 | fresh run reached 22 raw proposals |
| Ball | false field/UI candidate | foot-correlated recovery + detector bbox validation |
| Score | single-frame probe could show prior 0:0 | combined OCR + probe confirmation + observed candidate |
| Active | low association on bad frames | cyan bar + track association/hysteresis |
| Safety | correct fail-closed | retained |
| PPO | not ready | unchanged/not trained |

## 10. Remaining limitations
- RT-DETR is still a generic detector and can miss heavily occluded/small players.
- Team classification remains heuristic appearance analysis.
- Ball-at-feet recovery is heuristic CV, not a trained ball detector.
- Score recognition requires the current HUD layout; multi-layout matches still need additional fixtures.
- `world_valid/control_valid` should be evaluated over continuous live sequences, not a single still frame.

## 11. Exact launch commands
```powershell
python tools\verify_install.py
python tools\live_perception_snapshot.py --seconds 8
python tools\perception_probe.py live_snapshots\frame_0000.jpg --out probe_live_0000
python tools\perception_probe.py live_snapshots\frame_0008.jpg --out probe_live_0008
python tools\perception_probe.py live_snapshots\frame_0012.jpg --out probe_live_0012
pytest -q
python brain.py
```

## Status labels
- IMPLEMENTED: runtime contracts, detector loading, confidence-aware ball fusion, score OCR pipeline, active association, deterministic tracking, safety gating.
- HEURISTIC: CV ball recovery, team classification, possession, goal trajectory fallback.
- ML INFERENCE: local RT-DETR only.
- NOT TRAINED: PPO.
- PARTIAL: full-match continuous validation and broad score/HUD fixture coverage.


## Follow-up evidence: frame_0007 / frame_0014

- Verified current Windows install contract: RT-DETR assets present; Tesseract executable present.
- `frame_0014`: RT-DETR can produce 21 raw persons / 19 pitch-validated; active marker valid; ball detector can be valid.
- `frame_0007`: a compact white field ball candidate is visible around normalized `(0.603, 0.518)`; the prior probe result `(0.0898, 0.5347)` was a stale prediction, not the current frame's ball.
- Probe now confirms score on the frozen frame without rerunning the full pipeline, preventing artificial ball prediction from the score confirmation pass.
- Current live contract uses `controlled_side=left`, so HUD `0:3` maps canonically to `score_for=0`, `score_against=3`.
- PPO training remains gated until a multi-frame live stability check confirms that detector-source ball and model freshness are stable in the actual runtime.


## Current correction pass (V24.1 fixed v3)

### IMPLEMENTED
- Current live team side is encoded explicitly as `controlled_side=left`; HUD `0:3` now maps to canonical `score_for=0`, `score_against=3`.
- Frozen-frame probe confirms score without rerunning ball/tracking, avoiding artificial prediction introduced by the former second `engine.step()`.
- Strong first-frame white-ball CV acquisition is allowed when compactness, contrast, size and field checks all agree. On the supplied frame_0007 it selects approximately `(0.603, 0.518)`, matching the visible ball.
- Prediction is bounded by age and miss count and no longer recursively overwrites the last real measurement.
- Probe wait budget is 8 seconds so CPU RT-DETR warm-up is not misclassified as an immediate stale result.

### HEURISTIC
- CV ball recovery remains heuristic and must not outrank a validated detector proposal merely because its raw CV score is high.
- Team identity and possession remain temporal heuristics.

### ML INFERENCE
- RT-DETR remains the player/ball proposal model.

### NOT TRAINED
- PPO remains unchanged/not trained in this pass.

### PARTIAL
- Full 22-player recall is not guaranteed on every camera frame; provided evidence shows 19 validated players from 21 raw proposals on one good frame.
- A multi-frame live stability run is still required before enabling sustained PPO control.

Observation-only live stability tool: tools/live_perception_check.py (no XInput).

## Realtime scheduler repair (V24.1-v5)

The RT-DETR worker now uses latest-frame semantics with explicit source-frame
metadata: source timestamp, frame id, submit time, completion time, inference
latency, and age at completion. The main perception loop submits frames and
immediately continues; it never waits for detector inference. If a newer frame
arrives while inference is running, only the newest pending frame is retained.

`WorldState.debug` now exposes `model_age_s`, `model_source_frame_id`,
`model_inference_latency_s`, `model_age_at_completion_s`, `model_device`, and
worker counters. `control_valid` continues to fail closed when the detector
measurement is older than `max_model_stale_s` when model-backed control is
required.

The project is configured for CPU-only inference on the AMD/Windows target.
