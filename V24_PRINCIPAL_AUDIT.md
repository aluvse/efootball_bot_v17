# V24 Principal Engineering Audit

## Architecture

`SCREEN -> window capture -> player/ball/active perception -> tracking -> canonical FootballState -> possession/prediction -> tactical planner -> structured PPO -> safety -> XInput`

PPO never receives raw pixel detections directly. It receives a fixed semantic vector generated from the canonical world state.

## IMPLEMENTED

- Windows-only eFootball client-area capture; no silent full-screen fallback.
- Centralized UI exclusion with calibrated bottom HUD/minimap regions.
- Player footpoint/pitch validation.
- Separation of player existence from team identity; UNKNOWN players survive.
- Appearance-only team classifier that does not use field x/y to decide team.
- Persistent tracker with TENTATIVE/CONFIRMED/LOST/DEAD lifecycle.
- Deterministic Hungarian matching and stable track IDs.
- Cyan active-player marker detection using color + geometry and top-center association.
- Confidence-aware ball fusion: RT-DETR is preferred over weak CV candidates; temporal prediction is explicitly marked predicted/invalid for control.
- HUD score OCR with temporal confirmation.
- Ball trajectory goal-line fallback when score HUD is unavailable.
- Separate possession estimation with hysteresis; active player is not treated as ball owner.
- 0.25 / 0.50 / 1.00 second prediction layer.
- Canonical attacking coordinate system.
- Explicit `world_valid`, `control_valid`, `ball_valid`, `active_valid`, `possession_valid` flags.
- Control gate that refuses random/attack input when perception is not trustworthy.
- Structured categorical PPO: movement x football action x sprint.
- Planner bootstrap / behavior cloning before on-policy PPO.
- Persistent PPO checkpoint with bootstrap state.
- Y / goalkeeper rush is not in the action space and is never emitted.
- Action buttons are pulses rather than permanent holds.
- One-frame perception probe and regression tests.

## PARTIAL / HEURISTIC

- Team identity is an appearance heuristic with temporal smoothing, not a learned jersey classifier trained on an eFootball dataset.
- Active marker is CV heuristic + geometry; it needs live calibration on multiple resolutions/camera states.
- Possession is screen-geometry based.
- Shot-on-target is a conservative heuristic.
- Goal trajectory fallback is heuristic; HUD score change is the preferred screen-grounded signal.
- `game_mode` is currently inferred only coarsely (`PLAY` vs `UNKNOWN`).
- xT/space/danger are compact heuristics, not learned football models.
- Tactical planner is a bootstrap policy, not a learned expert.

## NOT TRAINED

The PPO policy is not claimed to be trained to strong eFootball performance. A new V24 policy starts from a heuristic bootstrap and then requires real-game experience. Google Research Football reports that football policies can require very large environment-step budgets; its benchmark results are evidence that data volume matters, not a promise about this screen-based agent. 

## Real-frame probe status

The archive contains 24 unlabeled eFootball screenshots. It does not contain the specific frames 300/1200/1500 requested by the historical audit, so exact before/after numbers for those frame IDs are not reported.

Offline probes were run on `frame_00.jpg` and `frame_14.jpg` without local RT-DETR weights. The system correctly keeps `control_valid=False` in that configuration. Cyan marker segmentation found one dominant marker candidate in each tested frame, but association cannot be validated on those frames without the primary player detector/model output. This is reported as PARTIAL rather than a successful acceptance claim.

## Tests

`pytest -q` -> all 8 tests pass in the build environment.

## Intentional reuse from V23

Only infrastructure concepts that were already useful were retained: local RT-DETR adapter, DXCam/window discovery pattern, XInput/vgamepad, existing eFootball screenshot dataset, and calibration starting points. Perception/world/PPO contracts were re-authored for V24.
