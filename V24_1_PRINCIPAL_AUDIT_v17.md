# V24.1 Principal Audit — V17

## Release basis

V17 is an architecture-preserving incremental build on V16. The existing chain
remains: `brain.py -> capture -> RT-DETR/CV perception -> tracking -> world ->
planner/PPO -> safety -> XInput`. No `main.py` was added. Windows capture remains
client-area `eFootball™` at 1280x720 with `window_only=True`; the AMD/Windows
contract remains CPU-only.

## Why V17 exists

The V16 live run showed a strong detector/tracker foundation but lower world/control
validity because the remaining bottleneck was temporal coherence between the ball,
designated player and possession. V17 therefore changes that layer instead of
continuing to retune active-marker thresholds.

## IMPLEMENTED

- Added a stateful designated-player selector with bounded hysteresis. The selector
  combines ball distance, previous-designated continuity, team continuity and a
  small velocity-to-ball alignment term.
- Kept designated-player semantics separate from the cyan active-player marker and
  possession owner.
- Added bounded possession memory through short UNKNOWN/weak-ball gaps. Prior OUR
  or OPP ownership decays into a receiving/control state instead of being erased
  immediately. A competing team needs a configured score margin to take ownership.
- Short predicted-ball states are now explicitly marked `valid` when their decayed
  confidence is sufficient for the existing bounded WorldState prediction bridge.
  This fixes the previous contract mismatch where the bridge required `ball.valid`
  while the prediction object was always emitted with `valid=False`.
- Reset designated/possession temporal memory together with the perception engine.
- Kept active-control memory from V16 unchanged.
- Kept the existing semantic action space: movement + football action + sprint. Y /
  goalkeeper rush remains outside PPO action space and is never emitted by the
  control mapping.
- Kept 4-frame WorldHistory and 8x32x48 WorldTensor infrastructure.

## HEURISTIC

- Designated-player hysteresis and velocity alignment are engineered heuristics.
- Possession gap retention, confidence decay and switch margin are engineered
  heuristics.
- No claim of learned player ownership or ball association is made.

## ML INFERENCE

- RT-DETR remains the asynchronous global proposal source and is still CPU-only.
- No new model or weight set was introduced in V17.

## NOT TRAINED

- PPO was not trained or hyperparameter-tuned in this release.
- No model weights were trained in V17.

## PARTIAL / OPEN

- A new 35-second Windows live acceptance run is still required to measure whether
  world/control validity improves over the V16 baseline.
- Score OCR and coarse game-mode classification remain heuristic/partial.
- The packaged project references the local RT-DETR model directory but does not
  embed the model weights; installation must provide those assets separately.

## Local verification performed

- `python -m py_compile ...` across the project: PASS.
- `pytest -q`: PASS, 71 tests.

The environment used for this build container is not the target Windows runtime,
so `tools/verify_install.py` is not treated as a release acceptance result here.
A Windows run should still execute it before live testing.

## V17 Windows acceptance command

```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 35 --hz 10 --out live_perception_check_v17
```

Primary acceptance metrics:
`world_valid_rate`, `control_valid_rate`, `active_valid_rate`,
`designated_valid_rate`, `possession_confidence`, `temporal_valid_rate`,
`ball_predicted_rate`, `score_invalid_transition_count`, model age and detector
inference latency.

## V16 baseline carried into V17

V16 reported approximately 0.9188 model-recent, 0.7727 ball-valid, 0.7890
active-valid, 0.9773 players-valid, 0.3799 world-valid and 0.3669 control-valid
over the supplied 35-second run. V17 is intended to target the ball/designated/
possession bottleneck behind those lower world/control rates.
