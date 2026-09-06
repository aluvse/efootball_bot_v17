# V24.1 Principal Audit — V16

## Basis
V16 is an incremental architecture-preserving step on V15. The original screen-based eFootball architecture remains intact: capture -> RT-DETR/CV perception -> tracking -> world -> planner/PPO -> safety -> XInput. `brain.py` remains the entry point; no `main.py` was added.

## Live evidence driving V16
The supplied V15 Windows run reported: observed_hz 9.0857, players_valid 0.9843, active_valid 0.7358, designated_valid 0.9088, temporal_valid 0.6981, world_valid 0.5157, control_valid 0.4277, model errors 0. The log also showed multiple frames where `world=True` while `control=False` because direct active-marker evidence temporarily disappeared, and short `PREDICTED` ball periods forced `world=False`.

## IMPLEMENTED
- Added bounded control-active memory separate from raw `active_valid`.
- Added `control_player`, `control_track_id`, `control_track_confidence`, and `control_active_memory` to `WorldState`.
- Control may continue through a short active-marker dropout when the remembered controlled track is still present, bounded by `control_active_grace_s`.
- Added short predicted-ball bridge for WorldState/control only when prediction is recent, sufficiently confident and temporal continuity is valid. UNKNOWN remains fail-closed.
- Preserved direct `active_valid` semantics for perception metrics.
- Preserved ball state machine, score validator, player tracker, designated-player semantics, game mode, sticky control and WorldTensor infrastructure.
- PPO is still not enabled for this validation task and no training was performed.

## HEURISTIC
- Active control memory grace is engineered, not learned.
- Predicted-ball bridge is engineered and intentionally bounded.

## ML INFERENCE
- RT-DETR remains CPU-only and asynchronous; no new model was introduced.

## NOT TRAINED
- No PPO training run.
- No model weight tuning.

## PARTIAL
- V16 still needs a fresh Windows live measurement.
- Exact set-piece/game-mode classification remains coarse.

## Regression verification
- Run the existing `pytest` suite and `compileall` before release.

## Next Windows acceptance run
```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 35 --hz 10 --out live_perception_check_v16
```
Primary metrics: `world_valid_rate`, `control_valid_rate`, `active_valid_rate`, `designated_valid_rate`, `temporal_valid_rate`, `ball_predicted_rate`, `score_invalid_transition_count`.
