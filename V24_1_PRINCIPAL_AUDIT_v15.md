# V24.1 Principal Audit — V15

## Basis
V15 is an architectural step on top of the verified V14 project. The original screen-based eFootball architecture is preserved: capture -> RT-DETR/CV perception -> tracking -> world -> planner/PPO -> safety -> XInput. `brain.py` remains the entry point; no `main.py` was added.

## IMPLEMENTED
- Bounded 4-sample `WorldHistory` with ball delta, active streak, possession-owner streak and continuity checks.
- Explicit separation of `active_player`, `designated_player` and possession owner.
- `WorldState` no longer requires an active cyan marker or a named possession owner on every frame to be considered coherent.
- Coarse `game_mode` field with conservative heuristic classification.
- Sticky high-level control intent is represented explicitly in the existing XInput boundary; action buttons remain bounded taps.
- Added `WorldTensorEncoder`: 8 spatial channels at 48x32, derived only from canonical `WorldState`.
- Live perception report now records designated-player and temporal metrics.
- PPO encoder contract updated from 153 to 170 dimensions; PPO action-space structure remains unchanged.

## HEURISTIC
- Designated-player selection is foot-proximity / relative-motion based.
- Game-mode classification is coarse and heuristic; no dedicated visual set-piece classifier is claimed.
- Temporal continuity and world/control thresholds remain engineered heuristics.
- WorldTensor rasterization is a representation layer, not a learned model.

## ML INFERENCE
- RT-DETR remains CPU-only and asynchronous as the global player re-detector.
- No new learned inference model was introduced.

## NOT TRAINED
- No PPO training run was performed as part of V15.
- No model weights were trained or tuned by the V15 build process.

## PARTIAL
- Sticky control state has been integrated into the existing XInput executor but has not been certified by an actual gameplay-control run in this build.
- `game_mode` is currently coarse (`PLAY/RESTART/UNKNOWN` in the present heuristic path); exact eFootball set-piece classes are not yet visually recognized.
- WorldTensor is prepared for a future spatial policy encoder but is not yet consumed directly by PPO.
- Existing V14 Windows live behavior must be re-measured with the V15 build.

## Regression verification
- Local regression suite: **66 passed**.
- Python bytecode compilation: passed.
- The CPU-only dependency/model contract is preserved.

## Next Windows acceptance run
Run the existing observation-only test for at least 35 seconds:

```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 35 --hz 10 --out live_perception_check_v15
```

Primary metrics: `world_valid_rate`, `control_valid_rate`, `designated_valid_rate`, `temporal_valid_rate`.

Success is not declared by this source build alone. The V15 live result must be judged from the user's actual Windows/eFootball run.
