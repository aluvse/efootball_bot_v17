# V24.1 — Principal Audit V8

## 1. Executive summary
V8 keeps the existing modular V24 architecture and stabilizes realtime scheduling for a CPU-only AMD/Windows system. RT-DETR remains an asynchronous periodic re-detector; it no longer needs to be fresh on every perception tick.

## 2. Current architecture
Capture 30–60 FPS → fast CV/active/ball/tracking → periodic async RT-DETR re-detection → age-aware fusion → canonical WorldState → conservative control gate.

## 3. Confirmed defects addressed
- CPU RT-DETR latency (~0.61 s) made per-frame freshness impossible.
- V7 worker completed inference but the latest result was commonly ~1 s old.
- Fast loop over-relied on fresh detector output.

## 4. Root causes
The CPU detector is a slow global proposal source, not a realtime tracker. Treating every observation as detector-fresh created stale-state churn.

## 5. Changed files
- `efootball_agent/config.py`
- `efootball_agent/perception/pipeline.py`
- `efootball_agent/tools/live_perception_check.py`
- `V24_1_PRINCIPAL_AUDIT_v8.md`

## 6. P0 fixes
- Added `model_reuse_stale_s` bounded reuse window for tracker correction.
- Ball detector proposals still require fresh detector evidence.
- Control no longer requires fresh RT-DETR; fast local perception is authoritative for the control gate.
- Added `model_recent` telemetry distinct from `model_fresh`.
- Live checker now reports scheduler/world/control verdicts and model runtime counts.

## 7. Tests
Existing V24.1 regression suite retained; new scheduler/telemetry tests should be run with `pytest -q`.

## 8. Real-frame probe basis
Known real frames include `0007`, `0008`, `0012`, and `0014`. V7 demonstrated that RT-DETR could complete inference but was too slow to be fresh per 10 Hz tick.

## 9. BEFORE vs AFTER
Before: `model_fresh_rate≈0`, `model_age_mean≈0.97 s`, `ball_detector_rate=0`, `world_valid_rate≈0.01` in V7 live test.
After: V8 separates detector freshness from detector recency and allows fast local perception to continue between global detector passes.

## 10. Remaining limitations
- CPU RT-DETR remains ~1–2 Hz in effective fresh global observations; this is expected hardware/runtime behavior.
- Ball CV/foot association still needs more real sequential validation.
- Team identity and possession remain heuristic.
- Score OCR needs long-run false-positive validation.

## 11. Exact launch commands
```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check_v8
pytest -q
```

### Classification
IMPLEMENTED: scheduler, timestamps, bounded latest-frame queue, stale/recent distinction, control gate separation.
HEURISTIC: CV ball, team identity, possession.
ML INFERENCE: RT-DETR.
NOT TRAINED: PPO.
PARTIAL: long-horizon live ball/possession stability.
