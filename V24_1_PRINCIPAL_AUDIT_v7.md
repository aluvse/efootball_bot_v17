# V24.1 Principal Audit — v7

## 1. Executive summary
V6 runtime testing exposed a deterministic RT-DETR worker defect: post-processing referenced an undefined `rgb` variable. The worker therefore failed before publishing any detector result. This explains the observed `model_completed_samples=0`, `model_fresh_rate=0`, repeated `inference error: NameError`, and the resulting absence of detector balls/players in the live checker.

V7 fixes that concrete defect while preserving the existing modular V24.1 architecture and CPU/AMD contract. The worker remains asynchronous and latest-frame/bounded-queue based. No CUDA/NVIDIA dependency is added and PPO is unchanged.

## 2. Confirmed V6 defect
Observed on Windows live runtime:
- RT-DETR loaded successfully on CPU.
- 49 frames were submitted over 10 seconds.
- 0 inference results completed.
- Repeated `NameError: name 'rgb' is not defined`.

Root cause: `models.py` built `rgb_buffer` but later computed target size from the non-existent `rgb` symbol.

## 3. V7 changes
- Fixed RT-DETR post-processing target size to use `rgb_buffer.shape[:2]`.
- Preserved reusable RGB buffer.
- Preserved asynchronous worker and single pending frame semantics.
- Worker now returns its frame buffer to the pool after transient inference errors.
- Added detector `errors` and `last_error` runtime telemetry.
- Updated `verify_install.py` to apply/verify the CPU thread contract (`8` intra-op, `1` inter-op) in a fresh verification process.
- Added regression test preventing the old undefined symbol from returning.
- Live checker summary now records model error rate.

## 4. CPU/AMD contract
Target hardware: Ryzen 7 5700X3D + Radeon RX 6600 + 32 GB RAM.

V7 deliberately remains CPU-only. CUDA/NVIDIA requirements are excluded. The RX 6600 is not used by the inference path.

## 5. Validation
Local test suite: **48 passed**.

The Windows live test from V6 is intentionally treated as a failure/regression fixture, not as an acceptance result. The next required validation is a fresh 10-second run on Windows after installing V7.

## 6. Required Windows test
```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check_v7
```

Inspect:
- `model_runtime.submitted`
- `model_runtime.completed`
- `model_runtime.errors`
- `model_fresh_rate`
- `model_age_mean_s`
- `model_age_max_s`
- `ball_valid_rate`
- `ball_detector_rate`
- `active_valid_rate`
- `world_valid_rate`
- `control_valid_rate`

Acceptance is not declared until the live run publishes real detector results with bounded source-frame age.

## 7. PPO status
**NOT TRAINED.** No PPO tuning or retraining has been performed in V7.
