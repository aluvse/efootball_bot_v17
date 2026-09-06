# V24.1 Principal Audit — CPU realtime scheduler repair

## 1. Executive summary
The existing modular V24 architecture is preserved. This pass removes the CUDA-specific installation path and optimizes the realtime perception scheduler for the supplied AMD Ryzen 7 5700X3D + RX 6600 Windows host. RT-DETR remains ML inference and runs in a dedicated CPU worker; fast perception remains in the main loop. PPO is not trained.

## 2. CPU runtime contract
- Windows / Python 3.11
- torch 2.5.1 + torchvision 0.20.1
- CPU-only RT-DETR for the AMD host; no CUDA dependency
- dedicated RT-DETR worker with latest-frame semantics
- bounded pending queue of one
- source timestamp + completion timestamp + inference latency + source age
- configurable CPU thread count, inter-op thread count and best-effort Windows worker affinity
- fixed 640x640 detector input by default with graceful processor fallback

## 3. Scheduler changes
The capture/perception loop never blocks on RT-DETR. The detector receives only the newest pending frame. Pending model frames use preallocated numpy storage to avoid per-frame allocation. RGB conversion reuses a worker buffer. The live checker no longer writes JPEGs by default.

## 4. Score OCR scheduling
Tesseract OCR is moved to its own latest-frame worker in the live pipeline so the 0.8–1.0s OCR subprocess latency cannot stall the perception loop. The synchronous API remains available to tests and frozen-frame probes. Score source timestamp and OCR age are retained.

## 5. Freshness semantics
`model_age_s = current_perception_timestamp - source_frame_timestamp`. Completion time is never used as a substitute for source age. `max_model_stale_s` bounds detector reuse for perception; `control_model_stale_s` is stricter for control gating.

## 6. No CUDA installation contract
`requirements-win-cuda124.txt` is removed. `inference_device` is CPU-only. `verify_install.py` reports CPU thread settings instead of treating CUDA availability as a required runtime condition.

## 7. Tests
The existing test suite remains the contract. New/updated tests should cover CPU device rejection for non-CPU requests, bounded pending semantics, source-age metadata, score worker non-blocking behaviour, and live checker no-JPEG default.

## 8. Status
IMPLEMENTED: CPU-only model contract; async latest-frame RT-DETR; preallocated frame buffers; worker affinity best effort; torch thread tuning; async score OCR; exact source-frame ages; bounded queue.
HEURISTIC: ball CV recovery, team classification, possession.
ML INFERENCE: RT-DETR only.
NOT TRAINED: PPO.
PARTIAL: actual Windows 5700X3D runtime benchmark must be rerun on the user's machine.

## 9. Exact validation commands
```powershell
python tools\verify_install.py
python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check
pytest -q
```
Do not run `brain.py` during scheduler validation.

## 10. Additional implementation notes
- RT-DETR pending frames use two reusable numpy BGR buffers: one can be processed while the other is pending/replaced.
- RGB conversion reuses a worker-owned numpy buffer.
- Model `submitted_at` now records the source/capture timestamp; `inference_latency_s` is wall-clock worker time; `model_age_s` is source age at perception time.
- Score OCR uses an independent worker in live mode, while synchronous mode remains available to tests and frozen-frame probes.
- PPO runtime device is CPU in the AMD/Windows contract; PPO was not trained.

## V6.1 follow-up
The V6 live test exposed a concrete asynchronous worker defect: the RT-DETR post-processing path referenced an undefined `rgb` variable. That caused every inference job to fail before publishing a result, which explains `model_completed_samples=0` and `model_fresh_rate=0` despite successful model loading. V6.1 replaces that reference with the reusable `rgb_buffer` and records worker error counters in runtime telemetry.
