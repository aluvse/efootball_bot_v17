from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time

import cv2
import numpy as np

from .schema import PlayerDetection


@dataclass(frozen=True)
class ModelBall:
    pos: tuple[float, float]
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ModelResult:
    persons: tuple[PlayerDetection, ...] = ()
    balls: tuple[ModelBall, ...] = ()
    # `stamp` is kept as completion time for backward compatibility.
    stamp: float = 0.0
    source_timestamp: float = 0.0
    frame_id: int = -1
    submitted_at: float = 0.0
    completed_at: float = 0.0
    inference_latency_s: float = 0.0
    age_at_completion_s: float = 0.0


class RTDETRAdapter:
    """Asynchronous RT-DETR worker with latest-frame semantics.

    The main perception loop never waits for inference. The worker always
    processes the newest pending frame and publishes the completed result
    together with the source-frame timestamp and measured inference latency.
    """

    def __init__(self, settings):
        self.settings = settings
        self.device = "cpu"
        self.detector = None
        self.processor = None
        self.latest = ModelResult()
        self.pending = None
        self._free_buffer = None
        self.last_submit = 0.0
        self.submit_count = 0
        self.complete_count = 0
        self.drop_count = 0
        self.error_count = 0
        self.last_error = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.load_error = None
        self._loading = False
        self._load()
        if self.detector is not None:
            self.thread = threading.Thread(
                target=self._worker,
                name="rtdetr",
                daemon=True,
            )
            self.thread.start()

    def _load(self):
        path = Path(self.settings.rtdetr_dir)
        weights = sorted(list(path.glob("*.safetensors")) + list(path.glob("*.bin")))
        processor = any((path / name).exists() for name in (
            "preprocessor_config.json", "image_processor_config.json", "processor_config.json"
        ))
        missing = []
        if not (path / "config.json").exists():
            missing.append("config.json")
        if not weights:
            missing.append("model.safetensors|pytorch_model.bin")
        if not processor:
            missing.append("preprocessor_config.json|image_processor_config.json|processor_config.json")
        if missing:
            print(
                f"[MODEL] RT-DETR local assets incomplete at {path}: "
                + ", ".join(missing)
                + ". Model proposals are disabled."
            )
            return
        try:
            import torch
            import torchvision  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            requested = str(getattr(self.settings.player, "inference_device", "cpu")).lower()
            if requested not in {"cpu", ""}:
                raise RuntimeError(
                    "This V24.1 Windows/AMD build is CPU-only. "
                    f"inference_device={requested!r} is not supported."
                )
            self.device = "cpu"

            # Tune the 5700X3D for a dedicated detector worker. Keep these
            # settings process-local and avoid touching CPU affinity globally.
            cpu_threads = max(1, int(getattr(self.settings.player, "cpu_num_threads", 8)))
            cpu_interop = max(1, int(getattr(self.settings.player, "cpu_num_interop_threads", 1)))
            os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
            try:
                torch.set_num_threads(cpu_threads)
            except RuntimeError:
                pass
            try:
                torch.set_num_interop_threads(cpu_interop)
            except RuntimeError:
                pass

            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    str(path),
                    local_files_only=True,
                    backend="torchvision",
                    use_fast=True,
                )
            except TypeError:
                self.processor = AutoImageProcessor.from_pretrained(
                    str(path),
                    local_files_only=True,
                    backend="torchvision",
                )

            self.detector = AutoModelForObjectDetection.from_pretrained(
                str(path),
                local_files_only=True,
            ).eval().to(self.device)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            print(f"[MODEL] RT-DETR loaded on {self.device}.")
        except Exception as exc:
            self.load_error = exc
            print(f"[MODEL] RT-DETR load failed: {type(exc).__name__}: {exc}")
            self.detector = None

    def submit(self, frame, timestamp=None, frame_id=-1):
        if self.detector is None:
            return False
        now = time.monotonic() if timestamp is None else float(timestamp)
        hz = max(0.5, float(self.settings.player.model_hz))
        if now - self.last_submit < 1.0 / hz:
            return False
        with self.lock:
            if self.pending is not None:
                self.drop_count += 1
                pending_frame = self.pending[0]
            else:
                if (
                    self._free_buffer is None
                    or self._free_buffer.shape != frame.shape
                    or self._free_buffer.dtype != frame.dtype
                ):
                    self._free_buffer = np.empty_like(frame)
                pending_frame = self._free_buffer
                self._free_buffer = None
            np.copyto(pending_frame, frame)
            self.pending = (pending_frame, now, int(frame_id))
            self.last_submit = now
            self.submit_count += 1
        return True

    def read(self):
        with self.lock:
            return self.latest

    def runtime_metrics(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            latest = self.latest
            return {
                "device": self.device,
                "submitted": int(self.submit_count),
                "completed": int(self.complete_count),
                "dropped_pending": int(self.drop_count),
                "errors": int(self.error_count),
                "last_error": self.last_error,
                "pending": self.pending is not None,
                "pending_age_s": (now - self.pending[1]) if self.pending is not None else None,
                "source_frame_id": int(latest.frame_id),
                "model_age_s": (now - latest.source_timestamp) if latest.source_timestamp > 0 else None,
                "inference_latency_s": float(latest.inference_latency_s),
                "age_at_completion_s": float(latest.age_at_completion_s),
            }

    @staticmethod
    def _apply_worker_affinity(spec: str, cpu_threads: int):
        """Best-effort Windows thread affinity; never fails model startup."""
        if os.name != "nt":
            return None
        try:
            spec = str(spec or "auto").strip().lower()
            logical = os.cpu_count() or cpu_threads or 1
            if spec == "auto":
                count = min(max(1, cpu_threads), logical)
                start = max(0, logical - count)
                cores = list(range(start, logical))
            else:
                cores = []
                for part in spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        a, b = [int(x.strip()) for x in part.split("-", 1)]
                        cores.extend(range(min(a,b), max(a,b)+1))
                    else:
                        cores.append(int(part))
                cores = [c for c in sorted(set(cores)) if 0 <= c < logical]
            if not cores:
                return None
            import ctypes
            kernel32 = ctypes.windll.kernel32
            thread = kernel32.GetCurrentThread()
            mask = ctypes.c_size_t(0)
            for c in cores:
                mask.value |= (1 << c)
            prev = kernel32.SetThreadAffinityMask(thread, mask)
            return {"cores": cores, "previous_mask": int(prev)}
        except Exception:
            return None

    def _worker(self):
        import torch

        cpu_threads = max(1, int(getattr(self.settings.player, "cpu_num_threads", 8)))
        affinity = self._apply_worker_affinity(
            getattr(self.settings.player, "cpu_affinity_cores", "auto"),
            cpu_threads,
        )
        rgb_buffer = None

        while not self.stop_event.is_set():
            with self.lock:
                item = self.pending
                self.pending = None
            if item is None:
                time.sleep(0.001)
                continue

            frame, source_timestamp, frame_id = item
            started = time.monotonic()
            try:
                if rgb_buffer is None or rgb_buffer.shape != frame.shape:
                    rgb_buffer = np.empty_like(frame)
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_buffer)
                model_size = int(getattr(self.settings.player, "model_input_size", 640))
                try:
                    inputs = self.processor(
                        images=rgb_buffer,
                        return_tensors="pt",
                        size={"height": model_size, "width": model_size},
                    )
                except (TypeError, ValueError):
                    inputs = self.processor(images=rgb_buffer, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = self.detector(**inputs)
                sizes = torch.tensor([rgb_buffer.shape[:2]], device=self.device)
                result = self.processor.post_process_object_detection(
                    outputs,
                    threshold=float(self.settings.player.min_model_conf),
                    target_sizes=sizes,
                )[0]

                persons = []
                balls = []
                h, w = frame.shape[:2]
                for box, score, label in zip(
                    result["boxes"],
                    result["scores"],
                    result["labels"],
                ):
                    b = box.detach().cpu().numpy()
                    conf = float(score)
                    label_id = int(label)
                    bbox = (
                        float(b[0] / w),
                        float(b[1] / h),
                        float(b[2] / w),
                        float(b[3] / h),
                    )
                    center = (
                        float((bbox[0] + bbox[2]) * 0.5),
                        float((bbox[1] + bbox[3]) * 0.5),
                    )
                    if label_id == self.settings.player.person_label:
                        persons.append(
                            PlayerDetection(
                                center=center,
                                bbox=bbox,
                                source="RTDETR",
                                confidence=conf,
                            )
                        )
                    elif label_id == self.settings.ball.ball_label:
                        balls.append(
                            ModelBall(
                                pos=center,
                                confidence=conf,
                                bbox=bbox,
                            )
                        )

                completed = time.monotonic()
                model_result = ModelResult(
                    persons=tuple(persons),
                    balls=tuple(balls),
                    stamp=completed,
                    source_timestamp=source_timestamp,
                    frame_id=frame_id,
                    submitted_at=source_timestamp,
                    completed_at=completed,
                    inference_latency_s=completed - started,
                    age_at_completion_s=max(0.0, completed - source_timestamp),
                )
                with self.lock:
                    self.latest = model_result
                    self.complete_count += 1
                    if self._free_buffer is None:
                        self._free_buffer = frame
            except Exception as exc:
                with self.lock:
                    self.error_count += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"[MODEL] inference error: {type(exc).__name__}: {exc}")
                # Return the frame buffer to the pool so a transient inference
                # error cannot starve the bounded latest-frame scheduler.
                with self.lock:
                    if self._free_buffer is None:
                        self._free_buffer = frame
                time.sleep(0.01)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

