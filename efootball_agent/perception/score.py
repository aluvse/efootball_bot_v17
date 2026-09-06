from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import time
import threading
import numpy as np

import cv2


class ScoreDetector:
    """HUD score/clock detector with exact 1280x720 ROI calibration and temporal confirmation."""

    def __init__(self, cfg, async_mode=False):
        self.cfg = cfg
        self.async_mode = bool(async_mode)
        self.last_ocr = 0.0
        self._lock = threading.Lock()
        self._pending = None
        self._stop = threading.Event()
        self._thread = None
        self._source_timestamp = 0.0
        self._completed_at = 0.0
        self._ocr_latency_s = 0.0
        self.last_score = (0, 0)
        self.pending_score = None
        self.pending_count = 0
        self.initialized = False
        self.confidence = 0.0
        self.clock_s = 0.0
        self._warned = False
        self.tesseract_path = self.find_tesseract()
        self.invalid_transition_count = 0
        self.last_debug = {
            "left": None,
            "right": None,
            "left_confidence": 0.0,
            "right_confidence": 0.0,
            "clock_text": "",
            "clock_s": 0.0,
            "source_timestamp": 0.0,
            "age_s": None,
            "invalid_transition": False,
        }
        if self.async_mode:
            self._thread = threading.Thread(target=self._worker, name="score-ocr", daemon=True)
            self._thread.start()

    def reset(self):
        self.close()
        self.__init__(self.cfg, async_mode=self.async_mode)

    @staticmethod
    def find_tesseract() -> str | None:
        candidates = []
        env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
        if env:
            candidates.append(Path(env))
        which = shutil.which("tesseract")
        if which:
            candidates.append(Path(which))
        candidates.extend([
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        ])
        for path in candidates:
            try:
                if path and path.is_file():
                    return str(path)
            except OSError:
                continue
        return None

    def tesseract_available(self) -> bool:
        return bool(self.tesseract_path)

    @staticmethod
    def crop(frame, roi):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi
        xa = int(max(0, min(w - 1, round(x1 * w))))
        xb = int(max(xa + 1, min(w, round(x2 * w))))
        ya = int(max(0, min(h - 1, round(y1 * h))))
        yb = int(max(ya + 1, min(h, round(y2 * h))))
        return frame[ya:yb, xa:xb]

    def score_pair_crop(self, frame):
        # One combined OCR region is considerably more robust for multi-digit
        # scores such as 0:3 than treating the two glyphs as independent words.
        l = self.cfg.left_score_roi
        r = self.cfg.right_score_roi
        roi = (min(l[0], r[0]), min(l[1], r[1]), max(l[2], r[2]), max(l[3], r[3]))
        return self.crop(frame, roi)

    @staticmethod
    def preprocess_pair(crop):
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark = ((hsv[:, :, 2] < 125) | (gray < 105)).astype('uint8') * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        return cv2.resize(dark, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)

    @classmethod
    def _ocr_score_pair(cls, crop, tesseract_path):
        if crop.size == 0 or not tesseract_path:
            return None, None, 0.0
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        except ImportError:
            return None, None, 0.0
        image = cls.preprocess_pair(crop)
        if image is None:
            return None, None, 0.0
        for psm in (7, 6):
            try:
                text = pytesseract.image_to_string(
                    image, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                ).strip()
            except Exception:
                continue
            digits = re.findall(r"\d", text)
            if len(digits) >= 2:
                return int(digits[0]), int(digits[1]), 0.96
        return None, None, 0.0

    @staticmethod
    def preprocess_digit(crop):
        if crop.size == 0:
            return []
        gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # The supplied HUD is a yellow cell with a dark glyph. Tight crop and
        # one deterministic binary representation are enough; running many
        # Tesseract variants was causing multi-second stalls in the realtime
        # perception loop.
        gray = gray0
        gray = cv2.resize(gray, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = cv2.bitwise_not(binary)
        # Remove the yellow panel border/background; retain the dark glyph as the salient component.
        for img in (binary, inv):
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) >= 20:
                    x, y, w, h = cv2.boundingRect(c)
                    pad = 8
                    xa, ya = max(0, x-pad), max(0, y-pad)
                    xb, yb = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
                    glyph = img[ya:yb, xa:xb]
                    if glyph.size:
                        return [glyph, img]
        return [binary, inv]

    @classmethod
    def _ocr_with_path(cls, crop, tesseract_path):
        if crop.size == 0 or not tesseract_path:
            return None, 0.0
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        except ImportError:
            return None, 0.0
        image = cls.preprocess_digit(crop)
        if not image:
            return None, 0.0
        configs = (
            "--psm 10 -c tessedit_char_whitelist=0123456789",
            "--psm 8 -c tessedit_char_whitelist=0123456789",
            "--psm 6 -c tessedit_char_whitelist=0123456789",
        )
        for variant in image:
            for config in configs:
                text = pytesseract.image_to_string(variant, config=config).strip()
                digits = re.findall(r"\d", text)
                if digits:
                    return int(digits[0]), 0.96
        return None, 0.0

    def ocr_number(self, crop):
        return self._ocr_with_path(crop, self.tesseract_path)

    def _ocr_clock(self, crop):
        if crop.size == 0 or not self.tesseract_path:
            return None, 0.0, ""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        except ImportError:
            return None, 0.0, ""
        image = self.preprocess_digit(crop)
        if not image:
            return None, 0.0, ""
        text = pytesseract.image_to_string(
            image[0],
            config="--psm 7 -c tessedit_char_whitelist=0123456789:",
        ).strip()
        m = re.search(r"(\d{1,2})[:;](\d{2})", text)
        if not m:
            return None, 0.0, ""
        mm, ss = int(m.group(1)), int(m.group(2))
        if not (0 <= mm <= 99 and 0 <= ss <= 59):
            return None, 0.0, ""
        return mm * 60 + ss, 0.95, f"{mm:02d}:{ss:02d}"

    def _empty(self, previous_stable):
        return {
            "left": self.last_score[0],
            "right": self.last_score[1],
            "score_changed": False,
            "previous_score": previous_stable,
            "confidence": self.confidence,
            "clock_s": self.clock_s,
            "initialized": self.initialized,
            "observed_left": self.last_debug.get("left"),
            "observed_right": self.last_debug.get("right"),
            "observed_left_confidence": self.last_debug.get("left_confidence", 0.0),
            "observed_right_confidence": self.last_debug.get("right_confidence", 0.0),
            "source_timestamp": self._source_timestamp,
            "age_s": (max(0.0, time.monotonic() - self._source_timestamp) if self._source_timestamp > 0 else None),
            "ocr_latency_s": self._ocr_latency_s,
        }

    def _update_sync(self, frame, now=None, force_confirm=False, bypass_interval=False):
        now = time.monotonic() if now is None else float(now)
        previous_stable = self.last_score

        if not self.cfg.enabled or (not bypass_interval and now - self.last_ocr < self.cfg.interval_s):
            return self._empty(previous_stable)

        self.last_ocr = now
        if not self.tesseract_available():
            if not self._warned:
                print(
                    "[HUD] Tesseract engine not found. Install Tesseract-OCR "
                    "or set TESSERACT_CMD/TESSERACT_PATH. "
                    "pytesseract import alone is not sufficient."
                )
                self._warned = True
            self.confidence = 0.0
            return self._empty(previous_stable)

        try:
            left, right, pair_conf = self._ocr_score_pair(
                self.score_pair_crop(frame), self.tesseract_path
            )
            if left is None or right is None:
                left, lc = self.ocr_number(self.crop(frame, self.cfg.left_score_roi))
                right, rc = self.ocr_number(self.crop(frame, self.cfg.right_score_roi))
            else:
                lc = rc = pair_conf
            _clock_s, clock_conf, clock_text = self._ocr_clock(
                self.crop(frame, self.cfg.clock_roi)
            )
        except Exception as exc:
            if not self._warned:
                print(f"[HUD] score OCR failed: {type(exc).__name__}: {exc}")
                self._warned = True
            self.confidence = 0.0
            return self._empty(previous_stable)

        self.last_debug = {
            "left": left,
            "right": right,
            "left_confidence": lc,
            "right_confidence": rc,
            "clock_text": clock_text,
            "clock_s": _clock_s,
            "invalid_transition": False,
        }
        if _clock_s is not None:
            self.clock_s = float(_clock_s)

        if left is None or right is None:
            self.confidence *= 0.95
            return self._empty(previous_stable)

        candidate = (int(left), int(right))
        self.confidence = float(min(lc, rc))
        changed = False

        # Temporal score sanity: after initialization, the HUD is only allowed
        # to advance by exactly one goal on exactly one side. A one-frame OCR
        # jump such as 0:0 -> 2:0 is rejected and can never become a stable score.
        plausible = True
        if self.initialized:
            dl = candidate[0] - self.last_score[0]
            dr = candidate[1] - self.last_score[1]
            plausible = (dl >= 0 and dr >= 0 and dl + dr <= 1 and not (dl and dr))
        if not plausible:
            self.invalid_transition_count += 1
            self.pending_score = None
            self.pending_count = 0
            result = self._empty(previous_stable)
            result["observed_left"] = candidate[0]
            result["observed_right"] = candidate[1]
            result["observed_left_confidence"] = lc
            result["observed_right_confidence"] = rc
            result["invalid_transition"] = True
            return result

        required_confirm = 1 if force_confirm else self.cfg.confirm_frames
        if not self.initialized:
            if candidate == self.pending_score:
                self.pending_count += 1
            else:
                self.pending_score = candidate
                self.pending_count = 1
            if self.pending_count >= required_confirm:
                self.last_score = candidate
                self.initialized = True
                self.pending_count = 0
        elif candidate != self.last_score:
            if candidate == self.pending_score:
                self.pending_count += 1
            else:
                self.pending_score = candidate
                self.pending_count = 1
            if self.pending_count >= self.cfg.confirm_frames:
                self.last_score = candidate
                changed = True
                self.pending_count = 0
        else:
            self.pending_score = candidate
            self.pending_count = 0

        result = self._empty(previous_stable)
        result["observed_left"] = candidate[0]
        result["observed_right"] = candidate[1]
        result["observed_left_confidence"] = lc
        result["observed_right_confidence"] = rc
        result["score_changed"] = changed
        return result


    def _worker(self):
        while not self._stop.is_set():
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                time.sleep(0.005)
                continue
            frame, source_timestamp, force_confirm = item
            started = time.monotonic()
            info = self._update_sync(frame, now=started, force_confirm=force_confirm, bypass_interval=True)
            completed = time.monotonic()
            with self._lock:
                self._source_timestamp = source_timestamp
                self._completed_at = completed
                self._ocr_latency_s = completed - started
                self.last_debug["source_timestamp"] = source_timestamp
                self.last_debug["age_s"] = max(0.0, completed - source_timestamp)

    def update(self, frame, now=None, force_confirm=False):
        now = time.monotonic() if now is None else float(now)
        if not self.async_mode or force_confirm:
            info = self._update_sync(frame, now=now, force_confirm=force_confirm)
            self.last_debug["source_timestamp"] = now
            self.last_debug["age_s"] = 0.0
            return info
        if not self.cfg.enabled:
            return self._empty(self.last_score)
        if now - self.last_ocr >= self.cfg.interval_s:
            self.last_ocr = now
            with self._lock:
                if self._pending is not None:
                    # latest-frame semantics; replace pending OCR input only.
                    self._pending = (np.array(frame, copy=True), now, False)
                else:
                    self._pending = (np.array(frame, copy=True), now, False)
        with self._lock:
            age = None if self._source_timestamp <= 0 else max(0.0, now - self._source_timestamp)
        info = self._empty(self.last_score)
        info["source_timestamp"] = self._source_timestamp
        info["age_s"] = age
        info["ocr_latency_s"] = self._ocr_latency_s
        return info

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
