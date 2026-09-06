from __future__ import annotations

import cv2
import numpy as np

from .schema import PlayerDetection
from .ui_mask import UIMask


class PitchFootpointValidator:
    def __init__(self, cfg, ui: UIMask):
        self.cfg = cfg
        self.ui = ui

    def validate(self, frame, det: PlayerDetection):
        x1, y1, x2, y2 = det.bbox
        bw = x2 - x1
        bh = y2 - y1
        if not (self.cfg.min_bbox_w <= bw <= self.cfg.max_bbox_w):
            return False, "size_rejected"
        if not (self.cfg.min_bbox_h <= bh <= self.cfg.max_bbox_h):
            return False, "size_rejected"
        aspect = bw / max(bh, 1e-6)
        if aspect > getattr(self.cfg, "max_player_aspect", 0.72):
            return False, "size_rejected"
        foot = det.footpoint
        if self.ui.is_ui(foot):
            return False, "ui_rejected"
        if foot[1] > self.cfg.field_y_max:
            return False, "pitch_rejected"
        field_score = self.ui.field_score(frame, foot)
        if field_score < self.cfg.footpoint_green_min:
            return False, "pitch_rejected"
        return True, None


class PlayerDetector:
    def __init__(self, cfg, ui):
        self.cfg = cfg
        self.validator = PitchFootpointValidator(cfg, ui)
        self.ui = ui

    def validate_model(self, frame, detections):
        accepted = []
        metrics = {
            "raw_person": len(detections),
            "recovery_candidates": 0,
            "pitch_validated": 0,
            "ui_rejected": 0,
            "size_rejected": 0,
            "pitch_rejected": 0,
            "player_candidates": 0,
        }
        for det in detections:
            ok, reason = self.validator.validate(frame, det)
            if ok:
                accepted.append(det)
                metrics["pitch_validated"] += 1
            elif reason:
                metrics[reason] += 1
        metrics["player_candidates"] = len(accepted)
        return accepted, metrics

    def cv_recovery(self, frame):
        if not self.cfg.enable_cv_recovery:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark = (
            (hsv[:, :, 2] <= self.cfg.black_v_max)
            & (hsv[:, :, 1] <= 190)
        ).astype(np.uint8) * 255
        white = (
            (hsv[:, :, 2] >= self.cfg.white_v_min)
            & (hsv[:, :, 1] <= self.cfg.white_sat_max)
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_or(dark, white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        out = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 14 or bh > int(h * 0.40) or bw < 5 or bw > int(w * 0.15):
                continue
            aspect = bw / max(1.0, bh)
            # Reject long horizontal pitch lines / UI strips. Players are
            # vertically elongated even when the bbox is loose.
            if aspect > min(2.2, getattr(self.cfg, "max_player_aspect", 0.72)) or aspect < 0.10:
                continue
            bbox = (x/w, y/h, (x+bw)/w, (y+bh)/h)
            det = PlayerDetection(
                center=((bbox[0]+bbox[2])*.5, (bbox[1]+bbox[3])*.5),
                bbox=bbox,
                source="CV_RECOVERY",
                confidence=0.20,
            )
            ok, _reason = self.validator.validate(frame, det)
            if ok:
                out.append(det)
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out[: int(self.cfg.cv_recovery_max)]
