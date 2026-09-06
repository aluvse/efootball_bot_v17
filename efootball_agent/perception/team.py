from __future__ import annotations

import cv2
import numpy as np


class TeamClassifier:
    """Appearance heuristic: position is deliberately not a team feature."""

    def __init__(self, cfg):
        self.cfg = cfg

    def classify(self, frame, det):
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        xa, xb = int(max(0, x1*w)), int(min(w, max(x1*w+1, x2*w)))
        ya = int(max(0, (y1 + 0.18*(y2-y1))*h))
        yb = int(min(h, max(ya+1, (y1 + 0.72*(y2-y1))*h)))
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            return "UNKNOWN", 0.0, np.zeros(6, np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        dark = float(((V <= self.cfg.black_v_max) & (S <= 200)).mean())
        bright = float(((V >= self.cfg.white_v_min) & (S <= self.cfg.white_sat_max)).mean())
        red = float((((H <= 12) | (H >= 170)) & (S >= 90) & (V >= 50)).mean())
        appearance = np.array([
            dark,
            bright,
            float(V.mean())/255,
            float(S.mean())/255,
            red,
            float(((H >= 25) & (H <= 100) & (S >= 30) & (V >= 40)).mean()),
        ], np.float32)

        black_score = dark + 0.20 * red
        white_score = bright
        margin = self.cfg.team_margin

        if black_score >= self.cfg.black_dark_fraction and black_score > white_score + margin:
            return "OUR", float(np.clip(0.55 + (black_score-white_score)*0.8, 0, .99)), appearance
        if white_score >= self.cfg.white_bright_fraction and white_score > black_score + margin:
            return "OPP", float(np.clip(0.55 + (white_score-black_score)*0.8, 0, .99)), appearance
        return "UNKNOWN", 0.35, appearance
