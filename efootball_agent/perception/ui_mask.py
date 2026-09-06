from __future__ import annotations

import cv2
import numpy as np


class UIMask:
    def __init__(self, cfg):
        self.rects = {
            "scoreboard": cfg.scoreboard,
            "minimap": cfg.minimap,
            "bottom_left_card": cfg.bottom_left_card,
            "bottom_right_card": cfg.bottom_right_card,
            "top_left_icon": cfg.top_left_icon,
        }

    @staticmethod
    def expand(rect, px, py):
        x1, y1, x2, y2 = rect
        return x1 - px, y1 - py, x2 + px, y2 + py

    @staticmethod
    def contains(point, rect):
        x, y = map(float, point)
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def reason(self, point):
        for name, rect in self.rects.items():
            if self.contains(point, self.expand(rect, 0.01, 0.01)):
                return name
        return None

    def is_ui(self, point):
        return self.reason(point) is not None

    def field_score(self, frame, point):
        x, y = map(float, point)
        if not (0 <= x <= 1 and 0 <= y <= 1) or self.is_ui(point):
            return 0.0
        h, w = frame.shape[:2]
        xi = int(np.clip(x * w, 0, w - 1))
        yi = int(np.clip(y * h, 0, h - 1))
        r = max(5, int(min(h, w) * 0.012))
        xa, xb = max(0, xi - r), min(w, xi + r + 1)
        ya, yb = max(0, yi - r), min(h, yi + r + 1)
        patch = frame[ya:yb, xa:xb]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        green = (H >= 25) & (H <= 100) & (S >= 30) & (V >= 40)
        return float(green.mean())
