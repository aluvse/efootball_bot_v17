from __future__ import annotations

import numpy as np


class Predictor:
    def __init__(self, horizons):
        self.horizons = tuple(float(h) for h in horizons)

    def point(self, position, velocity):
        position = np.asarray(position, np.float32)
        velocity = np.asarray(velocity, np.float32)
        return {
            f"{h:.2f}": np.clip(position + velocity*h, 0, 1).astype(np.float32)
            for h in self.horizons
        }
